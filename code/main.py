import os
import time
import copy
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Subset

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNMnist, Generator, Discriminator, generate_images, filter_images
from utils import get_dataset, average_weights, exp_details
from unlearn import (
    train_generator_ungan,
    SyntheticImageDataset,
    partition_synthetic_data_iid,
    get_synthetic_subset
)
from evaluate_mia import evaluate_mia


def select_model(args, train_dataset):
    if args.model == 'cnn':
        return CNNMnist(args=args)
    else:
        raise NotImplementedError


def main():
    start_time = time.time()
    args = args_parser()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    exp_details(args)

    # ===================== 1. 데이터셋 로딩 및 초기화 =====================
    train_dataset, test_dataset, user_groups = get_dataset(args)
    global_model = select_model(args, train_dataset).to(device)
    global_model.train()

    generator = Generator(z_dim=args.z_dim).to(device)
    discriminator = Discriminator().to(device)

    global_weights = global_model.state_dict()
    train_loss, train_accuracy = [], []

    forget_client = 0
    forget_idxs = user_groups[forget_client]

    # ===================== 2. 연합 학습 (FedAvg) =====================
    for epoch in tqdm(range(args.epochs), desc='Global Training Rounds'):
        print(f'\n| Global Training Round : {epoch + 1} |')

        local_weights, local_losses = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            if epoch == args.epochs - 1 and idx == forget_client:
                continue  # 마지막 라운드에 삭제 유저 제외

            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        acc, _ = test_inference(args, global_model, test_dataset)
        train_loss.append(loss_avg)
        train_accuracy.append(acc)

        print(f"Training Loss: {loss_avg:.4f} | Train Accuracy: {acc*100:.2f}%")

    test_acc_before, test_loss_before = test_inference(args, global_model, test_dataset)
    print(f"\n[Test Before Unlearning] Accuracy: {test_acc_before*100:.2f}% | Loss: {test_loss_before:.4f}")

    torch.save(global_model.state_dict(), args.save_model)
    print(f"[Saved] model to {args.save_model}\n")

    # ===================== 3. UNGAN 기반 Generator 학습 =====================
    retain_idxs = [i for i in range(len(train_dataset)) if i not in forget_idxs]

    generator = train_generator_ungan(
        generator=generator,
        discriminator=discriminator,
        dataset=train_dataset,
        retain_idxs=retain_idxs,
        forget_idxs=forget_idxs,
        device=device,
        lambda_adv=0.1,
        z_dim=args.z_dim,
        batch_size=64,
        epochs=10
    )


    # ===================== 4. Generator 이미지 생성 및 필터링 =====================
    # Generator 이미지 생성 및 필터링
    print("[Unlearning] Generating and filtering synthetic data...")
    synthetic_imgs, synthetic_labels = generate_images(generator, forget_idxs, train_dataset, device=device, z_dim=args.z_dim)
    filtered_imgs, filtered_labels = filter_images(discriminator, synthetic_imgs, synthetic_labels, threshold=args.gen_threshold, device=device)
                                              

    if len(filtered_imgs) < args.num_users:
        print(f"[Unlearning] Filtered images insufficient ({len(filtered_imgs)}) for unlearning. Skipping.")
        return

    synthetic_dataset = SyntheticImageDataset(filtered_imgs, filtered_labels)
    syn_user_groups = partition_synthetic_data_iid(synthetic_dataset, args.num_users)

    # ===================== 5. Synthetic 데이터 기반 재학습 =====================
    updated_weights = []
    for idx in range(args.num_users):
        synthetic_subset = get_synthetic_subset(synthetic_dataset, syn_user_groups, idx)
        local_model = LocalUpdate(args=args, dataset=synthetic_subset)
        w, _ = local_model.update_weights(model=copy.deepcopy(global_model), global_round=args.epochs)
        updated_weights.append(copy.deepcopy(w))

    global_weights = average_weights(updated_weights)
    global_model.load_state_dict(global_weights)

    test_acc_after, test_loss_after = test_inference(args, global_model, test_dataset)
    print(f"\n[Test After Unlearning] Accuracy: {test_acc_after*100:.2f}% | Loss: {test_loss_after:.4f}")

    # ===================== 6. MIA 평가 =====================
    print("[MIA] Evaluating membership inference attack...")

    mia_result = evaluate_mia(
        model=global_model,
        dataset=train_dataset,
        forget_idxs=forget_idxs,
        retain_idxs=retain_idxs,
        shadow_idxs=np.random.choice(len(train_dataset), len(forget_idxs), replace=False),
        device=device,
        save_path="./mia_result.json"
    )

    print(f"[MIA] AUC: {mia_result['auc']:.4f}")

    # ===================== 7. 결과 저장 =====================
    result_json = {
        "test_acc_before": test_acc_before,
        "test_loss_before": test_loss_before,
        "test_acc_after": test_acc_after,
        "test_loss_after": test_loss_after,
        "mia_auc": mia_result['auc']
    }

    with open('./results_unlearning.json', 'w') as f:
        json.dump(result_json, f, indent=4)

    print("[Saved] results_unlearning.json & mia_result.json")
    print('\nTotal Run Time: {:.2f} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
