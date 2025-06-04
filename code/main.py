import os
import time
import copy
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Subset

from torch.utils.data import ConcatDataset #unseen data를 통해서 언러닝 재학습에서 사용.
from torch.utils.data import TensorDataset

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNMnist, Generator, Discriminator, generate_images, filter_images
from utils import get_dataset, average_weights, exp_details, create_poisoned_dataset
from unlearn import (
    train_generator_ungan,
    SyntheticImageDataset,
    partition_synthetic_data_iid,
    get_synthetic_subset
)
from evaluate_mia import evaluate_mia

def move_dataset_to_device(dataset, device):
    images = []
    labels = []
    for x, y in dataset:
        images.append(x.to(device))
        labels.append(torch.tensor(y).to(device))
    return TensorDataset(torch.stack(images), torch.stack(labels))


def add_backdoor_trigger(x):
    x_bd = x.clone()
    x_bd[:, 25:28, 25:28] = 0.9
    return x_bd

def evaluate_backdoor_asr(model, dataset, target_label, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            # 백도어 트리거 삽입
            x_bd = add_backdoor_trigger(x).to(device)
            x_bd = x_bd.unsqueeze(0)  # 배치 차원 추가

            output = model(x_bd)
            pred = output.argmax(dim=1).item()

            total += 1
            if pred == target_label:
                correct += 1

    asr = correct / total
    return asr

def select_model(args, train_dataset):
    if args.model == 'cnn':
        return CNNMnist(args=args)
    else:
        raise NotImplementedError


def main():
    
    args = args_parser()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    exp_details(args)

    # ===================== 1. 데이터셋 로딩 및 초기화 =====================
    train_dataset, test_dataset,unseen_dataset, user_groups = get_dataset(args)
    

    full_dataset, user_groups = create_poisoned_dataset(train_dataset, user_groups, args,
                                                        malicious_client=0,
                                                        target_label=6,
                                                        poison_ratio=0.8)

    global_model = select_model(args, full_dataset).to(device)
    global_model.train()

    generator = Generator(z_dim=args.z_dim).to(device)
    discriminator = Discriminator().to(device)

    global_weights = global_model.state_dict()
    train_loss, train_accuracy = [], []

    forget_client = 0
    forget_idxs = user_groups[forget_client]
    retain_idxs = [i for i in range(len(train_dataset)) if i not in forget_idxs]
    test_idxs = np.random.choice(len(test_dataset), len(forget_idxs), replace=False)

    # ===================== 2. 연합 학습 (FedAvg) =====================
    for epoch in tqdm(range(args.epochs), desc='Global Training Rounds'):
        print(f'\n| Global Training Round : {epoch + 1} |')

        local_weights, local_losses = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            if epoch == args.epochs - 1 and idx == forget_client:
                continue  # 마지막 라운드에 삭제 유저 제외

            local_model = LocalUpdate(args=args, dataset=full_dataset, idxs=user_groups[idx])
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
    print("[MIA] Evaluating membership inference attack...")
    
    all_idxs = set(range(len(full_dataset)))
    non_member_candidates = list(all_idxs - set(forget_idxs))
    #여기에서 쉐도우에는 forget 데이터가 없도록 하기.
    mia_result = evaluate_mia(
        model=global_model,
        dataset=full_dataset,
        test_dataset= test_dataset,
        forget_idxs=forget_idxs,
        retain_idxs=test_idxs,
        shadow_idxs=np.random.choice(non_member_candidates, len(forget_idxs), replace=False),
        device=device,
        save_path="./mia_result_before.json"
    )

    print(f"[MIA] AUC: {mia_result['auc']:.4f}")
    print("\n[Backdoor Attack Success Rate Evaluation]")
    target_label = 6  # 공격 대상 라벨 (main() 함수와 맞춰야 함)
    asr = evaluate_backdoor_asr(global_model, test_dataset, target_label, device)
    print(f"Backdoor Attack Success Rate (ASR): {asr*100:.2f}%")

    torch.save(global_model.state_dict(), args.save_model)
    print(f"[Saved] model to {args.save_model}\n")
    # ===================== 재학습 언러닝 비교하기 ===================
    



    # ===================== 3. UNGAN 기반 Generator 학습 =====================
    
    start_time = time.time()
    generator = train_generator_ungan(
        generator=generator,
        discriminator=discriminator,
        dataset=full_dataset,
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
    synthetic_imgs, synthetic_labels = generate_images(generator, forget_idxs, full_dataset, device=device, z_dim=args.z_dim)
    filtered_imgs, filtered_labels = filter_images(discriminator, synthetic_imgs, synthetic_labels, threshold=args.gen_threshold, device=device)
                                              

    if len(filtered_imgs) < args.num_users:
        print(f"[Unlearning] Filtered images insufficient ({len(filtered_imgs)}) for unlearning. Skipping.")
        return

    synthetic_dataset = SyntheticImageDataset(filtered_imgs, filtered_labels)
    unseen_dataset = move_dataset_to_device(unseen_dataset, device)

    combined_dataset = ConcatDataset([synthetic_dataset, unseen_dataset])
    #syn_user_groups = partition_synthetic_data_iid(synthetic_dataset, args.num_users)
    syn_user_groups = partition_synthetic_data_iid(combined_dataset, args.num_users)
    

    # ===================== 5. Synthetic 데이터 기반 재학습 =====================
    updated_weights = []
    for idx in range(args.num_users):
        synthetic_subset = get_synthetic_subset(combined_dataset, syn_user_groups, idx)
        local_model = LocalUpdate(args=args, dataset=synthetic_subset)
        w, _ = local_model.update_weights(model=copy.deepcopy(global_model), global_round=args.epochs)
        updated_weights.append(copy.deepcopy(w))

    global_weights = average_weights(updated_weights)
    global_model.load_state_dict(global_weights)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Unlearning Time: {elapsed_time:.2f}초")
    test_acc_after, test_loss_after = test_inference(args, global_model, test_dataset)
    print(f"\n[Test After Unlearning] Accuracy: {test_acc_after*100:.2f}% | Loss: {test_loss_after:.4f}")

    # ===================== 6. MIA 평가 =====================
    print("[MIA] Evaluating membership inference attack...")

    all_idxs = set(range(len(full_dataset)))
    non_member_candidates = list(all_idxs - set(forget_idxs))

    mia_result = evaluate_mia(
        model=global_model,
        dataset=full_dataset,
        test_dataset = test_dataset,
        forget_idxs=forget_idxs,
        retain_idxs=test_idxs,
        shadow_idxs=np.random.choice(non_member_candidates, len(forget_idxs), replace=False),
        device=device,
        save_path="./mia_result.json"
    )

    print(f"[MIA] AUC: {mia_result['auc']:.4f}")
    asr = evaluate_backdoor_asr(global_model, test_dataset, target_label, device)
    print(f"Backdoor Attack Success Rate (ASR): {asr*100:.2f}%")

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
    #print('\nTotal Run Time: {:.2f} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    main()