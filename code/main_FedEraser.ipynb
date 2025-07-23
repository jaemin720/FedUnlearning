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
from models import CNNMnist, Generator, Discriminator, generate_images, filter_images, select_model
from utils import get_dataset, average_weights, exp_details, create_poisoned_dataset, get_transform, SyntheticImageDataset
from unlearn import (
    train_generator_ungan,
    partition_synthetic_data_iid,
    get_synthetic_subset
)



def move_dataset_to_device(dataset, device):
    images = []
    labels = []
    for x, y in dataset:
        images.append(x.to(device))
        labels.append(y.to(device) if isinstance(y, torch.Tensor) else torch.tensor(y).to(device))
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


def main():
    args = args_parser()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    exp_details(args)

    # ===================== 1. 데이터셋 로딩 및 초기화 =====================
    train_dataset, test_dataset,unseen_dataset, user_groups = get_dataset(args)
    

    full_dataset, user_groups = create_poisoned_dataset(train_dataset, user_groups, args,
                                                        malicious_client=0,
                                                        target_label=6,
                                                        poison_ratio=0.0)

    global_model = select_model(args, full_dataset).to(device)
    global_model.train()

    if args.dataset == 'mnist':
        img_shape = (1, 28, 28)
    elif args.dataset == 'cifar':
        img_shape = (3, 32, 32)
    else:
        raise ValueError("Unsupported dataset")

    generator = Generator(z_dim=args.z_dim, img_shape=img_shape).to(device)
    discriminator = Discriminator(img_shape=img_shape).to(device)

    #generator = Generator(z_dim=args.z_dim).to(device)
    #discriminator = Discriminator().to(device)

    global_weights = global_model.state_dict()
    train_loss, train_accuracy = [], []

    forget_client = 0
    forget_idxs = user_groups[forget_client]
    forget_dataset = Subset(full_dataset, forget_idxs)
    retain_idxs = [i for i in range(len(train_dataset)) if i not in forget_idxs]
    test_idxs = np.random.choice(len(test_dataset), len(forget_idxs), replace=False)

    # ===================== 2. 연합 학습 (FedAvg) =====================
    for epoch in tqdm(range(args.epochs), desc='Global Training Rounds'):
        print(f'\n| Global Training Round : {epoch + 1} |')

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        local_weights, local_losses, local_deltas = [], [], {}

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=full_dataset, idxs=user_groups[idx])
            w, loss, delta = local_model.FedErase_update_weights(model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

            # FedEraser: 델타 저장
            if idx not in local_deltas:
                local_deltas[idx] = []
            local_deltas[idx].append(copy.deepcopy(delta))  # 라운드별로 쌓기

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

    print("\n[Backdoor Attack Success Rate Evaluation]")
    target_label = 6  # 공격 대상 라벨 (main() 함수와 맞춰야 함)
    asr = evaluate_backdoor_asr(global_model, test_dataset, target_label, device)
    print(f"Backdoor Attack Success Rate (ASR): {asr*100:.2f}%")

    acc_forget, _ = test_inference(args, global_model, forget_dataset)
    print(f"[Forget Set Accuracy after Unlearning] {acc_forget:.4f}")

    torch.save(global_model.state_dict(), args.save_model)
    print(f"[Saved] model to {args.save_model}\n")
    # ===================== 재학습 언러닝 비교하기 ===================
    def reverse_update(model, delta):
        model_state = model.state_dict()
        for key in model_state.keys():
           model_state[key] -= delta[key]
        model.load_state_dict(model_state)   



    # ===================== FedEraser 기반 언러닝 =====================
    print("\n[Unlearning] Reversing client updates using FedEraser...")

    if forget_client not in local_deltas:
        print(f"[FedEraser] No updates found for forget_client {forget_client}. Skipping.")
    else:
        for delta in local_deltas[forget_client]:
            reverse_update(global_model, delta)

        print("[FedEraser] Finished reversing updates for forget client.")

    # FedEraser 적용 후 테스트 정확도
    test_acc_after, test_loss_after = test_inference(args, global_model, test_dataset)
    print(f"\n[Test After FedEraser] Accuracy: {test_acc_after*100:.2f}% | Loss: {test_loss_after:.4f}")
        # =============== Forget Accuracy =======================
    acc_forget, _ = test_inference(args, global_model, forget_dataset)
    print(f"[Forget Set Accuracy after Unlearning] {acc_forget:.4f}")
    # 예: FedEraser 적용 후 global_model에 반영 완료되었을 때
    fed_eraser_model_path = "./saved_models/unlearned_model_federaser.pth"

    torch.save(global_model.state_dict(), fed_eraser_model_path)

    print(f"[Saved] FedEraser unlearned model to {fed_eraser_model_path}")


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
    unseen_dataset = move_dataset_to_device(unseen_dataset, device)
    synthetic_imgs, synthetic_labels = generate_images(generator, forget_idxs, full_dataset, device=device, z_dim=args.z_dim)
    #filtered_imgs, filtered_labels = filter_images(discriminator, synthetic_imgs, synthetic_labels, threshold=args.gen_threshold, device=device)       

    #if len(filtered_imgs) < args.num_users:
    #    print(f"[Unlearning] Filtered images insufficient ({len(filtered_imgs)}) for unlearning. Skipping.")
    #    return

    #synthetic_dataset = SyntheticImageDataset(filtered_imgs, filtered_labels)
    syn_transform = get_transform(args.dataset)
    #synthetic_labels = torch.tensor(synthetic_labels)
    synthetic_dataset = SyntheticImageDataset(synthetic_imgs, synthetic_labels, transform=syn_transform, device=device)

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

    asr = evaluate_backdoor_asr(global_model, test_dataset, target_label, device)
    print(f"Backdoor Attack Success Rate (ASR): {asr*100:.2f}%")

    # =============== Forget Accuracy =======================
    acc_forget, _ = test_inference(args, global_model, forget_dataset)
    print(f"[Forget Set Accuracy after Unlearning] {acc_forget:.4f}")

    # ===================== 7. 결과 저장 =====================
    result_json = {
        "test_acc_before": test_acc_before,
        "test_loss_before": test_loss_before,
        "test_acc_after": test_acc_after,
        "test_loss_after": test_loss_after,
    }

    with open('./results_unlearning.json', 'w') as f:
        json.dump(result_json, f, indent=4)

    print("[Saved] results_unlearning.json & mia_result.json")
    #print('\nTotal Run Time: {:.2f} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
