import os
import time
import copy
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet
from utils import get_dataset, average_weights, exp_details
from UNGAN import UnGANTrainer
from mia import evaluate_mia

def select_model(args, train_dataset):
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            return CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            return CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            return CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        input_dim = np.prod(img_size)
        return MLP(dim_in=input_dim, dim_hidden=64, dim_out=args.num_classes)
    elif args.model == 'resnet':
        return ResNet(args=args)
    else:
        raise ValueError(f"Error: Unrecognized model {args.model}")

def distill_global_model(global_model, generator, retain_idxs, dataset, device, epochs=1, lr=0.001):
    global_model.train()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)

    loader = DataLoader(Subset(dataset, retain_idxs), batch_size=64, shuffle=True)
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                teacher_logits = generator(x)
                teacher_probs = F.softmax(teacher_logits, dim=1)
            student_logits = global_model(x)
            student_log_probs = F.log_softmax(student_logits, dim=1)
            loss = loss_fn(student_log_probs, teacher_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():
    start_time = time.time()
    args = args_parser()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./unlearning_result', exist_ok=True)
    logger = SummaryWriter('./logs')

    exp_details(args)

    train_dataset, test_dataset, user_groups, unseen_idxs, shadow_idxs = get_dataset(args)
    global_model = select_model(args, train_dataset).to(device)

    if args.load_model is not None and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        global_model.load_state_dict(torch.load(args.load_model, map_location=device))
        global_weights = global_model.state_dict()
    else:
        print("Training from scratch.")
        global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    print_every = 2

    for epoch in tqdm(range(args.epochs), desc='Global Training Rounds'):
        print(f'\n | Global Training Round : {epoch + 1} |')

        local_weights, local_losses = [], []
        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        selected_users = np.random.choice(range(args.num_users), m, replace=False)

        for user_idx in selected_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[user_idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        avg_loss = sum(local_losses) / len(local_losses)
        train_loss.append(avg_loss)

        global_model.eval()
        accuracies = []
        for user_idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[user_idx], logger=logger)
            acc, _ = local_model.inference(global_model)
            accuracies.append(acc)
        avg_acc = sum(accuracies) / len(accuracies)
        train_accuracy.append(avg_acc)

        if (epoch + 1) % print_every == 0:
            print(f'\nAvg Training Stats after {epoch + 1} rounds:')
            print(f'Training Loss: {avg_loss:.4f}')
            print(f'Train Accuracy: {100 * avg_acc:.2f}%\n')

        logger.add_scalar('Loss/train', avg_loss, epoch)
        logger.add_scalar('Accuracy/train', avg_acc, epoch)

        if (epoch + 1) % args.untrain_rounds == 0:
            print("\n[UnGAN] Unlearning Triggered!")
            snapshot_path = f'./saved_models/snapshot_round_{epoch+1}.pth'
            torch.save(global_model.state_dict(), snapshot_path)

            G_hat = select_model(args, train_dataset).to(device)
            G_hat.load_state_dict(torch.load(snapshot_path, map_location=device))
            G_hat.eval()

            target_user = np.random.choice(list(user_groups.keys()))
            print(f"[UnGAN] Target user for unlearning: {target_user}")

            ungan = UnGANTrainer(args, train_dataset, user_groups, G_hat,
                                 unseen_idxs=unseen_idxs, target_user=target_user)
            log_path = f"./logs/ungan_log_{epoch+1}.json"
            un_model_path = f"./saved_models/unlearn/ungan_generator_{epoch+1}.pth"
            ungan.train(epochs=args.untrain_epochs, log_path=log_path)
            ungan.save_generator(un_model_path)

            print("\n[Distillation] Updating global model using generator...")
            retain_idxs = [idx for uid, idxs in user_groups.items() if uid != target_user for idx in idxs]
            distill_global_model(global_model, ungan.generator, retain_idxs, train_dataset, device)

            print("\n[MIA] Starting MIA Evaluation...")
            mia_path = f"./unlearning_result/mia_result_{epoch+1}.json"  # 🔹 이 줄 추가
            evaluate_mia(
                model=global_model,
                dataset=train_dataset,
                target_user_idx=target_user,
                forget_idxs=user_groups[target_user],
                retain_idxs=retain_idxs,
                shadow_idxs=shadow_idxs,
                device=device,
                save_path=mia_path,
                args=args,
                k=10
            )



    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f'\nResults after {args.epochs} global rounds of training:')
    print(f"|---- Avg Train Accuracy: {100 * train_accuracy[-1]:.2f}%")
    print(f"|---- Test Accuracy: {100 * test_acc:.2f}%")

    if args.save_model is not None:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(global_model.state_dict(), args.save_model)
        print(f"Saved trained model to {args.save_model}")

    logger.close()
    print(f"\nTotal Run Time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
