import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json
from utils import average_weights


class Generator(nn.Module):
    def __init__(self, logits_dim=100):  # 100 = num_classes
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, logits_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten image
        return self.model(x)



class Discriminator(nn.Module):
    def __init__(self, logits_dim=100):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(logits_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class UnGANTrainer:
    def __init__(self, args, train_dataset, user_groups, G_hat, unseen_idxs=None, target_user=None):
        self.args = args
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

        self.G_hat = G_hat.to(self.device).eval()

        # G, D 초기화
        img_shape = train_dataset[0][0].shape
        self.G = Generator(logits_dim=args.num_classes).to(self.device)
        self.generator = self.G
        self.D = Discriminator(logits_dim=args.num_classes).to(self.device)

        self.criterion_kd = nn.MSELoss()
        self.criterion_adv = nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.args.untrain_lr)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.args.untrain_lr)

        self.target_user = target_user
        self.unseen_idxs = unseen_idxs

        # 데이터셋 분할
        self.dataloader_f, self.dataloader_u = self._prepare_data(train_dataset, user_groups)

    def _prepare_data(self, dataset, user_groups):
        target_user = self.target_user
        forget_idxs = user_groups[target_user]
        D_f = Subset(dataset, forget_idxs)

        retain_idxs = []
        for uid, idxs in user_groups.items():
            if uid != target_user:
                retain_idxs.extend(idxs)

        if self.unseen_idxs is not None:
            D_u = Subset(dataset, self.unseen_idxs)
        else:
            all_idxs = set(range(len(dataset)))
            seen_idxs = set(forget_idxs + retain_idxs)
            unseen_idxs = list(all_idxs - seen_idxs)
            D_u = Subset(dataset, unseen_idxs)

        return DataLoader(D_f, batch_size=64, shuffle=True), DataLoader(D_u, batch_size=64)

    def train(self, epochs=5, lambda_adv=0.01, log_path='../logs/ungan_train_log/unlearn_train.json'):
        self.G.train()
        self.D.train()

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        training_log = {
            "epochs": epochs,
            "lambda_adv": lambda_adv,
            "log": []
        }

        for epoch in range(epochs):
            total_kd_loss = 0.0
            total_adv_loss = 0.0
            total_d_loss = 0.0
            num_batches = 0

            for x_f, _ in self.dataloader_f:
                x_f = x_f.to(self.device)

                with torch.no_grad():
                    z_hat_f = self.G_hat(x_f)

                z_f = self.G(x_f)

                self.D.zero_grad()
                real_preds = self.D(z_hat_f)
                fake_preds = self.D(z_f.detach())

                real_loss = self.criterion_adv(real_preds, torch.ones_like(real_preds))
                fake_loss = self.criterion_adv(fake_preds, torch.zeros_like(fake_preds))
                loss_D = (real_loss + fake_loss) / 2
                loss_D.backward()
                self.optimizer_D.step()

                self.G.zero_grad()
                kd_loss = self.criterion_kd(z_f, z_hat_f)
                adv_loss = self.criterion_adv(self.D(z_f), torch.ones_like(fake_preds))
                loss_G = kd_loss + lambda_adv * adv_loss
                loss_G.backward()
                self.optimizer_G.step()

                total_kd_loss += kd_loss.item()
                total_adv_loss += adv_loss.item()
                total_d_loss += loss_D.item()
                num_batches += 1

            avg_kd = total_kd_loss / num_batches
            avg_adv = total_adv_loss / num_batches
            avg_d = total_d_loss / num_batches

            training_log["log"].append({
                "epoch": epoch + 1,
                "kd_loss": round(avg_kd, 6),
                "adv_loss": round(avg_adv, 6),
                "discriminator_loss": round(avg_d, 6)
            })

            print(f"[Epoch {epoch+1}] KD: {avg_kd:.4f} | Adv: {avg_adv:.4f} | D: {avg_d:.4f}")

        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=4)
        print(f"Training log saved to {log_path}")

    def evaluate(self):
        print("Evaluation method not implemented.")

    def save_generator(self, path='./saved_models/unlearn/ungan_generator.pth'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.G.state_dict(), path)
        print(f"Generator model saved to: {path}")
