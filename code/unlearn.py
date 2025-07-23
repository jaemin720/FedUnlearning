# unlearn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np
from torch.utils.data import Dataset


# -------------------- UNGAN Generator 학습 --------------------
def train_generator_ungan(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                          lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10):
    """
    단순 adversarial loss 기반 UNGAN Generator 학습 (KL 제거)
    """
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    print(f"[UNGAN] Training Generator (adversarial only) for {epochs} epochs...")

    generator.train()
    for epoch in range(epochs):
        for _ in range(len(forget_idxs) // batch_size):
            z = torch.randn((batch_size, z_dim), device=device)
            gen_imgs = generator(z)

            # Adversarial loss: log(D(G(z))) → G가 D를 속이도록 유도
            d_preds = discriminator(gen_imgs)
            adv_loss = -torch.mean(torch.log(d_preds + 1e-8))

            g_optimizer.zero_grad()
            adv_loss.backward()
            g_optimizer.step()

    print("[UNGAN] Generator training completed.\n")
    return generator

def train_gd_ungan(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                          lambda_adv=1.0, z_dim=100, batch_size=64, epochs=10):
    """
    UNGAN Generator & Discriminators 동시 학습 (adversarial loss 포함)
    """
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    criterion = nn.BCELoss()  # Binary cross entropy loss (GAN에서 주로 사용)

    # Forget 데이터셋의 실제 이미지들을 위한 DataLoader 준비
    forget_subset = torch.utils.data.Subset(dataset, forget_idxs)
    forget_loader = torch.utils.data.DataLoader(forget_subset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"[UNGAN] Training Generator and Discriminator for {epochs} epochs...")

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for real_imgs, _ in forget_loader:
            real_imgs = real_imgs.to(device)

            # ---------------------
            # 1. Discriminator 학습
            # ---------------------
            d_optimizer.zero_grad()

            # 진짜 이미지에 대해 1로 레이블 지정
            real_labels = torch.ones((real_imgs.size(0), 1), device=device)
            # 가짜 이미지에 대해 0으로 레이블 지정
            fake_labels = torch.zeros((real_imgs.size(0), 1), device=device)

            # Discriminator가 진짜 이미지에 대해 잘 맞추도록
            real_preds = discriminator(real_imgs)
            d_loss_real = criterion(real_preds, real_labels)

            # 가짜 이미지 생성
            z = torch.randn((real_imgs.size(0), z_dim), device=device)
            fake_imgs = generator(z)

            # Discriminator가 가짜 이미지에 대해 잘 맞추도록
            fake_preds = discriminator(fake_imgs.detach())  # detach로 G의 그래프 분리
            d_loss_fake = criterion(fake_preds, fake_labels)

            # Discriminator 전체 손실 및 역전파
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ---------------------
            # 2. Generator 학습
            # ---------------------
            g_optimizer.zero_grad()

            # Generator는 Discriminator를 속여야 하므로 진짜(1) 레이블로 학습
            fake_preds = discriminator(fake_imgs)
            g_loss = criterion(fake_preds, real_labels)

            g_loss.backward()
            g_optimizer.step()

        print(f"[UNGAN] Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    print("[UNGAN] Generator and Discriminator training completed.\n")
    return generator, discriminator

def train_gd_ungan_with_unseen(generator, discriminator, dataset, retain_idxs, forget_idxs, device,
                   lambda_adv, z_dim, batch_size, epochs, unseen_dataset=None):
    """
    UNGAN Generator & Discriminator 학습
    → Discriminator는 Unseen+Forget 데이터를 모두 Real로 학습
    → Generator는 adversarial loss + (optional) unseen similarity loss 사용
    """
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    criterion = nn.BCELoss()  # GAN용 이진 분류 손실

    # Forget + Unseen 데이터 로더 구성
    forget_subset = torch.utils.data.Subset(dataset, forget_idxs)

    if unseen_dataset is not None:
        real_dataset = torch.utils.data.ConcatDataset([forget_subset, unseen_dataset])
    else:
        real_dataset = forget_subset

    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"[UNGAN] Training Generator and Discriminator for {epochs} epochs...")

    generator.train()
    discriminator.train()
    if unseen_dataset is not None:
        unseen_loader = torch.utils.data.DataLoader(unseen_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        unseen_iter = iter(unseen_loader)
    for epoch in range(epochs):
        for real_imgs, _ in real_loader:
            real_imgs = real_imgs.to(device)

            # ---------------------
            # 1. Discriminator 학습
            # ---------------------
            d_optimizer.zero_grad()

            real_labels = torch.ones((real_imgs.size(0), 1), device=device)
            fake_labels = torch.zeros((real_imgs.size(0), 1), device=device)

            real_preds = discriminator(real_imgs)
            d_loss_real = criterion(real_preds, real_labels)

            z = torch.randn((real_imgs.size(0), z_dim), device=device)
            fake_imgs = generator(z)

            fake_preds = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_preds, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ---------------------
            # 2. Generator 학습
            # ---------------------
            g_optimizer.zero_grad()
            fake_preds = discriminator(fake_imgs)
            g_loss_adv = criterion(fake_preds, real_labels)

            sim_loss = 0.0
            if unseen_dataset is not None:
                try:
                    unseen_imgs, _ = next(unseen_iter)
                except StopIteration:
                    unseen_iter = iter(unseen_loader)
                    unseen_imgs, _ = next(unseen_iter)

                unseen_imgs = unseen_imgs.to(device)
                min_len = min(fake_imgs.size(0), unseen_imgs.size(0))
                sim_loss = F.mse_loss(fake_imgs[:min_len], unseen_imgs[:min_len])

            g_loss = g_loss_adv + lambda_adv * sim_loss
            g_loss.backward()
            g_optimizer.step()

        print(f"[UNGAN] Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Sim Loss: {sim_loss:.4f}")

    print("[UNGAN] Generator and Discriminator training completed.\n")
    return generator, discriminator


# -------------------- Synthetic Dataset 정의 --------------------
class SyntheticImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# -------------------- IID 분배 --------------------
def partition_synthetic_data_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    indices = np.random.permutation(len(dataset))
    user_groups = {}

    for i in range(num_users):
        user_groups[i] = indices[i * num_items:(i + 1) * num_items].tolist()

    return user_groups


# -------------------- Subset 추출 --------------------
def get_synthetic_subset(dataset, user_groups, user_idx):
    return Subset(dataset, user_groups[user_idx])
