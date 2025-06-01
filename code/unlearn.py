# unlearn.py

import torch
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
