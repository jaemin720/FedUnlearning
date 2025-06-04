import torch
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset

from torch.utils.data import random_split #안 본 데이터 만들기.
#----백도어----
from torch.utils.data import Dataset
import random

class CustomDataset(Dataset):
    def __init__(self, data):
        """
        data: list of (image_tensor, label) tuples
        """
        self.data = data
        self.targets = torch.tensor([label for _, label in data])  # targets 속성 추가

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def add_backdoor_trigger(x):
    x_bd = x.clone()
    # 채널이 1개일 때, 이미지 마지막 부분에 2x2 크기의 밝은 점 추가
    x_bd[:, 25:28, 25:28] = 0.9
    return x_bd


def create_poisoned_dataset(train_dataset, user_groups, args, malicious_client=0, target_label=6, poison_ratio=0.1):
    # 1. 악성 클라이언트 인덱스 중 일부만 백도어로 선택
    malicious_idxs = user_groups[malicious_client]
    num_poison = int(len(malicious_idxs) * poison_ratio)
    poisoned_idxs = set(random.sample(malicious_idxs, num_poison))  # 순서 상관없음, lookup 빠름

    full_data = []

    for i in range(len(train_dataset)):
        x, y = train_dataset[i]
        if i in poisoned_idxs:
            x = add_backdoor_trigger(x)
            y = target_label
        full_data.append((x, y))

    full_dataset = CustomDataset(full_data)
    return full_dataset, user_groups

# -------------------- 데이터셋 로딩 --------------------
def get_dataset(args):
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
        train_dataset, unseen_dataset = random_split(train_dataset, [55000, 5000])

    elif args.dataset == 'cifar':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    user_groups = partition_data(train_dataset, args)
    return train_dataset, test_dataset,unseen_dataset, user_groups


# -------------------- 데이터셋 분할 --------------------
def partition_data(dataset, args):
    num_items = int(len(dataset) / args.num_users)
    user_groups = {}

    if args.iid:
        idxs = np.random.permutation(len(dataset))
        for i in range(args.num_users):
            user_groups[i] = idxs[i * num_items:(i + 1) * num_items].tolist()
    else:
        labels = dataset.targets.numpy()
        idxs = np.argsort(labels)
        shards_per_user = 2
        num_shards = args.num_users * shards_per_user
        shard_size = len(dataset) // num_shards
        shards = [idxs[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

        user_groups = {i: [] for i in range(args.num_users)}
        for i in range(args.num_users):
            assigned_shards = shards[i * shards_per_user:(i + 1) * shards_per_user]
            for shard in assigned_shards:
                user_groups[i] += shard.tolist()

    return user_groups


# -------------------- 가중치 평균 --------------------
def average_weights(w_list):
    avg_weights = copy.deepcopy(w_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(w_list)):
            avg_weights[key] += w_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(w_list))
    return avg_weights


# -------------------- 실험 설정 출력 --------------------
def exp_details(args):
    print("===== Experiment Settings =====")
    print(f"Model           : {args.model}")
    print(f"Dataset         : {args.dataset}")
    print(f"Num Clients     : {args.num_users}")
    print(f"Fraction        : {args.frac}")
    print(f"IID             : {args.iid}")
    print(f"Local Epochs    : {args.local_ep}")
    print(f"Batch Size      : {args.local_bs}")
    print(f"Learning Rate   : {args.lr}")
    print(f"Generator z_dim : {args.z_dim}")
    print(f"Disc. Threshold : {args.gen_threshold}")
    print("===============================")


# -------------------- Synthetic Dataset 클래스 --------------------
class SyntheticImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# -------------------- Synthetic 데이터 IID 분배 --------------------
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