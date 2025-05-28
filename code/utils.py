import torch
import copy
from torchvision import datasets, transforms
import numpy as np

def get_dataset(args):
    """데이터셋 로드 및 사용자 그룹 분할 (train / unseen / shadow 포함)"""
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        full_dataset = datasets.MNIST('../data/mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data/mnist', train=False, download=True, transform=transform)

    elif args.dataset == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.2860,), (0.3530,))])
        full_dataset = datasets.FashionMNIST('../data/fashion_mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('../data/fashion_mnist', train=False, download=True, transform=transform)

    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])
        full_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform)

    else:
        raise ValueError(f'Unsupported dataset {args.dataset}')

    train_idxs, unseen_idxs, shadow_idxs = split_dataset_custom(full_dataset)

    # user_groups는 학습 데이터 기준으로만 나눔
    user_groups = partition_data(train_idxs, args)

    return full_dataset, test_dataset, user_groups, unseen_idxs, shadow_idxs


def split_dataset_custom(dataset, train_ratio=0.8, unseen_ratio=0.1, seed=42):
    """전체 데이터를 train / unseen / shadow 비율로 나누기"""
    np.random.seed(seed)
    total_size = len(dataset)
    idxs = np.random.permutation(total_size)

    train_end = int(total_size * train_ratio)
    unseen_end = train_end + int(total_size * unseen_ratio)

    train_idxs = idxs[:train_end]
    unseen_idxs = idxs[train_end:unseen_end]
    shadow_idxs = idxs[unseen_end:]

    return train_idxs.tolist(), unseen_idxs.tolist(), shadow_idxs.tolist()


def partition_data(train_idxs, args):
    """train_idxs만을 사용해서 IID 또는 Non-IID로 유저 그룹 나누기"""
    num_items = int(len(train_idxs) / args.num_users)
    user_groups = {}

    if args.iid == 1:
        shuffled = np.random.permutation(train_idxs)
        for i in range(args.num_users):
            user_groups[i] = shuffled[i * num_items:(i + 1) * num_items].tolist()
    else:
        raise NotImplementedError("현재 Non-IID는 지원되지 않습니다. IID로 설정해주세요.")

    return user_groups


def average_weights(w_list):
    """가중치 리스트를 평균내는 함수"""
    avg_weights = copy.deepcopy(w_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(w_list)):
            avg_weights[key] += w_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(w_list))
    return avg_weights


def exp_details(args):
    print(f'Experimental details:')
    print(f'    Model       : {args.model}')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Number of users : {args.num_users}')
    print(f'    Fraction of clients  : {args.frac}')
    print(f'    IID         : {bool(args.iid)}')
    print(f'    Local epochs: {args.local_ep}')
    print(f'    Local batch size: {args.local_bs}')
    print(f'    Learning rate: {args.lr}')
    print(f'    GPU enabled : {args.gpu}')
