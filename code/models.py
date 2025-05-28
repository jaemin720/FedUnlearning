import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_hidden)
        self.layer2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNFashion_Mnist(CNNMnist):
    # Fashion MNIST와 구조 동일, 필요하면 별도 구현 가능
    pass

    
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# ResNet은 일반적으로 torchvision 모델을 사용하거나 직접 구현 가능
# 예시로 간단한 ResNet18 불러오기

from torchvision.models import resnet18

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.model = resnet18(pretrained=False)
        # CIFAR같은 작은 이미지에 맞게 첫 conv 수정 필요할 수 있음
        self.model.fc = nn.Linear(self.model.fc.in_features, args.num_classes)

    def forward(self, x):
        return self.model(x)
