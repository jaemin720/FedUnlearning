import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
import numpy as np

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class LocalUpdate:
    def __init__(self, args, dataset, idxs, logger=None,
                 override_ep=None, override_lr=None, override_momentum=None,
                 client_id=None, unlearn_flag=False):
        self.args = args
        self.logger = logger
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

        self.local_ep = override_ep if override_ep is not None else args.local_ep
        self.lr = override_lr if override_lr is not None else args.lr
        self.momentum = override_momentum if override_momentum is not None else args.momentum

        self.client_id = client_id
        self.unlearn_flag = unlearn_flag

        self.train_loader = DataLoader(Subset(dataset, idxs),
                                       batch_size=self.args.local_bs,
                                       shuffle=True)

        if self.unlearn_flag:
            self.inject_backdoor(self.train_loader.dataset, attack_portion=1.0)

        self.criterion = nn.CrossEntropyLoss()

    def inject_backdoor(self, dataset, attack_portion=1.0):
        print(f"   ↪ Injecting backdoor into client {self.client_id}'s data...")
        for i in range(len(dataset)):
            if np.random.rand() < attack_portion:
                image, label = dataset[i]
                image = transform_to_image(image)
                image.putpixel((0, 0), 255)  # 좌상단 픽셀 트리거
                dataset[i] = (transform_to_tensor(image), 0)  # 타겟 클래스 0

    def update_weights(self, model, global_round):
        model.to(self.device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        epoch_loss = []

        for epoch in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        model.to(self.device)
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        test_loss /= len(self.train_loader.dataset)
        accuracy = correct / len(self.train_loader.dataset)

        return accuracy, test_loss


# 전역 테스트 함수 (평가용으로 분리)
def test_inference(args, model, test_dataset):
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return accuracy, test_loss
