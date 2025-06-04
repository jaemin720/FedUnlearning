import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset


class LocalUpdate:
    def __init__(self, args, dataset, idxs=None):
        self.args = args
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

        # Dataset 설정 (real or synthetic 모두 가능)
        if idxs is not None:
            self.train_loader = DataLoader(Subset(dataset, idxs),
                                           batch_size=self.args.local_bs,
                                           shuffle=True)
        else:
            self.train_loader = DataLoader(dataset,
                                           batch_size=self.args.local_bs,
                                           shuffle=True)

        self.criterion = nn.CrossEntropyLoss()

    def update_weights(self, model, global_round):
        model.to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for epoch in range(self.args.local_ep):
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
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        test_loss /= len(self.train_loader.dataset)
        accuracy = correct / len(self.train_loader.dataset)

        return accuracy, test_loss


def test_inference(args, model, test_dataset):
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False)

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