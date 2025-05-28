import os
import copy
import torch
import json
from update import LocalUpdate
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, ResNet
from utils import get_dataset, average_weights
from options import args_parser


class FedUnlearning:
    def __init__(self, args, logger, file_manager):
        self.args = args
        self.logger = logger
        self.file_manager = file_manager
        self.device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

        # 데이터셋과 유저 그룹 로드
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(args)

        # 모델 초기화
        self.global_model = self._select_model()
        self._load_saved_model()

    def _select_model(self):
        if self.args.model == 'cnn':
            if self.args.dataset == 'mnist':
                return CNNMnist(args=self.args).to(self.device)
            # elif self.args.dataset == 'fmnist':
            #     return CNNFashion_Mnist(args=self.args).to(self.device)
            elif self.args.dataset == 'cifar':
                return CNNCifar(args=self.args).to(self.device)
        elif self.args.model == 'mlp':
            input_dim = torch.prod(torch.tensor(self.train_dataset[0][0].shape)).item()
            return MLP(dim_in=input_dim, dim_hidden=64, dim_out=self.args.num_classes).to(self.device)
        elif self.args.model == 'resnet':
            return ResNet(args=self.args).to(self.device)
        else:
            raise ValueError(f"Invalid model type: {self.args.model}")

    def _load_saved_model(self):
        if self.args.load_model and os.path.exists(self.args.load_model):
            print(f"Loading model from {self.args.load_model}")
            self.global_model.load_state_dict(torch.load(self.args.load_model, map_location=self.device))
        else:
            raise FileNotFoundError(f"No model found at {self.args.load_model}")

    def unlearn_user(self, target_user):
        print(f">>> Unlearning for user: {target_user}")

        remaining_users = [u for u in range(self.args.num_users) if u != target_user]
        local_weights, local_losses = [], []

        # 1 라운드만 진행
        print(f'\n | Unlearning Training Round : 1 |')

        for user in remaining_users:
            print(f'   |- Local Update from user {user}')
            local_model = LocalUpdate(
                args=self.args,
                dataset=self.train_dataset,
                idxs=self.user_groups[user],
                logger=self.logger,
                override_ep=self.args.unlearn_epochs,
                override_lr=self.args.unlearn_lr,
                override_momentum=self.args.unlearn_momentum
            )
            w, loss = local_model.update_weights(model=copy.deepcopy(self.global_model), global_round=0)
            print(f'      ↪ Loss: {loss:.4f}')
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)

        # 평균 후 모델 적용
        updated_global_weights = average_weights(local_weights)
        self.global_model.load_state_dict(updated_global_weights)

        avg_loss = sum(local_losses) / len(local_losses)
        print(f'\nAvg Unlearning Stats after 1 round:')
        print(f'Unlearning Loss: {avg_loss:.4f}')
        print(">>> Unlearning complete.")



    def evaluate(self, return_result=False):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0

        backdoor_correct = 0
        backdoor_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.global_model(images)
                loss = criterion(outputs, labels).item()

                total_loss += loss * labels.size(0)
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

                # Backdoor trigger: 좌상단 픽셀 하얗게
                for i in range(len(images)):
                    images[i][:, 0, 0] = 1.0
                backdoor_outputs = self.global_model(images)
                backdoor_preds = backdoor_outputs.argmax(1)
                backdoor_correct += (backdoor_preds == 0).sum().item()
                backdoor_total += labels.size(0)

        acc = correct / total
        loss = total_loss / total
        asr = backdoor_correct / backdoor_total

        print(f"Test Accuracy after unlearning: {acc*100:.2f}%, Loss: {loss:.4f}, ASR: {asr*100:.2f}%")
        print(f"Backdoor ASR: {asr*100:.2f}%")

        if return_result:
            return acc, loss, asr


    def save_unlearned_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.global_model.state_dict(), path)
        print(f"Unlearned model saved to {path}")

def save_unlearning_log(output_path, user_id, acc, loss, args):
        log = {
           "target_user": user_id,
            "test_accuracy": round(acc * 100, 2),
            "test_loss": round(loss, 4),
            "unlearn_epochs": args.unlearn_epochs,
            "unlearn_lr": args.unlearn_lr,
            "unlearn_momentum": args.unlearn_momentum
    }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(log, f, indent=4)
        print(f"Unlearning log saved to {output_path}")


if __name__ == '__main__':
    args = args_parser()
    args.load_model = 'code/saved_models/pre-trained/model.pth'
    args.save_model = 'code/saved_models/unlearn/unlearned_model.pth'

    logger = None
    file_manager = None

    fed_unlearner = FedUnlearning(args, logger, file_manager)

    # 1. Unlearning 실행
    fed_unlearner.unlearn_user(target_user=args.target_user)

    # 2. 평가 결과 반환
    acc, loss, asr = fed_unlearner.evaluate(return_result=True)

    # 3. 로그 저장
    log_path = 'logs/unlearning_result.json'
    save_unlearning_log(log_path, args.target_user, acc, loss, args)

    # 4. 모델 저장
    fed_unlearner.save_unlearned_model(args.save_model)
