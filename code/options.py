import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # 기본 파라미터
    parser.add_argument('--epochs', type=int, default=10, help="Number of global training rounds")
    parser.add_argument('--num_users', type=int, default=10, help="Number of users/clients")
    parser.add_argument('--frac', type=float, default=0.5, help="Fraction of clients used per round")
    parser.add_argument('--local_ep', type=int, default=5, help="Local epochs")
    parser.add_argument('--local_bs', type=int, default=32, help="Local batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")

    # 데이터셋 및 모델 선택
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'cifar'], help='Dataset name')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp', 'resnet'], help='Model architecture')

    # 기타 옵션
    parser.add_argument('--iid', type=int, default=1, help="IID or Non-IID data distribution (1 for IID, 0 for Non-IID)")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, set to -1 for CPU")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of output classes")

    # 모델 학습 결과 저장 및 로드
    parser.add_argument('--load_model', type=str, default=None,
                    help="Path to a saved model to resume training from")
    parser.add_argument('--save_model', type=str, default='./saved_models/model.pth',
                    help="Path to save the trained model")

        # Generator 관련 파라미터 (필요 시)
    parser.add_argument('--z_dim', type=int, default=100, help="Dimension of noise vector for Generator")
    parser.add_argument('--gen_threshold', type=float, default=0.7, help="Discriminator filtering threshold")
    parser.add_argument('--num_gen_samples', type=int, default=128, help="Number of generated samples per client")



    args = parser.parse_args()
    args.gpu = args.gpu >= 0

    return args
