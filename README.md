# FedUnlearning
졸업 프로젝트 2025

code 폴더안에 있는 main을 실행하면, default 옵션에 맞춘 학습이 진행됩니다. (다양한 실험을 위해서 arguments를 직접 수정하여 실행할 수도 있습니다.)
여기서의 실험 세팅은 데이터셋 : MNIST / 모델 : 간단한 CNN / 클라이언트 수 : 10명 / 언러닝할 클라이언트 : index 0(첫 번째 클라이언트) 외에 다른 args는 수정 가능하나 앞서 언급된 데이터셋 / 모델 / 언러닝 요청 클라이언트의 정보가 수정될 경우에 정상적으로 코드가 작동하지 않을 수 있습니다. (각 데이터셋에 맞게끔 분할이 안 되거나, 언러닝 클라이언트를 선택하는 것은 코드 내용 안에 있어서 그 부분을 수정해야하며, 언러닝 효과에 대한 평가지표가 정확하지 않을 수 있습니다.)

<img width="862" alt="스크린샷 2025-06-02 오후 4 48 51" src="https://github.com/user-attachments/assets/2a87b569-8716-49ed-abb7-da319a663067" />


<img width="681" alt="스크린샷 2025-06-02 오후 10 27 01" src="https://github.com/user-attachments/assets/9c2ad587-c757-4750-bf43-c36595d062c3" />

실행 흐름
--
사용할 데이터셋을 args를 통해서 기입하면, utils.py를 통해서 데이터셋을 연합학습에 맞추어 클라이언트별로 로컬데이터를 분할합니다.
get_dataset()함수와 partition_data()함수를 통해서 클라이언트 수에 맞추어 데이터가 분할하게 됩니다. 로컬 데이터는 원본 데이터셋의 인덱스를 클라이언트 인덱스에 랜덤으로 부여해주어서 주게 됩니다.
현재 IID한 것으로 가정을 하였기에 전체 데이터셋에 클래스 별 데이터 수와 로컬 데이터 수는 비슷하게 분할하게 됩니다.

이후 Backdoor Attack을 통해서 언러닝 효과를 검증하기 위해 언러닝 클라이언트에 백도어 데이터를 삽입하게 됩니다. utils.py의 create_poisoned_dataset()에서 이뤄지게 됩니다.
다시 메인 함수에서 돌아가며 데이터셋을 global_model에 같이 로드합니다.(모델은 앞서 args를 통해서 정해진 모델로 models.py에서 갖고 오게 됩니다.)

언러닝을 위해 generator와 discriminator를 준비합니다.(이는 언러닝 전 아무 때나 해도 상관은 없었습니다.) -models.py에 정의된 GAN 모델
이후 연합학습을 시작합니다. 각 로컬 업데이트를 진행하고, 로컬의 결과를 취합하여 글로벌 모델에 반영하고 다시 글로벌 모델이 각 로컬들에게 글로벌 모델을 주면서 라운드가 1번 진행됩니다. 이렇게 args를 통해 지정한 local 학습 수와 라운드 수에 맞추어서 학습을 진행합니다.

args에는 frac을 통해서 실제 연합학습처럼 매 라운드마다 클라이언트가 학습에 참여할 수도 있고, 아닐수도 있도록 합니다.(실험에서는 모든 클라이언트가 학습에 참여하게 됩니다. frac=1)

연합학습을 진행한 후에는 언러닝 후와 비교하기 위해서 언러닝 클라이언트 데이터셋, 남은 데이터셋, 아직 학습에 사용되지 않은 데이터셋을 통해서 MIA(Membership Inference Attack)을 하고, 그에 대한 평가를 진행합니다.
Retain(유지해야할 데이터셋)과 Forget(잊어야할 데이터셋)의 쉐도우 모델이 출력하는 확신도와 AUC를 구하게 됩니다. 여기서 언러닝 시 확신도는 Forget은 줄어들어야 하며, AUC는 0.5에 가까워지거나 이전보다 줄어들면 언러닝이 잘 된 것 입니다.

이후 백도어 어택에 대한 ASR(Attack Success Rate)을 구합니다.(Test Set을 동일한 Backdoor Poison 방식으로 데이터의 트리거를 정하고 학습된 모델이 이 트리거에 맞추어 잘못된 클래스로 분류하면 성공한 것으로 맞춘 것과 못 맞춘 것으로 ASR을 계산합니다.)

언러닝 부분
--
train_generator_ungan()을 이용하여 GAN 모델(Generator: G / Discriminator : D)을 학습합니다. -unlearn.py
여기서는 단순하게 Dataset을 기반으로 G와 D를 학습하는 것이 아니라, Adversarial Loss를 통해서 G가 D를 속이는 것에만 목적을 두어 노이즈를 주도록 합니다.

다시 메인 함수에서 Forget 데이터셋(언러닝 요청 클라이언트의 데이터셋)을 GAN을 통해서 데이터를 생성/합성 합니다.
이후 더 정밀하게 합성 결과를 얻기 위해서 Discriminator를 통해서 통과한 것만 언러닝 데이터셋으로 활용하게 됩니다. filter_images()

Forget Data를 합성한 Data인 Synthetic_dataset과 언러닝 시에만 사용하는 서버의 공유 데이터 Unseen Data를 병합합니다.
Unseen Data는 아직 글로벌 모델이 학습하지 않은 데이터셋입니다.

다시 해당 데이터셋을 통해서 글로벌 모델을 재학습하면서 Fine-tuning하게 됩니다. (언러닝 재학습의 에폭 수도 수정할 수 있습니다. 실험에서는 10번으로 하였습니다.)

이후 다시 동일하게 언러닝을 평가합니다.

평가 결과
--
컴퓨팅 자원 : 
CPU : Intel64 Family 6 Model 151 Stepping 2 GenuineIntel ~ 3600Mhz
GPU : GeForce RTX 3080
RAM : 32 Gb/ OS : Window 10 Home
Anaconda 활용 / Python 3.12.9 사용

실험 설정 :
Model : CNN 모델
DataSet : MNIST
IID하게 분리하며, Train / Test / Unseen을 11 : 2 : 1의 비율로 데이터셋 양이 정해졌습니다.
(Train - 55,000 개 / Test - 10,000 개 / Train에서 뽑아서 Unseen - 5,000 개)

클라이언트 수 : 10명 (매 학습마다 전체 참여 - Frac=1)
언러닝 요청 클라이언트는 1명, 첫 번 째 클라이언트 / 언러닝 클라이언트의 백도어 데이터 및 정상 데이터 비율 (8:2)

연합학습 라운드 : 200
로컬 에폭 : 10 / 배치 사이즈 : 64 / Optimizer : SGD
언러닝 재학습은 10 Epochs

언러닝 이전
[Test Accuract] : 99.24 % / Loss : 0.0360
[MIA] :
Evaluate Retain Confidence mean : 0.99791354
Evaluate Forget Confidence mean : 0.99810106
ACU : 0.6122
[ASR] : 99.40 %

언러닝 이후
Test Accuracy : 99.06 % / Loss : 0.0322
[MIA] :
Evaluate Retain Confidence mean : 0.99680257
Evaluate Forget Confidence mean : 0.9634214
AUC : 0.7868
[ASR] : 22.03 %
언러닝 시간 : 7.05초

재학습 (언러닝 클라이언트 제외하고 학습)
Test Accuracy : 99.08%
시간 : 1876.33초
[MIA] :
Evaluate Retain Confidence mean : 0.9978992
Evaluate Forget Confidence mean : 0.9983963
AUC : 0.5014
[ASR] : 9.50 %

재학습에서의 MIA 평가는 글로벌 모델이 Forget Dataset에 대해서 학습을 해야 평가가 가능한데, 재학습에서는 아예 없었기 때문에 MIA 평가로 비교하는 것은 부적합할 수 있습니다.(Forget Confidence Mean이 높은 이유)
