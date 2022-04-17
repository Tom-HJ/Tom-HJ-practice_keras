# practice_keras   
작성자: 김현준(010-4890-5240 / ggkids9211@gmail.com)   

이 레퍼지토리는 케라스 입문을 위해 생성되었습니다.   
이곳은 다음의 케라스 기능을 연습합니다.   
1. Stochastic Gradient Descent, SGD 연습.   
2. Artificial Neural Network, ANN 연습.

* * *   
## 세부 기능 정리    
### 1. Stochastic Gradient Descent, SGD    
> 확률적 경사 하강법이라고도 합니다.     
> 이것은 추출된 데이터 한개에 대해서 그래디언트를 계산하고, 경사 하강 알고리즘을 적용하는 것입니다.   
> 특징으로는 전체 데이터를 사용하는 것이 아니라 랜덤하게 추출한 일부 데이터를 학습에 사용하기 때문에, 학습의 중간과정에서 결과의 진폭이 크고 불안정하며, 학습 속도가 매우 빠른 것입니다.   
> 또한 데이터 하나씩 처리하기 때문에 오차율이 매우 크며, GPU의 성능을 모두 사용하지 못하는 단점이 있습니다.    
> 이러한 단점을 보완하기 위해 나온 방법은 Mini Batch를 이용한 방법이며 SGD보다 노이즈를 줄이면서도 전체 배치를 더 효율적으로 학습하는 것으로 알려져 있습니다.   
> 이렇게 경사 하강법에도 몇가지 계산 방법이 있는데 크게는 이 세가지 입니다.   
> 1. Batch: 모든 데이터를 한꺼번에 학습하는 방법입니다. 부드럽게 학습되는 것이 특징이나 샘플 개수 만큼 계산해야하기 때문에 시간이 다소 걸립니다.   
> 2. Stochastic: 데이터를 랜덤으로 추출하여 학습해보고, 이를 모든 학습 데이터에 적용해보는 계산 방법입니다. 위에서도 언급했듯이 학습속도는 매우 빠르나 학습 중간과정의 진폭이 크고 불안정합니다.   
> 3. Mini Batch: 전체 학습 데이터를 배치 사이즈로 나눠서 순차적으로 학습합니다. 일반적인 딥러닝에 사용되는 방법이며 Batch 보다 학습이 빠르고 SGD 보다 낮은 오차율을 가지고 있습니다.     
   
예제 링크: [SGD 예제 코드](https://github.com/Tom-HJ/Tom-HJ-practice_keras/blob/main/0_SGD.py)

### 2. Artificial Neural Network, ANN   
> 인공신경망이라고 불리며 생체의 신경망을 흉내 낸 인공지능입니다.     
> 입력, 은닉, 출력 계층으로 구성되어 있으며 은닉 계층을 한 개 이상 포함할 수 있습니다.     
> 또한 각 계층은 여러 노드로 구성됩니다. ANN은 넓은 의미로 신경망을 총칭하는 용어로 사용되기도 해서 단일 은닉 계층의 ANN을 얕은 신경망(Shallow Neural Network, SNN)으로 구분해서 부르기도 합니다.     
> 이 신경망은 신경망 개발 역사 초기에 처리할 데이터 양이 늘어나거나 비정형 데이터라 복잡도가 높아지는 경우에 활용되었습니다.     
> ANN은 다음과 같이 4단계로 동작합니다.    
> 1단계: 입력계층을 통해 들어온 데이터(x)에 가중치 행렬(W_xh)를 곱하여 은닉계층으로 보냅니다.          
> 2단계: 은닉 계층의 각 노드는 자신에게 입력된 신호 벡터에 활성화 함수(Activation Function, f_h())를 적용한 결과값 (h)로 내보냅니다. 뉴런의 동작을 흉내내고 비선형성을 보상하는 활성화 함수로 시그모이드, 하이퍼볼릭탄젠트 함수 등을 사용합니다.      
> 3단계: 은닉 계층의 결과값에 새로운 가중치 행렬 W_hy를 곱한 뒤 출력 계층으로 보냅니다.       
> 4단계: 출력 계층으로 들어온 값에 출력 활성화 함수인 f_y()를 적용하고 그 결과인 y를 신경망 외부로 최종 출력합니다. 분류의 경우 출력용 활성화 함수로 소프트맥스(Softmax)를 주로 사용합니다.         
>        
> ANN의 기본적인 활용에는 분류와 회귀로 나눌 수 있습니다. 분류 ANN은 입력 정보를 클래스별로 분류하는 방식이며, 회귀 ANN은 입력 정보로 다른 값을 예측하는 방식입니다.       
   
예제 링크: [ANN 예제 코드](https://github.com/Tom-HJ/Tom-HJ-practice_keras/blob/main/1_ANN.py)

### 9. Class Activation Map, CAM   
> 가끔 ConvNet의 의사결정과정을 Visualization을 해야할 때가 있는데, 이때 어느부분이 ConvNet의 최종 분류 결정에 기여하였는가를 Debugging을 할 때 사용합니다.   
> 또 이미지의 특정 물체 위치를 파악할 때도 사용됩니다.   
> 결국 CAM은 특정 클래스 출력에 대해 입력 이미지의 모든 위치를 계산한 2D 점수 그리드를 Heatmap으로 표현한 것으로 각 위치가 얼마나 중요한지를 말해줍니다.   
> CAM의 자세한 설명은 다음 논문을 참조하세요.   
> 링크: [Visual Explanaitions from Deep Networks via Gradient based Localization](https://arxiv.org/pdf/1610.02391.pdf)   
   
예제 링크: [CAM 예제 코드](https://github.com/Tom-HJ/Tom-HJ-practice_keras/blob/main/9_CAM.py)   