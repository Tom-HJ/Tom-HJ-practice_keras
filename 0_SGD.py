import keras
import numpy as np

# X array를 만듭니다.
x = np.array([0, 1, 2, 3, 4, 5])
# 우리는 y = 2x + 1의 함수를 SGD에 학습시켜 볼 것입니다.
y = x * 2 + 1
# Y array를 확인합니다.
print(y)


# 케라스로 인공신경망 모델을 만들기 시작합니다.
model = keras.models.Sequential()
'''
인공지능 계층을 추가합니다.
추가한 인공지능 계층은 입력 노드 하나와 가중치 하나를 가지는 선형처리계층입니다.
케라스의 계층은 내부적으로 편향값을 가지고 있으므로 미지수 둘을 포함하는 셈입니다.
여기서 입력 계층의 노드 수 input_shape의 지정에 의해 자동으로 생성됩니다.
'''
model.add(keras.layers.Dense(1, input_shape=(1,)))
# 만들 모델을 확률적 경사하강법(SGD)으로 학습시키고 손실 함수는 평균제곱오차(Mean Squared Error, mse)를 사용하여 컴파일합니다.
model.compile('SGD', 'mse')

'''
각 배열의 두개의 샘플 데이터를 가지고 학습을 시작합니다.

여기서 epochs는 학습을 진행하는 총 에포크를 의미합니다. 
에포크(epoch)는 인공신경망을 학습할 때 데이터 전체가 사용된 한회 또는 한 세대를 의미하며 1 epoch는 전체 학습 데이터 셋이 한 신경망에 적용되어 순전파와 역전파를 통해 신경망을 한번 통과하였다는 의미입니다.
에포크 값을 높일수록 다양한 무작위 가중치로 학습해보는 것이므로 적합한 파라미터를 찾을 확률이 올라갑니다 (= 손실값이 내려감.)
단 너무 에포크 값을 높인다면 그 학습 데이터셋에 과적합되어 다른 데이터에 대해선 제대로 된 예측을 하지 못할 수 있습니다.

verbose는 학습 진행 상황의 표시 여부를 정합니다. (0 = 표시하지 않음, 1 = 표시함.)
'''
model.fit(x[:2], y[:2], epochs=1000, verbose=0)

'''
이제 학습이 잘 되었는지 확인합니다.
flatten()은 매트리스 출력값을 벡터 형태로 바꿔주는 기능을 합니다.
'''
y_pred = model.predict(x[2:]).flatten()


print('Tagets:', y[2:])
print('Predictions:', y_pred)
print('Errors:', y[2:] - y_pred)
