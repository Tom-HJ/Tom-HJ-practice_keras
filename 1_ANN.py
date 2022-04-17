# 이번 프로젝트는 인공지능 구현 6단계를 따릅니다.
# 1단계: 분류 ANN 구현용 패키지 가져오기
# 2단계: 분류 ANN에 필요한 매개변수 설정
# 3단계: 분류 ANN 모델 구현
# 4단계: 학습과 성능 평가용 데이터 가져오기
# 5단계: 분류 ANN 학습 및 검증
# 6단계: 분류 ANN 학습 결과 분석

# 1단계: 분류 ANN 구현용 패키지 가져오기
from operator import mod
from keras import layers, models


# 기본 형태(분산 방식)의 분류 ANN 구현하기
class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')
        
        # 입력 계층을 layers.Input() 함수로 지정합니다. 원소를 Nin개 가지는 입력신호 벡터는 입력 노드에 따른 계층의 shape를 (Nin,)로 지정합니다.
        x = layers.Input(shape=(Nin,))
        # 노드가 Nh개인 은닉 계층은 layers.Dense(Nh)로 지정합니다. x를 입력노드로 받아들이고 싶으면 뒤에 (x)를 추가합니다. 
        h = relu(hidden(x))
        # 출력 계층은 다음과 같이 지정합니다. 출력 노드수는 Nout으로 지정합니다. 이때 출력 노드에 입력되는 정보는 은닉노드의 출력값입니다.
        # 분류의 경우는 출력 노드의 활성화 함수로 소프트맥스 연산을 수행합니다.
        y = softmax(output(h))
        
        super().__init__(x, y)
        # 컴파일은 다음과 같이 구현합니다.
        # loss는 손실함수를 지정하는 인자입니다. 케라스가 제공하는 손실함수 외에도 직접 새로운 손실함수를 지정할 수 있습니다.
        # optimizer는 최적화 함수를 지정합니다.
        # metrics는 학습이나 예측이 진행될 때 성능 검증을 위해 손실 뿐 아니라 정확도도 측정하라는 의미입니다.
        self.complie(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 연쇄 방식의 ANN 모델 구현하기
class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        

# 4단계: 학습과 성능 평가용 데이터 가져오기.
import tensorflow as tf
import numpy as np
from keras.utils import np_utils

# Keras Database에서 Mnist 데이터를 가져오고 가공하기.
def Data_func():
    mnist = tf.keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    
    L, H, W = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    return (X_train, Y_train), (X_test, Y_test)

# 분류 ANN 학습 결과의 그래프 구현
import matplotlib.pyplot as plt

def plot_loss(history, title = None):
    if not isinstance(history, dict):
        history = history.history
        
    plt.plot(history['loss'])
    plt. plot(history['val_loss'])
    
    if title is not None:
        plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['training', 'Validation'], loc=0)
    
def plot_acc(history, title = None):
    if not isinstance(history, dict):
        history = history.history
        
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if title is not None:
        plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc=0)
    
    
# 분류 ANN 학습 및 성능 분석
def main():
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class
    
    # model = ANN_models_class(Nin, Nh, Nout)
    model = ANN_seq_class(Nin, Nh, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    
    # 학습 시작
    history = model.fit(X_train, Y_train, epochs=5, batch_size=100, validation_split=0.2)
    performance_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('정확도 및 손실 테스트 -> ', performance_test)
    
    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()
    
# 프로세스 실행
if __name__ == '__main__':
    main()