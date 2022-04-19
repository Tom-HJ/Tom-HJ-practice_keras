# 기본 매개변수 설정
Nin = 784
Nh_l = [500, 500, 100, 50]
number_of_class = 10
Nout = number_of_class

# DNN 모델 구현
from keras import layers, models
class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nh_l[2], activation='relu', name='Hidden-3'))
        self.add(layers.Dense(Nh_l[3], activation='relu', name='Hidden-4'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
# 데이터 준비
import numpy as np
import tensorflow as tf
from keras.utils import np_utils

def Data_func():
    mnist = tf.keras.datasets.mnist # 로컬에서 데이터를 로드하고 싶을 때 https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    
    L, H, W = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return (X_train, Y_train), (X_test, Y_test)

# 데이터 표현 준비
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

# 학습 및 성능평가
import matplotlib.pyplot as plt

def main():
    model = DNN(Nin, Nh_l, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    
    history = model.fit(X_train, Y_train, epochs=50, batch_size=500, validation_split=0.2)
    performace_test = model.evaluate(X_test, Y_test, batch_size=500)
    print("Test Loss and Accuracy ->", performace_test)
    
    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()
    
# 프로세스 실행
if __name__ == '__main__':
    main()