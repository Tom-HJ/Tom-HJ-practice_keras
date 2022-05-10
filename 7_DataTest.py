
# imports needed for CNN
import cv2
import os
from cv2 import imread
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# 학습 및 성능평가
import matplotlib.pyplot as plt
# print(check_output(["ls", "/input"]).decode("utf8"))
# DNN 모델 구현
from keras import layers, models

import numpy as np

# Load the data
def load_data(data_dir):
    """
    From: https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.v471kaepx
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []

    category = 0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
         # adding an early stop for sake of speed
        stop = 0
        for f in file_names:
            img = cv2.imread(f)
            imresize = cv2.resize(img, (200, 200))
            #im = np.array(imresize).reshape(1, 120000)
            #plt.imshow(imresize)
            imresize = cv2.Canny(imresize, 150, 150)
            im = np.array(imresize).reshape(40000)
            images.append(im)
            labels.append(category)
            # remove this to use full data set
            if stop > 30:
                          break
            stop += 1
            # end early stop
            
        category += 1

    return images, labels

# 기본 매개변수 설정
Nin = 40000
Nh_l = [500, 500, 100, 50]
number_of_class = 9
Nout = number_of_class

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dense(Nh_l[2], activation='relu', name='Hidden-3'))
        self.add(layers.Dense(Nh_l[3], activation='relu', name='Hidden-4'))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      
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

from keras.utils import np_utils

def main():
    
    data_dir = "input"
    images, labels = load_data(data_dir)
    
    x = np.array(images)
    y = np_utils.to_categorical(labels)
    #(x_train, y_train), (x_test, y_test) = train_test_split(x, y, test_size=0.2)
    
    model = DNN(Nin, Nh_l, Nout)
    history = model.fit(x, y, epochs=50, batch_size=500, validation_split=0.2)
    
    # performace_test = model.evaluate(x_test, y_test, batch_size=500)
    # print("Test Loss and Accuracy ->", performace_test)
    
    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()

# 프로세스 실행
if __name__ == '__main__':
    main()