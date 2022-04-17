import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
# K.gradients()을 진행할 때 "tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead" 에러가 나오는 경우에 사용해줍니다.
tf.compat.v1.disable_eager_execution()
from keras import models
from keras.applications.vgg16 import VGG16
# VGG16을 로드합니다.
model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
img_path = 'example\\human.jpg'
output = 'example\\human_cam.jpg'

# 이미지를 불러오고, VGG16의 입력 픽셀인 224 x 224로 사이즈를 변환합니다.
img = image.load_img(img_path, target_size=(224, 224))
x= image.img_to_array(img)
x=np.expand_dims(x, axis=0)

# 이미지의 채널별 평균 값으로 전처리합니다.
x=preprocess_input(x)

# 이미지를 예측합니다.
preds = model.predict(x)
print('Predicted Top 3: ', decode_predictions(preds, top=3)[0])
print('Predicted: ', decode_predictions(preds, top=3)[0][0][1])

# VGG16의 이미지 분류 인덱스 번호를 가져옵니다.
index = np.argmax(preds[0])
print('Predicted Index: ', index)

# 예측된 분류 인덱스 번호로 해당 출력 게층을 가져오기 위해 준비합니다.
predict_output = model.output[:, index]

last_conv_layer = model.get_layer('block5_conv3')
grads= K.gradients(predict_output, last_conv_layer.output)[0]
pooled_grads= K.mean(grads, axis=(0,1,2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *=pooled_grads_value[i]
    
# heatmap을 구성합니다.
heatmap = np.mean(conv_layer_output_value, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.imshow(heatmap)

# heatmap을 visualization합니다.
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 +img
cv2.imwrite(output, superimposed_img)

