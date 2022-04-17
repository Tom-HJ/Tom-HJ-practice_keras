import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras import models
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
img_path = 'example\\human.jpg'
output = 'example\\result.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x= image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x=preprocess_input(x)

preds = model.predict(x)
print('Predicted Top 3: ', decode_predictions(preds, top=3)[0])
print('Predicted: ', decode_predictions(preds, top=3)[0][0][1])

index = np.argmax(preds[0]) #386 -> elephant
print('Predicted Index: ', index)

african_elepant_output = model.output[:, index]

last_conv_layer = model.get_layer('block5_conv3')
grads= K.gradients(african_elepant_output, last_conv_layer.output)[0]
pooled_grads= K.mean(grads, axis=(0,1,2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *=pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis = -1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 +img
cv2.imwrite(output, superimposed_img)

