import keras
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5])
y = x * 2 + 1

print(y)

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x[:2], y[:2], epochs=1000, verbose=0)

y_pred = model.predict(x[2:]).flatten()
print('Tagets:', y[2:])
print('Predictions:', y_pred)
print('Errors:', y[2:] - y_pred)
