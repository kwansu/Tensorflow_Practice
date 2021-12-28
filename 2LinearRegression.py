import numpy as np
import tensorflow.keras.models as models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

x_train = [1, 2, 3, 4]
y_train = [2, 0, -2, -4]

model = models.Sequential()
model.add(Dense(1, activation='linear'))

model.compile(optimizer=SGD(learning_rate=0.1), loss='mse')

model.fit(np.array(x_train, np.float), np.array(y_train, np.float), epochs=1000)

predict = model.predict([-2, 7])
print(predict)
