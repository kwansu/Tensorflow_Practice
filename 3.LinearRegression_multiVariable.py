import numpy as np
import tensorflow.keras.models as models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

x_data = np.array([[73., 80., 75.],
                    [93., 88., 93.],
                    [89., 91., 90.],
                    [96., 98., 100.],
                    [73., 66., 70.]])

y_data = np.array([152., 185., 180., 196., 142.])

model = models.Sequential()
model.add(Dense(1, activation='linear'))

model.compile(optimizer=SGD(learning_rate=0.00001), loss='mse')

model.fit(x_data, y_data, epochs=1000)

predict = model.predict(np.array([[100, 100, 80]]))
print(predict)
