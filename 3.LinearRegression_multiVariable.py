import numpy as np
import tensorflow.keras.models as models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


x_data = np.array(
    [
        [73.0, 80.0, 75.0],
        [93.0, 88.0, 93.0],
        [89.0, 91.0, 90.0],
        [96.0, 98.0, 100.0],
        [73.0, 66.0, 70.0],
    ]
)

y_data = np.array([152.0, 185.0, 180.0, 196.0, 142.0])

model = models.Sequential()
model.add(Dense(1, activation="linear"))

model.compile(optimizer=SGD(learning_rate=0.00001), loss="mse")

model.fit(x_data, y_data, epochs=1000)

predict = model.predict(np.array([[100, 100, 80]]))
print(predict)
