import numpy as np
import tensorflow.keras as keras


loadedData = np.loadtxt("data/data-01-test-score.csv", delimiter=",", dtype=np.float32)
x_data = loadedData[:, 0:-1]
y_data = loadedData[:, [-1]]

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, activation="linear"))

model.compile(optimizer=keras.optimizers.SGD(lr=1e-5), loss="mse")

model.fit(x_data, y_data, epochs=1000)

predict = model.predict(np.array([[80, 80, 80]]))
print(predict)
