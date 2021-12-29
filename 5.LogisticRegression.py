import numpy as np
import tensorflow.keras as keras


x_data = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]], np.float)

y_data = np.array([[0], [0], [0], [1], [1], [1]], np.float)

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(
    optimizer=keras.optimizers.SGD(), loss="binary_crossentropy", metrics=["accuracy"]
)

model.fit(x_data, y_data, epochs=5000)
