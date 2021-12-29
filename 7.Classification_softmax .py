import numpy as np
import tensorflow.keras as keras


x_data = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7],
]

y_data = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
]

model = keras.models.Sequential()
model.add(keras.layers.Dense(3, activation="softmax"))

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(np.array(x_data, np.float), np.array(y_data, np.float), epochs=1500)

predict = model.predict(np.array([[1, 11, 7, 9]]))
print(predict)
