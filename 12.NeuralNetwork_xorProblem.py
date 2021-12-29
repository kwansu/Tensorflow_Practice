import numpy as np
import tensorflow.keras as keras


x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# accuracy가 0.5가 나온다. xor문제
result = model.fit(x_data, y_data, epochs=1000)
