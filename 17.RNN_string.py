import numpy as np
import tensorflow.keras as keras


# hihello
x_data = np.array(
    [
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ]
    ],
    np.float,
)

y_data = [[1, 0, 2, 3, 3, 4]]
y_data = keras.utils.to_categorical(y_data, num_classes=5)

model = keras.Sequential()
cell = keras.layers.LSTMCell(5)
model.add(keras.layers.RNN(cell=cell, return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(5, activation="softmax")))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_data, y_data, epochs=50)

vocabularies = ["h", "i", "e", "l", "o"]

predictions = model.predict(x_data)
for i, prediction in enumerate(predictions):
    print([vocabularies[c] for c in np.argmax(prediction, axis=1)])
