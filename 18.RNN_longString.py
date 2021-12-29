import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


sentence = "this program was make by kwansu"
charSet = list(set(sentence))
charSetToIndex = {c: i for i, c in enumerate(charSet)}

charSetLength = len(charSet)
sequenceLength = 6

x_data = []
y_data = []
for i in range(0, len(sentence) - sequenceLength):
    x_str = sentence[i : i + sequenceLength]
    y_str = sentence[i + 1 : i + sequenceLength + 1]

    x = [charSetToIndex[c] for c in x_str]
    y = [charSetToIndex[c] for c in y_str]

    x_data.append(x)
    y_data.append(y)

batchSize = len(x_data)

x_oneHot = tf.one_hot(x_data, charSetLength)
y_oneHot = tf.one_hot(y_data, charSetLength)

model = keras.Sequential()
model.add(
    layers.LSTM(
        charSetLength,
        input_shape=(charSetLength, x_oneHot.shape[2]),
        return_sequences=True,
    )
)
model.add(layers.LSTM(charSetLength, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(charSetLength, activation="softmax")))
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    metrics=["accuracy"],
)

model.fit(x_oneHot, y_oneHot, epochs=100)

results = model.predict(x_oneHot)
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j == 0:
        print("".join([charSet[t] for t in index]), end="")
    else:
        print(charSet[index[-1]], end="")
