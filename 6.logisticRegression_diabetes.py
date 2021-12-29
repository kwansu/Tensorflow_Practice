import numpy as np
import tensorflow.keras as keras


loadedData = np.loadtxt("data/data-03-diabetes.csv", delimiter=",", dtype=np.float32)

# 테스트용으로 5개만 미리 빼놓았다.
x_test = loadedData[0:5, 0:-1]
y_test = loadedData[0:5, [-1]]
# 학습용
x_train = loadedData[5:, 0:-1]
y_train = loadedData[5:, [-1]]


model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_dim=x_train.shape[1], activation="sigmoid"))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=1000)

print("학습결과 평가")
evaluation = model.evaluate(x_test, y_test)
print(evaluation)
