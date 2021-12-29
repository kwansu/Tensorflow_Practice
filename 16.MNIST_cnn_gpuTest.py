import time
import tensorflow.keras as keras
from tensorflow.keras import layers


# 28x28 사이즈의 손글씨 데이타베이스 mnist를 로드한다. train 60000, test 10000
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

dropoutRate = 0.3

model = keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="SAME"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation="softmax"))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

start_time = time.time()
start = time.gmtime(start_time)
print("훈련 시작 : %d시 %d분 %d초" % (start.tm_hour, start.tm_min, start.tm_sec))

model.fit(x_train, y_train, batch_size=512, epochs=20, validation_split=0.1)

end_time = time.time()
end = time.gmtime(end_time)
print("훈련 끝 : %d시 %d분 %d초" % (end.tm_hour, end.tm_min, end.tm_sec))
# 소요 시간 측정
end_start = end_time - start_time
end_start = time.gmtime(end_start)
print("소요시간 : %d시 %d분 %d초" % (end_start.tm_hour, end_start.tm_min, end_start.tm_sec))

print("학습결과 평가")
predictions = model.evaluate(x_test, y_test)
print("Loss, Accuracy :", predictions)  # max 0.990
