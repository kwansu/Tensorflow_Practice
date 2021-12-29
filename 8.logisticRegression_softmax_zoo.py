import numpy as np
import tensorflow.keras as keras


loadedData = np.loadtxt('data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
 
# 테스트용으로 5개만 미리 빼놓았다.
x_test = loadedData[0:5, 0:-1]
y_test = loadedData[0:5, [-1]]

# 학습용
x_train = loadedData[5:, 0:-1]
y_train = loadedData[5:, [-1]]
y_train = keras.utils.to_categorical(y_train, 7)

model = keras.models.Sequential()
model.add(keras.layers.Dense(7, input_dim=x_train.shape[1], activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)

print('학습결과 평가')
pred = model.predict_classes(x_test)
for p, y in zip(pred, y_test.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
