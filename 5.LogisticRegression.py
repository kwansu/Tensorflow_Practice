import numpy
import tensorflow

x_data = numpy.array([[1, 2],
                      [2, 3],
                      [3, 1],
                      [4, 3],
                      [5, 3],
                      [6, 2]], numpy.float)

y_data = numpy.array([[0],
                      [0],
                      [0],
                      [1],
                      [1],
                      [1]], numpy.float)

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tensorflow.keras.optimizers.SGD(),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=5000)  
