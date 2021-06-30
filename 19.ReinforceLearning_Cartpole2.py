import gym
import random
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import dtype

environment = gym.make('CartPole-v0')

episodeCount = 1000
discount_rate = 0.99
targetInterval = 8  # 몇번마다 한번씩 메인을 타겟에 복사할지
rewardSum = 0

inputSize = environment.observation_space.shape[0]

main_model = tf.keras.models.Sequential()
main_model = tf.keras.Sequential()
main_model.add(tf.keras.layers.Dense(16, 'relu', True, 'glorot_normal'))
main_model.add(tf.keras.layers.Dense(2, kernel_initializer='glorot_normal'))
main_model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

targetModel = tf.keras.models.clone_model(main_model)

bufferSize = 2048

states_buffer = np.zeros([bufferSize, inputSize], dtype=float)
actions_buffer = np.zeros([bufferSize], dtype=int)
rewards_buffer = np.zeros([bufferSize], dtype=int)
terminals_buffer = np.zeros([bufferSize], dtype=bool)
orderList = tuple(range(0, bufferSize))
nextOrderList = np.arange(1, bufferSize)

countR = 1 / episodeCount
continueCount = 0  # 몇번 연속 기준을 통과했는지 판단용
temp = 0
bufferIndex = 0

for i in range(episodeCount):
    e = countR * i
    stepCount = 0
    isTerminal = False
    states_buffer[0] = environment.reset()

    while not isTerminal:
        if random.random() > e:
            actions_buffer[stepCount] = environment.action_space.sample()
        else:
            x = np.reshape(states_buffer[stepCount], [1, inputSize])
            actions_buffer[stepCount] = np.argmax(main_model.predict(x))

        states_buffer[stepCount+1], rewards_buffer[stepCount], isTerminal, _ = environment.step(
            actions_buffer[stepCount])

        terminals_buffer[stepCount] = isTerminal

        if terminals_buffer[stepCount]:
            rewards_buffer[stepCount] = -1

        if bufferIndex > 10:
            if bufferIndex > 64:
                batch_size = 64
            else:
                batch_size = bufferIndex

            order = random.sample(orderList[0:bufferIndex], batch_size)
            nextOrder = nextOrderList[order]

            Q_target = rewards_buffer[:batch_size] + discount_rate * np.max(targetModel.predict(
                states_buffer[1:batch_size+1]), axis=1) * ~terminals_buffer[:batch_size]
            y = main_model.predict(states_buffer[:batch_size])
            y[np.arange(batch_size), actions_buffer[:batch_size]] = Q_target

            main_model.fit(states_buffer[:batch_size],y, batch_size, verbose=0)

        if temp > 10:
            targetModel.set_weights(main_model.get_weights())
            temp = 0

        bufferIndex += 1
        if bufferIndex >= bufferSize:
            bufferIndex = 0

        temp += 1
        stepCount += 1

    print("episode: {}  steps: {}".format(i, stepCount))

    if stepCount > 150:
        continueCount += 1
        if continueCount > 50:
            print('통과기준을 만족')
            break
    else:
        continueCount = 0

main_model.save('model/cartpole2.h5')
