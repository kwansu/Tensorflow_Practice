from python_simulation.MainPyGame import MainPygame
from Othello_World import World_Othello as World
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import collections
import random
import threading


blackModel = keras.Sequential()
blackModel.add(keras.layers.Conv2D(8, kernel_size=(1, 1), activation="relu"))
blackModel.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
blackModel.add(keras.layers.Conv2D(192, kernel_size=(3, 3), activation="relu"))
blackModel.add(keras.layers.Flatten())
blackModel.add(keras.layers.Dense(512, tf.nn.relu, True))
blackModel.add(keras.layers.Dense(64))
blackModel.compile(keras.optimizers.Adam(learning_rate=0.001), loss="mse")

blackTargetModel = keras.models.clone_model(blackModel)
whiteModel = keras.models.clone_model(blackModel)
whiteModel.compile(keras.optimizers.Adam(learning_rate=0.001), loss="mse")
whiteTargetModel = keras.models.clone_model(blackModel)

bufferSize = 10000
blackBuffer = collections.deque(maxlen=bufferSize)
whiteBuffer = collections.deque(maxlen=bufferSize)

statesBuffer = np.zeros([bufferSize + 1, 8, 8, 1])
batchSize = 64
# targetInterval = 10
# targetCount = targetInterval-batchSize

lineLenth = 320
simulation = MainPygame(lineLenth, lineLenth, speed=1, fps=5)
world = World(lineLenth, simulation.window)


def runSimulation():
    simulation.run(world)


simulationThread = threading.Thread(target=runSimulation)
simulationThread.start()

stateIndex = 0

episodeCount = 1000
discountRate = 0.99
reciprocal = 1.0 / episodeCount ** 2

for i in range(episodeCount):
    e = reciprocal * ((episodeCount - i) ** 2)
    isTerminal = False
    blackRewardSum = 0
    whiteRewardSum = 0
    # isBlackWin = False
    isBlackTurn = True
    world.setup(statesBuffer[stateIndex])

    while not isTerminal:
        state = statesBuffer[stateIndex]
        nextState = statesBuffer[stateIndex + 1]
        if isBlackTurn:
            buffer = blackBuffer
            model = blackModel
            targetModel = blackTargetModel
        else:
            buffer = whiteBuffer
            model = whiteModel
            targetModel = whiteTargetModel

        r = random.random()
        if r < e ** 4:
            action = random.randrange(0, 64)
            actionPos = (action & 0b111, action >> 3)
        elif r < e:
            actionPos = world.getRandomPutablePos()
            action = int(actionPos[0] | (actionPos[1] << 3))
        else:
            x = np.reshape(state, [1, 8, 8, 1])
            action = np.argmax(model.predict(x))
            actionPos = (action & 0b111, action >> 3)

        reward, isTerminal = world.step(actionPos, nextState)

        buffer.append([state, action, reward, nextState, isTerminal])

        if len(buffer) > batchSize:
            trainBatch = random.sample(buffer, batchSize)
            states = np.array([x[0] for x in trainBatch])
            actions = np.array([x[1] for x in trainBatch])
            rewards = np.array([x[2] for x in trainBatch])
            nextStates = np.array([x[3] for x in trainBatch])
            terminals = np.array([x[4] for x in trainBatch])

            Q_target = (
                rewards
                + discountRate
                * np.max(targetModel.predict(nextStates), axis=1)
                * ~terminals
            )

            y = model.predict(states)
            y[np.arange(len(states)), actions] = Q_target

            model.fit(states, y, batchSize, verbose=0)

        stateIndex += 1
        if stateIndex >= bufferSize:
            stateIndex = 0

        if isBlackTurn:
            blackRewardSum += reward
        else:
            whiteRewardSum += reward

        isBlackTurn = not isBlackTurn

    if len(blackBuffer) > batchSize + 1:
        blackTargetModel.set_weights(blackModel.get_weights())
    if len(whiteBuffer) > batchSize + 1:
        whiteTargetModel.set_weights(whiteModel.get_weights())
    print(
        "episode: {}  black rewardSum: {}  white rewardSum: {}".format(
            i, blackRewardSum, whiteRewardSum
        )
    )


blackModel.save("model/othello.h5")
simulationThread.join()
