from python_simulation.MainPyGame import MainPygame
from Othello_World import World_Othello as World
import pygame
import numpy as np

pygame.init()
width = 320
height = 320
window = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
timeSpeed = 0.001 * 1 # 1배속 = 0.001
minDeltaTime = 0.5 * 1000
fps = 60
isRunning = True
isStoped = False

world = World(320, window)

state = np.zeros([8,8],dtype=int)
world.setup(state)

while isRunning:
    deltaTime = clock.tick(fps)
    if deltaTime > minDeltaTime:
        deltaTime = minDeltaTime

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False
        if event.type == pygame.KEYDOWN:
            # if event.key == pygame.K_SPACE:
            #     world.setup()
            if event.key == pygame.K_SPACE:
                world.randomPut()

        if event.type == pygame.MOUSEBUTTONUP:
            world.putPlayer(list(event.pos))

    if isStoped:
        continue
    
    pygame.display.update()
pygame.quit()     