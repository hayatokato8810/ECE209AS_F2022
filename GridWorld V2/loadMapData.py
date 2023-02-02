import numpy as np
import matplotlib.pyplot as plt

def loadMapData():
	# Define Gridworld map via matrix representation
	worldWidth = 5
	worldHeight = 5
	stateSpace = np.zeros((worldWidth,worldHeight))
	# 0 = Empty Space
	# 1 = Obstacles
	stateSpace[1,3] = 1
	stateSpace[2,3] = 1
	stateSpace[1,1] = 1
	stateSpace[2,1] = 1
	# 2,3 = Icecream Shops
	stateSpace[2,2] = 2
	stateSpace[2,0] = 3
	# 4 = Road
	stateSpace[4,0] = 4
	stateSpace[4,1] = 4
	stateSpace[4,2] = 4
	stateSpace[4,3] = 4
	stateSpace[4,4] = 4

	# All possible actions that can be taken by the agent
	actionSpace = [(1,0),(0,-1),(-1,0),(0,1),(0,0)]

	rewardSpace = np.zeros(5)
	rewardSpace[0] = 0
	rewardSpace[1] = 0
	rewardSpace[2] = 1
	rewardSpace[3] = 10
	rewardSpace[4] = -10

	return stateSpace, actionSpace, rewardSpace