import numpy as np
import matplotlib.pyplot as plt

from GridWorld import GridWorld

def main():
	print("Starting...")

	# Define Gridworld map via matrix representation
	worldWidth = 5
	worldHeight = 5
	stateSpace = np.zeros((worldWidth,worldHeight))
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

	world = GridWorld(
		stateSpace = stateSpace,
		actionSpace = actionSpace,
		errorProb = 0.2,
		startState = (2,4),
	)

	# Plot Figure
	fig = plt.figure(figsize=(6,5))
	ax = plt.subplot()

	world.plotProbDistOverStateSpace(ax,world.P[17,1,:],)

	ax.set_title(f'Probability Distribution')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()