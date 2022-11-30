
import numpy as np
import matplotlib.pyplot as plt

from GridWorld import GridWorld

def main():
	print("Starting...")

	worldWidth = 5
	worldHeight = 5
		# Define Gridworld map via matrix representation
	grid_map = np.zeros((worldWidth,worldHeight))
	# Obstacles
	grid_map[1,3] = 1
	grid_map[2,3] = 1
	grid_map[1,1] = 1
	grid_map[2,1] = 1
	# Icecream Shops
	grid_map[2,2] = 2
	grid_map[2,0] = 3
	# Road
	grid_map[4,0] = 4
	grid_map[4,1] = 4
	grid_map[4,2] = 4
	grid_map[4,3] = 4
	grid_map[4,4] = 4

	stateSpace = {}
	for x in range(worldWidth):
		for y in range(worldHeight):
			if grid_map[x,y] == 
			stateSpace.append((x,y))
	stateSize = len(stateSpace)

	actionSpace = [(1,0),(0,-1),(-1,0),(0,1),(0,0)]
	actionSize = len(actionSpace)

	world = GridWorld((worldWidth,worldHeight),stateSpace,actionSpace)

	# Plot Figure
	fig = plt.figure(figsize=(5,5))
	ax = plt.subplot()

	world.plotStateSpace(ax,zorder=2)
	p = np.ones((stateSize,1))/stateSize
	world.plotProbDistOverStateSpace(ax,p)

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()