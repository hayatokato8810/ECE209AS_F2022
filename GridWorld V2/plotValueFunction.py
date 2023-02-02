import numpy as np
import matplotlib.pyplot as plt

from GridWorld import GridWorld
from loadMapData import loadMapData

def main():
	print("Starting...")

	stateSpace, actionSpace, rewardSpace = loadMapData()

	world = GridWorld(
		stateSpace = stateSpace,
		actionSpace = actionSpace,
		rewardSpace = rewardSpace,
		errorProb = 0.2,
		startState = (2,4),
		method = "Value Iteration",
	)

	# Plot Figure
	fig = plt.figure(figsize=(6,5))
	ax = plt.subplot()

	world.plotValueFunction(ax, world.V)
	ax.set_title(f'Optimal Value Function at $k = {world.iter}$ Using {world.method}')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()