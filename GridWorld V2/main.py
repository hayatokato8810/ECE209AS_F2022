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
	)

	fig = plt.figure(figsize=(5,5))
	ax = plt.subplot()

	world.plotStateSpace(ax)

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()