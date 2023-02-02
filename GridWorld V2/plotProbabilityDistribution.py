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
		errorProb = 0.25,
		startState = (2,4),
	)

	# Plot Figure
	fig = plt.figure(figsize=(6,5))
	ax = plt.subplot()

	state = 4
	action = 4
	world.plotProbDistOverStateSpace(ax,world.P[state,action,:],)

	ax.set_title(f'Probability Distribution (State={world.S[state]}, Action={world.A[action]})')

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()