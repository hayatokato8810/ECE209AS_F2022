import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from GridWorld import GridWorld
from loadMapData import loadMapData

def main():

	def animate(time,ax):
		ax.cla()
		world.plotStateSpace(ax)
		nextAction = world.A[world.Pi[world.S.index(world.currentState)]]
		world.updateState(nextAction)
		ax.set_title(f't = {time}')

	print("Starting...")

	stateSpace, actionSpace, rewardSpace = loadMapData()
	
	world = GridWorld(
		stateSpace = stateSpace,
		actionSpace = actionSpace,
		rewardSpace = rewardSpace,
		errorProb = 0.2,
		startState = (2,4),
	)

	# Plot Figure
	fig = plt.figure(figsize=(5,5))
	ax = plt.subplot()

	world.plotStateSpace(ax)

	anim = animation.FuncAnimation(fig, animate, fargs = (ax,), interval = 500)
	
	plt.show()


if __name__ == '__main__':
	main()