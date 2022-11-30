import math
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt

class GridWorld():
	vector = tuple[float]

	def __init__(self, 
		dimension:tuple[int], 
		stateSpace:list[vector], 
		actionSpace:list[vector],
		errorProb:float,
	):
		self.DIM = dimension
		self.p_e = errorProb

		self.S = stateSpace
		self.A = actionSpace

		N_S = len(self.S)
		N_A = len(self.A)

		self.P = np.zeros((N_S,N_A,N_S))


	def plotStateSpace(self, ax:plt.axis, **kwarg):
		pass
		'''
		for state in self.S:
			ax.scatter(state[0],state[1],1,color='k',**kwarg)
		ax.set_xticks(np.arange(0, self.DIM[0], 1))
		ax.set_yticks(np.arange(0, self.DIM[1], 1))
		ax.set_xticks(np.arange(-.5, self.DIM[0]+.5, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.DIM[1]+.5, 1), minor=True)
		ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=1)
		ax.set_aspect('equal')
		'''

	def plotProbDistOverStateSpace(self, ax:plt.axis, pr:np.ndarray, **kwarg):
		pr = pr.reshape(self.DIM).transpose()
		for (j,i),label in np.ndenumerate(pr):
			ax.text(i,j,round(label,4),ha='center',va='center')
		im = ax.imshow(pr, origin = 'lower',**kwarg)
		ax.set_xticks(np.arange(0, self.DIM[0], 1))
		ax.set_yticks(np.arange(0, self.DIM[1], 1))
		ax.set_xticks(np.arange(-.5, self.DIM[0]+.5, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.DIM[1]+.5, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		ax.set_aspect('equal')



