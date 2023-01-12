import math
import numpy as np
from collections import defaultdict 
from numpy.random import choice
import matplotlib.pyplot as plt

class GridWorld():
	vector = tuple[int]

	def __init__(self, 
		stateSpace:np.ndarray, 
		actionSpace:list[vector],
		errorProb:float,
		startState:vector,
	):
		self.DIM = np.shape(stateSpace)
		self.P_E = errorProb

		N_S = np.size(stateSpace)
		N_A = len(actionSpace)
		N_O = self.DIM[0]

		# State
		self.S = []
		for x in range(self.DIM[0]):
			for y in range(self.DIM[1]):
				self.S.append((x,y))
		self.S_DICT = defaultdict(list)
		for x in range(self.DIM[0]):
			for y in range(self.DIM[1]):
				self.S_DICT[stateSpace[x,y]].append((x,y))
		self.currentState = startState

		# Action
		self.A = actionSpace

		# Probability
		self.P = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			for a in range(N_A):
				for s_ in range(N_S):
					self.P[s,a,s_] = self.transitionProbability(self.S[s],self.A[a],self.S[s_])

		# Reward
		self.R_DICT = defaultdict(int)
		self.R_DICT[2] = 1   # Icecream Store D
		self.R_DICT[3] = 10  # Icecream Store S
		self.R_DICT[4] = -10 # Road
		self.R = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			if self.S[s] in self.S_DICT[2]:
				self.R[s,:,:] = self.R_DICT[2]
			elif self.S[s] in self.S_DICT[3]:
				self.R[s,:,:] = self.R_DICT[3]
			elif self.S[s] in self.S_DICT[4]:
				self.R[s,:,:] = self.R_DICT[4]

		# Horizon
		self.H = 10

		# Discount Gamma
		self.g = 0.8

		# Observation
		self.O = []
		for o in range(N_O):
			self.O.append(o)
		self.observation = np.zeros((N_S,N_O))
		for s in range(N_S):
			for o in range(N_O):
				self.observation[s,o] = self.observationProbability(self.S[s], o, ((2,2),(2,0)))

		print(sum(self.observation,0))



# === Computational Functions ===
	# Defines the system dynamics of the agent
	def systemDynamics(self, state:vector, action:vector):
		desired_state = (state[0]+action[0],state[1]+action[1])
		if action not in self.A or desired_state not in self.S or desired_state in self.S_DICT[1]:
			return None
		return desired_state

	# Computes set of all states reachable by a single action
	def S_adj(self, state:vector):
		space = []
		for action in self.A:
			new_state = self.systemDynamics(state,action)
			if new_state != None:
				space.append(new_state)
		return space

	# Calculates the corresponding transition probability for each state, action, next state triplet
	def transitionProbability(self, state:vector, action:vector, next_state:vector):
		probability = 0
		# States must not be occupied by obstacle and they must be adjacent
		if next_state not in self.S_DICT[1] and state not in self.S_DICT[1] and next_state in self.S_adj(state):
			# If desired action succeeded
			if next_state == self.systemDynamics(state, action):
				# If the state changes
				if state == next_state:
					probability = 1
				else:
					probability = 1 - self.P_E
			# If desired action failed
			else:
				# If the state changes
				if state == next_state:
					probability = 1
					# Subtract the probability of all surrounding states
					for s in self.S_adj(state):
						if s != state:
							probability -= self.transitionProbability(state, action, s)
				else:
					if action != self.A[-1]:	
						probability = self.P_E/4.0
		return probability

	# Calculates the corresponding observation probability
	def observationProbability(self, state:vector, observation:int, environment:tuple[vector]):
		S_iceD, S_iceS = environment
		dD = math.dist(list(S_iceD),list(state))
		dS = math.dist(list(S_iceS),list(state))
		try:
			h = 2/(1/dD+1/dS)
		except:
			h = 0
		high_h = math.ceil(h)
		low_h = math.floor(h)
		if observation == high_h:
			probability = 1-(high_h - h)
		elif observation == low_h:
			probability = high_h - h
		else:
			probability = 0
		return probability

	# Updates the current state of the agent stociastically based off a desired action
	def updateState(self, action:vector):
		try:
			s = self.S.index(self.currentState)
			a = self.A.index(action)
		except:
			print("Not a valid state action pair")
			exit()
		p = self.P[s,a,:]
		randomIdx = choice(range(len(p)),p=p)
		self.currentState = self.S[randomIdx]

# === Plotting Functions ===
	# Plots visualization of the environment
	def plotStateSpace(self, ax:plt.axis, **kwarg):
		for x,y in self.S_DICT[1]:
			rx = [x-0.5,x+0.5,x+0.5,x-0.5]
			ry = [y-0.5,y-0.5,y+0.5,y+0.5]
			plt.fill(rx,ry,color='gray',zorder=-2)
			plt.scatter(x,y,500,'k',marker='x')
		for x,y in self.S_DICT[2]:
			rx = [x-0.5,x+0.5,x+0.5,x-0.5]
			ry = [y-0.5,y-0.5,y+0.5,y+0.5]
			plt.fill(rx,ry,color='lime',zorder=-2)
		for x,y in self.S_DICT[3]:
			rx = [x-0.5,x+0.5,x+0.5,x-0.5]
			ry = [y-0.5,y-0.5,y+0.5,y+0.5]
			plt.fill(rx,ry,color='lime',zorder=-2)
		for x,y in self.S_DICT[4]:
			rx = [x-0.5,x+0.5,x+0.5,x-0.5]
			ry = [y-0.5,y-0.5,y+0.5,y+0.5]
			plt.fill(rx,ry,color='red',zorder=-2)
		if self.currentState:
			plt.scatter(self.currentState[0],self.currentState[1],500,'k')
		ax.set_xticks(np.arange(0, self.DIM[0], 1))
		ax.set_yticks(np.arange(0, self.DIM[1], 1))
		ax.set_xticks(np.arange(-.5, self.DIM[0]+.5, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.DIM[1]+.5, 1), minor=True)
		ax.set_xlim([-.5, self.DIM[0]-.5])
		ax.set_ylim([-.5, self.DIM[1]-.5])
		ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
		ax.set_aspect('equal')

	def plotProbDistOverStateSpace(self, ax:plt.axis, pr:np.ndarray, **kwarg):
		pr = pr.reshape(self.DIM).transpose()
		for (j,i),label in np.ndenumerate(pr):
			ax.text(i,j,round(label,4),ha='center',va='center')
		im = ax.imshow(pr, origin = 'lower',vmin=0,vmax=1,**kwarg)
		plt.colorbar(im)
		ax.set_xticks(np.arange(0, self.DIM[0], 1))
		ax.set_yticks(np.arange(0, self.DIM[1], 1))
		ax.set_xticks(np.arange(-.5, self.DIM[0]+.5, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.DIM[1]+.5, 1), minor=True)
		ax.set_xlim([-.5, self.DIM[0]-.5])
		ax.set_ylim([-.5, self.DIM[1]-.5])
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		ax.set_aspect('equal')




