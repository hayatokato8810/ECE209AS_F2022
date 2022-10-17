#!/usr/bin/env python

""" GridWorld.py """

__author__ = "Hayato Kato"

import MDP

from copy import copy
import numpy as np
import matplotlib.pyplot as plt

class GridWorld(MDP.MDP):
	def __init__(self, worldmap, p_e):
		self.S = []
		self.x_dim, self.y_dim = worldmap.shape
		for i in range(self.x_dim):
			for j in range(self.y_dim):
				self.S.append((i,j))

		self.S_obs = []
		self.S_iceD = []
		self.S_iceS = []
		self.S_road = []
		for state in self.S:
			if worldmap[state] == 1:
				self.S_obs.append(state)
			elif worldmap[state] == 2:
				self.S_iceD.append(state)
			elif worldmap[state] == 3:
				self.S_iceS.append(state)
			elif worldmap[state] == 4:
				self.S_road.append(state)

		self.p_e = p_e
		self.R_D = 1
		self.R_S = 10
		self.R_W = -1

		self.A = ['up','down','left','right','center']
		N_S = len(self.S)
		N_A = len(self.A)
		self.P = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			for a in range(N_A):
				for s_ in range(N_S):
					self.P[s,a,s_] = self.pr(self.S[s],self.A[a],self.S[s_])
		self.R = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			if self.S[s] in self.S_iceD:
				self.R[s,:,:] = 1
			elif self.S[s] in self.S_iceS:
				self.R[s,:,:] = 10
			elif self.S[s] in self.S_road:
				self.R[s,:,:] = -1
		self.H = 10
		self.g = 0.8

		super().__init__(self.S,self.A,self.P,self.R,self.H,self.g)

	def drawState(self, state):
		print('  ┌'+'─'*(2*self.x_dim+1)+'┐')
		for j in range(self.y_dim-1,-1,-1):
			line = str(j) + ' │ '
			for i in range(self.x_dim):
				if state != (i,j):
					if (i,j) in self.S_obs:
						line += 'X ' # Obstacle
					elif (i,j) in self.S_road:
						line += '# ' # Road
					elif (i,j) in self.S_iceD:
						line += 'D ' # Road
					elif (i,j) in self.S_iceS:
						line += 'S ' # Road
					else:
						line += '• ' # Empty Space
				else:
					line += '@ ' # Agent
			line += '│'
			print(line)
		print('  └'+'─'*(2*self.x_dim+1)+'┘')
		line = '    '
		for j in range(self.x_dim):
			line += str(j) + ' '
		print(line)

	def f(self, state, action):
		desired_state = list(state)
		if action == self.A[0]: 
			if state[1] != self.y_dim-1:
				desired_state[1] += 1
			else:
				return None
		elif action == self.A[1]: 
			if state[1] != 0:
				desired_state[1] -= 1
			else:
				return None
		elif action == self.A[2]:
			if state[0] != 0:
				desired_state[0] -= 1
			else:
				return None
		elif action == self.A[3]:
			if state[0] != self.x_dim-1:
				desired_state[0] += 1
			else:
				return None
		elif action == self.A[4]:
			pass
		else:
			print("Not a valid action")
			exit()
		return tuple(desired_state)

	def pr(self, state, action, next_state):
		probability = 0
		# States must not be occupied by obstacle and they must be adjacent
		if next_state not in self.S_obs and state not in self.S_obs and next_state in self.S_adj(state):
			# If desired action succeeded
			if next_state == self.f(state, action):
				# If the state changes
				if state == next_state:
					probability = 1
				else:
					probability = 1 - self.p_e
			# If desired action failed
			else:
				# If the state changes
				if state == next_state:
					probability = 1
					# Subtract the probability of all surrounding states
					for s in self.S_adj(state):
						if s != state:
							probability -= self.pr(state, action, s)
				else:
					if action != self.A[-1]:
						probability = self.p_e/4.0
		return probability

	def S_adj(self, state):
		space = []
		for action in self.A:
			new_state = self.f(state,action)
			if new_state != None:
				space.append(new_state)
		return space
		
	def plotProbability(self, probabilities):
		probabilities = probabilities.reshape(5,5).transpose()
		fig, ax = plt.subplots(1,1)
		for (j,i),label in np.ndenumerate(probabilities):
		    ax.text(i,j,round(label,2),ha='center',va='center')
		im = ax.imshow(probabilities, origin = 'lower')
		ax.set_xticks(np.arange(-.5, self.x_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		plt.colorbar(im)
		plt.show()

	def plotValue(self):
		matrix = self.V.reshape(5,5).transpose()
		fig, ax = plt.subplots(1,1)
		for (j,i),label in np.ndenumerate(matrix):
		    ax.text(i,j,round(label,2),ha='center',va='center')
		im = ax.imshow(matrix, cmap="RdBu", origin = 'lower')
		ax.set_xticks(np.arange(-.5, self.x_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		plt.colorbar(im)
		plt.show()

def main():
	print("start")

	grid_map = np.zeros((5,5))
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

	world = GridWorld(grid_map,0.4)
	world.plotProbability(world.P[17,0,:])

	world.valueIteration()
	world.plotValue()




if __name__ == '__main__':
	main()