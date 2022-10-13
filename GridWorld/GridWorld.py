#!/usr/bin/env python

""" GridWorld.py """

__author__ = "Hayato Kato"

from copy import copy
import numpy as np
import matplotlib.pyplot as plt

# Boolean functions for decoding the grid world map
def isAlwaysTrue(matrix,s):
	return True

def isObstacle(matrix, s):
	return matrix[s[0],s[1]] == 1

def isIcecreamShop_D(matrix, s):
	return matrix[s[0],s[1]] == 2

def isIcecreamShop_S(matrix, s):
	return matrix[s[0],s[1]] == 3

def isRoad(matrix, s):
	return matrix[s[0],s[1]] == 4

# Converts the 2D position in a matrix into a set of states
def convert2Set(matrix,indicatorFunc):
	result = []
	x_dim, y_dim = matrix.shape
	for i in range(x_dim):
		for j in range(y_dim):
			if indicatorFunc(matrix,(i,j)):
				result.append([i,j])
	return result

# GridWorld Class
class GridWorld:
	# Takes the initial state and the map of obstacles represented using a numpy array
	def __init__(self, worldmap):
		# State Space
		self.S = convert2Set(worldmap, isAlwaysTrue)
		# Action Space
		self.A = ['up','down','left','right','center']

		# Obstacle Space
		self.S_obs = convert2Set(worldmap, isObstacle)
		# Road Space
		self.S_W = convert2Set(worldmap,isRoad)
		# Icecream Shop D Space
		self.S_D = convert2Set(worldmap, isIcecreamShop_D)
		# Icecream Shop S Space
		self.S_S = convert2Set(worldmap, isIcecreamShop_S)

		# Reward
		self.R_D = 1
		self.R_S = 10
		self.R_W = -1

		self.output = 0
		self.map = worldmap

		# Constants
		self.p_e = 0.4
		self.gamma = 0.8
		self.converge_threshold = 0.00001

		self.x_dim, self.y_dim = self.map.shape

		# Optimal Value
		self.V_star = np.zeros((self.x_dim, self.y_dim))

	# Updates the current state to the next state (to be implemented)
	def update(self, action):
		print('Update the state')

	# Transition probability function returns a probability given transition triplet
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
					if action != "center":
						probability = self.p_e/4.0
		return probability

	# State Dynamics function (Desired State Function)
	def f(self, state, action):
		desired_state = copy(state)
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
		return desired_state

	# Reward function
	def R(self, state, action, next_state):
		reward = 0
		rewardType = 'default'
		match rewardType:
			# Default Reward Function
			case 'default':
				if state in self.S_D:
					reward = self.R_D
				elif state in self.S_S:
					reward = self.R_S
				elif state in self.S_W:
					reward = self.R_W
			# Reward Function with Intentional Entry
			case 'intentional':
				if next_state in self.S_D and next_state == self.f(state, action):
					reward = self.R_D
				elif next_state in self.S_S and next_state == self.f(state, action):
					reward = self.R_S
				elif next_state in self.S_W:
					reward = self.R_W
		return reward


	def compute_V_star(self, plot = False):
		new_V = np.zeros((self.x_dim, self.y_dim))
		diff = []
		k = 0
		while True:
			for s in self.S:
				max_V = 0
				for a in self.A:
					summation = 0
					for s_ in self.S:
						summation += self.pr(s,a,s_)*(self.R(s,a,s_)+self.gamma*self.V_star[s_[0],s_[1]])
					if max_V < summation:
						max_V = summation
				new_V[s[0],s[1]] = max_V
			temp = np.linalg.norm(self.V_star - new_V)
			diff.append(temp)
			print(temp)
			if temp < self.converge_threshold:
				break
			self.V_star = copy(new_V)
			if plot:
				fig,ax = self.plot(self.V_star, k)
			k += 1
		return k, diff

	def plot(self, V, iteration, blocking = False):
		values = V.transpose()
		fig, ax = plt.subplots(1,1)
		for (j,i),label in np.ndenumerate(values):
		    ax.text(i,j,round(label,4),ha='center',va='center')
		im = ax.imshow(values, cmap="RdBu", origin = 'lower')
		ax.set_xticks(np.arange(-.5, self.x_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		ax.set_title('Value Iteration k=' + str(iteration))
		plt.colorbar(im)
		plt.show(block=blocking)
		plt.pause(0.01)
		plt.close('all')
		return fig, ax

	# Computes all adjacent states (all next states reachable via action in action space)
	def S_adj(self, state):
		space = []
		for action in self.A:
			new_state = self.f(state,action)
			if new_state != None:
				space.append(new_state)
		return space

	# Visualizes transition probabilities across all possible next states given a current state and action
	def compute_P(self, state, action, plot=False):
		probabilities = np.zeros((self.x_dim,self.y_dim))
		for x in range(self.x_dim):
			for y in range(self.y_dim):
				probabilities[x,y] = self.pr(state,action,[x,y])
		if plot:
			probabilities = probabilities.transpose()
			fig, ax = plt.subplots(1,1)
			for (j,i),label in np.ndenumerate(probabilities):
			    ax.text(i,j,round(label,2),ha='center',va='center')
			im = ax.imshow(probabilities, origin = 'lower')
			ax.set_xticks(np.arange(-.5, self.x_dim, 1), minor=True)
			ax.set_yticks(np.arange(-.5, self.y_dim, 1), minor=True)
			ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
			ax.set_title('s_t=['+str(state[0])+','+str(state[1])+'], a=' + action + ', p_e=' + str(self.p_e))
			plt.colorbar(im)
			plt.show()

	# Draws an ASCII art of the current grid world
	def draw_ASCII(self, state):
		print('  ┌'+'─'*(2*self.x_dim+1)+'┐')
		for j in range(self.y_dim-1,-1,-1):
			line = str(j) + ' │ '
			for i in range(self.x_dim):
				if state != [i,j]:
					if [i,j] in self.S_obs:
						line += 'X ' # Obstacle
					elif [i,j] in self.S_W:
						line += '# ' # Road
					elif [i,j] in self.S_D:
						line += 'D ' # Road
					elif [i,j] in self.S_S:
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

def main():
	# Define matrix that describes the gridworld
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

	# Initial State
	starting_state = [2,2]

	# Initialize Gridworld class
	world = GridWorld(grid_map)

	# ASCII Visualization of particular state
	world.draw_ASCII(starting_state)

	# Displays all transition probabilities for a given state and action pair
	world.compute_P(starting_state,'right',plot=True)

	# Runs Value Iteration and finds the optimal value function
	converge_k, d = world.compute_V_star(plot=False)
	#print(converge_k)
	world.plot(world.V_star, converge_k, blocking=True)

if __name__ == '__main__':
	main()