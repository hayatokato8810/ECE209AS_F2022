#!/usr/bin/env python

""" GridWorld.py """

__author__ = "Hayato Kato"

from copy import copy
import numpy as np
import matplotlib.pyplot as plt

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

def convert2Set(matrix,indicator):
	result = []
	x_dim, y_dim = matrix.shape
	for i in range(x_dim):
		for j in range(y_dim):
			if indicator(matrix,(i,j)):
				result.append([i,j])
	return result

# GridWorld Class
class GridWorld:
	# Takes the initial state and the map of obstacles represented using a numpy array
	def __init__(self, worldmap):
		self.S = convert2Set(worldmap, isAlwaysTrue)
		self.A = ['up','down','left','right','center']

		self.S_obs = convert2Set(worldmap, isObstacle)
		self.S_W = convert2Set(worldmap,isRoad)
		self.S_D = convert2Set(worldmap, isIcecreamShop_D)
		self.S_S = convert2Set(worldmap, isIcecreamShop_S)

		self.output = 0
		self.map = worldmap

		self.p_e = 0.4

		self.x_dim, self.y_dim = self.map.shape

	# Updates the current state to the next state (to be implemented)
	def update(self, action):
		print('Update the state')

	# Returns the transition probability given a transition triplet
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

	def f(self, state, action):
		desired_state = copy(state)
		if action == self.A[0]: 
			if state[1] != self.y_dim-1:
				desired_state[1] = desired_state[1] + 1
			else:
				return None
		elif action == self.A[1]: 
			if state[1] != 0:
				desired_state[1] = desired_state[1] - 1
			else:
				return None
		elif action == self.A[2]:
			if state[0] != 0:
				desired_state[0] = desired_state[0] - 1
			else:
				return None
		elif action == self.A[3]:
			if state[0] != self.x_dim-1:
				desired_state[0] = desired_state[0] + 1
			else:
				return None
		elif action == self.A[4]:
			pass
		else:
			print("Not a valid action")
			exit()
		return desired_state

	def S_adj(self, state):
		space = []
		for action in self.A:
			new_state = self.f(state,action)
			if new_state != None:
				space.append(new_state)
		return space

	def displayAllProbability(self, state, action):
		probabilities = np.zeros((self.x_dim,self.y_dim))
		for x in range(self.x_dim):
			for y in range(self.y_dim):
				probabilities[x,y] = self.pr(state,action,[x,y])
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
	def draw(self, state):
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
	grid_map = np.zeros((5,5))
	
	grid_map[1,3] = 1
	grid_map[2,3] = 1
	grid_map[1,1] = 1
	grid_map[2,1] = 1

	grid_map[2,2] = 2
	grid_map[2,0] = 3

	grid_map[4,0] = 4
	grid_map[4,1] = 4
	grid_map[4,2] = 4
	grid_map[4,3] = 4
	grid_map[4,4] = 4
	
	starting_state = [0,2]
	state = starting_state

	world = GridWorld(grid_map)

	world.draw([2,2])
	#result = world.f([0,2],"left")
	#result = round(world.pr([2,2],"left",[3,2]),2)
	#print(result)

	#result = world.S_adj([0,0])
	world.displayAllProbability([0,2],'up')
	#print(result)


if __name__ == '__main__':
	main()