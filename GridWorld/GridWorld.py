#!/usr/bin/env python

""" GridWorld.py """

__author__ = "Hayato Kato"

from copy import copy
import numpy as np

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

		self.S_obs  = convert2Set(worldmap, isObstacle)
		self.S_W = convert2Set(worldmap,isRoad)
		self.S_D = convert2Set(worldmap, isIcecreamShop_D)
		self.S_S = convert2Set(worldmap, isIcecreamShop_S)

		self.output = 0
		self.map = worldmap

		self.p_e = 0.2

		self.x_dim, self.y_dim = self.map.shape

	# Updates the current state to the next state (to be implemented)
	def update(self, action):
		print('Update the state')

	# NOT FINISHED
	def pr(self, state, action, next_state):
		probability = 0
		# Next state is adjacent to current state
		if next_state not in self.S_obs:
			if next_state == self.f(state, action):
				if next_state in self.getAdjacentStates(state):
					print("Case 1")
					probability = 1
				else:
					print("Case 2")
					probability = 1 - self.p_e
			else:
				if next_state in self.getAdjacentStates(state):
					print("Case 3")
					probability = 1
					for s in range(self.getAdjacentStates(state)):
						probability = probability - self.pr(state, action, s)
				else:
					print("Case 4")
					probability = self.p_e/4.0

		return probability

	def f(self, state, action):
		desired_state = copy(state)
		if action == self.A[0]: 
			if state[1] != self.y_dim-1:
				desired_state[1] = desired_state[1] + 1
		elif action == self.A[1]: 
			if state[1] != 0:
				desired_state[1] = desired_state[1] - 1
		elif action == self.A[2]:
			if state[0] != 0:
				desired_state[0] = desired_state[0] - 1
		elif action == self.A[3]:
			if state[0] != self.x_dim-1:
				desired_state[0] = desired_state[0] + 1
		elif action == self.A[4]:
			pass
		return desired_state

	def getAdjacentStates(self, state):
		space = []
		for action in self.A:
			new_state = self.f(state,action)
			if new_state != state:
				space.append(new_state)
		return space

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
	
	starting_state = [1,2]
	state = starting_state

	world = GridWorld(grid_map)
	world.draw(state)

	result = world.pr([1,1],"up",[1,2])
	print(result)

	result = world.getAdjacentStates(state)
	print(result)


if __name__ == '__main__':
	main()