#!/usr/bin/env python

""" GridWorld.py """

__author__ = "Hayato Kato"

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
				result.append((i,j))
	return result

# GridWorld Class
class GridWorld:
	# Takes the initial state and the map of obstacles represented using a numpy array
	def __init__(self, worldmap):
		self.state_space = convert2Set(worldmap, isAlwaysTrue)
		self.action_space = ['u','d','l','r','c']

		self.obstacle_space = convert2Set(worldmap, isObstacle)
		self.road_space = convert2Set(worldmap,isRoad)
		self.icecream_D_space = convert2Set(worldmap, isIcecreamShop_D)
		self.icecream_S_space = convert2Set(worldmap, isIcecreamShop_S)

		self.output = 0
		self.map = worldmap
		self.t_prob = np.zeros_like(worldmap)

		self.x_dim, self.y_dim = self.map.shape

	# Updates the current state to the next state (to be implemented)
	def update(self, action):
		print('Update the state')

	def pr(self, state, action, next_state):
		probability = 0


		return probability

	# Draws an ASCII art of the current grid world
	def draw(self, state):
		print('  ┌'+'─'*(2*self.x_dim+1)+'┐')
		for j in range(self.y_dim-1,-1,-1):
			line = str(j) + ' │ '
			for i in range(self.x_dim):
				if state != [i,j]:
					if (i,j) in self.obstacle_space:
						line += 'X ' # Obstacle
					elif (i,j) in self.road_space:
						line += '# ' # Road
					elif isIcecreamShop_D(self.map,(i,j)):
						line += 'D ' # Road
					elif isIcecreamShop_S(self.map,(i,j)):
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

	# Checks if a particular state is occupied by an obstacle
	def isObstacle(self, x, y):
		return self.map[x,y] == 1

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

	starting_state = [2,4]
	state = starting_state

	world = GridWorld(grid_map)
	world.draw(state)

	#obstacle_set = convert2Set(obstacle_map,1)
	#print(state in world.obstacle_space)
	#print(world.state_space)

if __name__ == '__main__':
	main()