import numpy as np

# GridWorld Class
class GridWorld:
	# Takes the initial state and the map of obstacles represented using a numpy array
	def __init__(self, init_state, worldmap):
		self.state = init_state # List
		self.action = ['u','d','l','r','c']
		self.output = 0
		self.map = worldmap
		self.t_prob = np.zeros_like(worldmap)

		self.x_dim, self.y_dim = self.map.shape

	# Updates the current state to the next state (to be implemented)
	def update(self, action):
		print('Update the state')

	# Draws an ASCII art of the current grid world
	def draw(self):
		print('  ┌'+'─'*(2*self.x_dim+1)+'┐')
		for j in range(self.y_dim-1,-1,-1):
			line = str(j) + ' │ '
			for i in range(self.x_dim):
				if self.state != [i,j]:
					if self.isObstacle(i,j):
						line += 'X ' # Obstacle
					else:
						line += '• ' # Empty Space
				else:
					line += 'O ' # Agent
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
	obstacles = np.zeros((5,5))
	obstacles[1,3] = 1
	obstacles[2,3] = 1
	obstacles[1,1] = 1
	obstacles[2,1] = 1

	starting_state = [2,4]

	world = GridWorld(starting_state, obstacles)
	world.draw()

if __name__ == '__main__':
	main()