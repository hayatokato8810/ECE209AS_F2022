import numpy as np
import matplotlib.pyplot as plt

class Value:
	def __init__(self, states, dim):
		self.V = states.flatten()
		self.X_dim = dim[0]
		self.Y_dim = dim[1]

	def plot(self, iteration):
		values = self.V.reshape((self.X_dim, self.Y_dim)).transpose()

		fig, ax = plt.subplots(1,1)
		for (j,i),label in np.ndenumerate(values):
		    ax.text(i,j,label,ha='center',va='center')
		im = ax.imshow(values, origin = 'lower')
		ax.set_xticks(np.arange(-.5, self.X_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.Y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		ax.set_title('Value Iteration t=' + str(iteration))
		plt.colorbar(im)
		plt.show()

def main():
	gridworld = np.zeros((5,5))

	states = gridworld.flatten()

	v = Value(states,(5,5))
	v.plot(0)

if __name__ == '__main__':
	main()
