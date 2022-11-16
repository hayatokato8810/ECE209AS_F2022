#!/usr/bin/env python

""" Numberline.py """

__author__ = "Hayato Kato"

import MDP

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import random

class NumberLine(MDP.MDP):
	def __init__(self, start_state, end_state, x_dim, y_dim, x_delta, y_delta, force):
		self.start_state = start_state
		self.end_state = end_state
		# Vertices
		self.V = [start_state]
		self.x_dim, self.y_dim = x_dim, y_dim
		#self.x_min, self.y_min = x_min, y_min
		#self.x_max, self.y_max = x_max, y_max
		self.x_delta, self.y_delta = x_delta, y_delta
		self.force = force #list of possible forces
		self.mass = 1
		self.t_max = 20
		self.amplitude = 1
		self.vmax = 10
		self.pc = 0.1
		self.particles = None
		
		# Edges
		self.E = []
	
	def y_next(self, y, v):
		return y + v

	def v_next(self, v, input, field):
		roll = np.random()
		if roll < (abs(v) - self.y_dim * self.pc / self.y_dim):
			velocity = 0
		else:
			velocity = v + (1/self.mass)*(input+field) + np.random.normal(0, (0.1*v),1)
		return velocity

	def field(self, y, amplitude): #assuming field = Acos(y)
		return amplitude * np.sin(y)

	def is_close(self, s1, s2):
		if self.find_distance(s1, s2) < 2:
			return True
		else:
			return False
		
	def is_connected(self, s1, s2):
		y1 = s1[0]
		y2 = s2[0]
		v1 = s1[1]
		v2 = s2[1]

		for f in self.force:
			t = 0
			y = y1
			v = v1
			for t in range(1, self.t_max):
				static_field = self.field(y, self.amplitude)
				y = self.y_next(y,v)
				v = self.v_next(v, f, static_field)
				if self.is_close(s2, (y,v)):
					return True
		return False

	def find_distance(self, p1, p2): #takes two 2D points
		distance = np.sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]))
		return distance
	
	def find_closest(self, p1):
		closest_v = self.start_state
		closest_d = self.find_distance(p1, closest_v)
		for v in self.V:
			temp_d = self.find_distance(v, p1)
			if temp_d < closest_d:
				closest_d = temp_d
				closest_v = v
		return closest_v

	def prm(self):
		while self.end_state not in self.V: #
		#for i in range(1000):
			test_v = None
			if random.random() > 0.9:
				test_v = copy(self.end_state)
			else:
				test_v = ((2*random.random()-1)*self.x_dim, (2*random.random()-1)*self.y_dim)
			#print(test_v)

			closest_v = self.find_closest(test_v)

			if self.is_connected(closest_v,test_v):
				#print('connected')
				self.V.append(test_v)
				self.E.append((closest_v,test_v))

				self.plotEdges(False)

	def plotEdges(self, blockingStatus):
		markerSize = 50
		plt.close()
		fig, ax = plt.subplots(1,1)
		for start_point, end_point in self.E:
			plt.plot([start_point[0],end_point[0]],[start_point[1],end_point[1]])
		plt.scatter(self.start_state[0],self.start_state[1],markerSize,color='b')
		plt.scatter(self.end_state[0],self.end_state[1],markerSize,color='r')
		ax.grid(color='k', linestyle='-', linewidth=1)
		ax.set_xlim([-self.x_dim,self.x_dim])
		ax.set_ylim([-self.y_dim,self.y_dim])
		ax.set_aspect('equal')
		ax.set_title('Number of Edges = ' + str(len(self.E)))
		ax.set_xlabel('Position')
		ax.set_ylabel('Velocity')
		plt.show(block=blockingStatus)
		plt.pause(.1)

	def particle_filter(self, Nparticles, random: bool):
		particles = np.array((Nparticles, 3))
		particles[:,2] = np.ones((1, Nparticles))/Nparticles
		if random:
			particles[:,0] = np.random.uniform(-1*self.x_dim, self.x_dim, Nparticles)
			particles[:,1] = np.random.uniform(-1*self.y_dim, self.y_dim, Nparticles)
		else:
			particles[:,0] = np.ones((1,Nparticles))*self.start_state[0]
			particles[:,1] = np.ones((1,Nparticles))*self.start_state[1]
		self.particles = particles
		return particles
	
	#def update_state(self, action): #returns next state
	#def simulate(self, state, action): #returns observation 
	#step 1: update states of each particle given action
	#step 2: update weights of each particle given observation from robot
	#robot state is highest weight particle
	#someone needs to write global class to keep track of robot xd

    def update_particle_states(self, particles, action):
        particles[:, 0] += particles[:, 1]
        particles[:, 1] += action
        collision_prob = (abs(particles[:, 1]) - self.vmax) * self.pc / self.vmax
        mask = np.random.uniform(size=len(particles)) > collision_prob
        particles[:, 1] *= mask
        return particles

    def update_particle_weights(self, particles, obs):
        # sigma = 0.5v
        sigma = 0.5 * particles[:, 1]
        var = sigma ** 2
        p_o_si = -((obs - particles[:, 0]) ** 2) / var
        p_o_si = np.exp(p_o_si) / np.sqrt(2 * np.pi * var)
        particles[:, 2] = (p_o_si * particles[:, 2]) / np.sum(p_o_si * particles[:, 2])
        return particles

		


			

def main():
	print("starting")
#<<<<<<< HEAD
	force = [x/10 for x in range(-10, 10, 1)]
	n = NumberLine((9,9),(1,1),10,10,0.1,0.1, force)
#=======

#	n = NumberLine((0,0),(9,9),10,10)
#>>>>>>> 988e254 (Added Observation probabilities)
	#print(n.is_connected((0,0),(1,1)))			# The two states are connected via "straight" line
	#print(n.is_connected((0,0),(2,-2))) 		# Has negative time
	#print(n.is_connected((0,0.4),(0.2,0))) # No direct path between the two states

	n.prm()

	n.plotEdges(True)

	print('finished')

if __name__ == '__main__':
	main()


