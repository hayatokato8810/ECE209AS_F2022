#!/usr/bin/env python

""" Numberline.py """

__author__ = "Hayato Kato"

import MDP

from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import random

class NumberLine(MDP.MDP):
	def __init__(self, start_state, end_state, x_dim, y_dim):
		self.start_state = start_state
		self.end_state = end_state
		# Vertices
		self.V = [start_state]
		self.x_dim, self.y_dim = x_dim, y_dim

		# Edges
		self.E = []

	def is_connected(self, s1, s2):
		y1 = s1[0]
		v1 = s1[1]
		y2 = s2[0]
		v2 = s2[1]
		try:
			a = (v2-v1)*(v2+v1)/(2*(y2-y1)+(v2-v1))
			t = (v2-v1)/a
			if t >= 0:
				return a,t
			else:
				return None
			pass
		except:
			return None

	def findDistance(self, p1, p2): #takes two 2D points
		distance = np.sqrt((p2[0]-p1[0]**2) + (p2[1]-p1[1]**2))
		return distance
	
	def find_closest(self, p1):
		closest_vertex = ()
		closest_distance = np.Inf
		for vertex in self.V:
			temp_distance = self.find_distance(vertex, p1)
			if temp_distance < closest_distance:
				closest_distance = temp_distance
				closest_vertex = vertex
		return closest_vertex

	def prm(self):
		while self.end_state not in self.V:
			test_v = None
			if random.random() > 0.9:
				test_v = self.end_state
			else:
				test_v = (random.random()*self.x_dim, random.random()*self.y_dim)
			closest_v = self.find_closest(test_v)
			if self.is_connected(test_v, closest_v):
				self.V.append(test_v)
				self.E.append((closest_v,test_v))





def main():
	print("starting")

	n = NumberLine((1,1),(11,11),12,12)
	print(n.is_connected((0,0),(1,1)))			# The two states are connected via "straight" line
	print(n.is_connected((0,0),(2,-2))) 		# Has negative time
	print(n.is_connected((0,0.4),(0.2,0))) # No direct path between the two states

if __name__ == '__main__':
	main()


