#!/usr/bin/env python

""" Numberline.py """

__author__ = "Hayato Kato"

import MDP

from copy import copy
import numpy as np
import matplotlib.pyplot as plt

class NumberLine(MDP.MDP):
	def __init__(self):
		# State Space
		self.S = []

	def isConnected(self, s1, s2):
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




def main():
	print("starting")

	n = NumberLine()
	print(n.isConnected((0,0),(1,1)))			# The two states are connected via "straight" line
	print(n.isConnected((0,0),(2,-2))) 		# Has negative time
	print(n.isConnected((0,0.4),(0.2,0))) # No direct path between the two states

if __name__ == '__main__':
	main()


