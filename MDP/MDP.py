#!/usr/bin/env python

""" MDP.py """

__author__ = "Hayato Kato"

from copy import copy
import numpy as np
import matplotlib.pyplot as plt

class MDP(object):
	def __init__(self, S, A, P, R, H, g):
		self.S = S
		self.A = A
		self.P = P
		self.R = R
		self.H = H
		self.gamma = g

		self.V = np.zeros(len(S))
		self.threshold = 0.00001

	def valueIteration(self):
		N_S = len(self.S)
		N_A = len(self.A)
		new_V = copy(self.V)
		diff = []
		k = 0
		while True:
			for s in range(N_S):
				max_V = 0
				for a in range(N_A):
					summation = 0
					for s_ in range(N_S):
						summation += self.P[s,a,s_]*(self.R[s,a,s_]+self.gamma*self.V[s_])
					if max_V < summation:
						max_V = summation
				new_V[s] = max_V
			temp = np.linalg.norm(self.V - new_V)
			diff.append(temp)
			if temp < self.threshold:
				break
			self.V = copy(new_V)
			k += 1
		return diff, k

def main():
	print("starting MDP")

if __name__ == '__main__':
	main()

