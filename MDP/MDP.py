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

		N_S = len(self.S)
		N_A = len(self.A)
		self.V = np.zeros(N_S)
		self.Pi = np.zeros(N_S)
		self.Q = np.zeros((N_S,N_A))
		self.threshold = 0.00001

	# Value Iteration
	def valueIteration(self):
		N_S = len(self.S)
		N_A = len(self.A)
		self.V = np.zeros(N_S)
		self.Q = np.zeros((N_S,N_A))
		new_V = np.zeros(N_S)
		diff = []
		k = 0
		# Iterate until convergence
		while True:
			# Iterate through all transitions
			for s in range(N_S):
				max_V = 0
				for a in range(N_A):
					q_val = 0
					for s_ in range(N_S):
						# Compute the max value
						q_val += self.P[s,a,s_]*(self.R[s,a,s_]+self.gamma*self.V[s_])
					self.Q[s,a] = q_val
				new_V = np.amax(self.Q, axis=-1)
			temp = np.linalg.norm(self.V - new_V)
			diff.append(temp)
			if temp < self.threshold:
				break
			self.V = copy(new_V)
			k += 1
		return diff, k

	# Policy Iteration
	def policyIteration(self):
		N_S = len(self.S)
		N_A = len(self.A)
		self.V = np.zeros(N_S)
		self.Pi = np.zeros(N_S)
		self.Q = np.zeros((N_S,N_A))
		new_V = np.zeros(N_S)
		diff = []
		k = 0
		# Iterate until convergence
		while True:
			for s in range(N_S):
				a = int(self.Pi[s])
				q_val = 0
				for s_ in range(N_S):
					# Compute both the max value and optimal policy
					q_val += self.P[s,a,s_]*(self.R[s,a,s_]+self.gamma*self.V[s_])
				self.Q[s,a] = q_val
			self.Pi = np.argmax(self.Q,axis=-1)
			new_V = np.amax(self.Q, axis=-1)
			temp = np.linalg.norm(self.V - new_V)
			diff.append(temp)
			if temp < self.threshold:
				break
			self.V = copy(new_V)
			k += 1
		return diff,k

def main():
	print("starting MDP")

if __name__ == '__main__':
	main()

