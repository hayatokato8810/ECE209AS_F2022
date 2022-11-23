#!/usr/bin/env python

""" GridWorld.py """

__author__ = "Hayato Kato"

import MDP

from copy import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from Global import Global

class GridWorld(MDP.MDP):
	def __init__(self, worldmap, p_e):
		# State Space
		self.S = []
		self.x_dim, self.y_dim = worldmap.shape
		for i in range(self.x_dim):
			for j in range(self.y_dim):
				self.S.append((i,j))
		self.S_obs = []
		self.S_iceD = []
		self.S_iceS = []
		self.S_road = []
		for state in self.S:
			if worldmap[state] == 1:
				self.S_obs.append(state)
			elif worldmap[state] == 2:
				self.S_iceD.append(state)
			elif worldmap[state] == 3:
				self.S_iceS.append(state)
			elif worldmap[state] == 4:
				self.S_road.append(state)

		self.p_e = p_e
		self.R_D = 1
		self.R_S = 10
		self.R_W = -10

		# Action Space
		self.A = ['up','down','left','right','stay']

		# Transition Probability
		N_S = len(self.S)
		N_A = len(self.A)
		self.P = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			for a in range(N_A):
				for s_ in range(N_S):
					self.P[s,a,s_] = self.pr(self.S[s],self.A[a],self.S[s_])

		# Reward Function
		self.R = np.zeros((N_S,N_A,N_S))
		for s in range(N_S):
			if self.S[s] in self.S_iceD:
				self.R[s,:,:] = self.R_D
			elif self.S[s] in self.S_iceS:
				self.R[s,:,:] = self.R_S
			elif self.S[s] in self.S_road:
				self.R[s,:,:] = self.R_W

		# Horizon
		self.H = 10 # Not currently used

		# Discount Gamma
		self.g = 0.8

		# Observation Space
		self.O = []
		#maxDist = math.sqrt(self.x_dim*self.x_dim+self.y_dim*self.y_dim)
		for i in range(5):
			self.O.append(i)

		super().__init__(self.S,self.A,self.P,self.R,self.H,self.g)

	# System Dynamics
	def f(self, state, action):
		desired_state = list(state)
		if action == self.A[0]: 
			if state[1] != self.y_dim-1:
				desired_state[1] += 1
			else:
				return None
		elif action == self.A[1]: 
			if state[1] != 0:
				desired_state[1] -= 1
			else:
				return None
		elif action == self.A[2]:
			if state[0] != 0:
				desired_state[0] -= 1
			else:
				return None
		elif action == self.A[3]:
			if state[0] != self.x_dim-1:
				desired_state[0] += 1
			else:
				return None
		elif action == self.A[4]:
			pass
		else:
			print("Not a valid action")
			exit()
		return tuple(desired_state)

	# Transition Probabilities
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
					if action != self.A[-1]:
						probability = self.p_e/4.0
		return probability

	# Computes set of all states reachable by a single action
	def S_adj(self, state):
		space = []
		for action in self.A:
			new_state = self.f(state,action)
			if new_state != None:
				space.append(new_state)
		return space

	# Observation Probabilities
	def observePr(self, state, observation, S_iceD, S_iceS):
		dD = min(math.dist(list(s),list(state)) for s in S_iceD)
		dS = min(math.dist(list(s),list(state)) for s in S_iceS)
		try:
			h = 2/(1/dD+1/dS)
		except:
			h = 0
		high_h = math.ceil(h)
		low_h = math.floor(h)
		if observation == high_h:
			probability = 1-(high_h - h)
		elif observation == low_h:
			probability = high_h - h
		else:
			probability = 0
		return probability

	def conditionalObservePr(self, S_iceD, S_iceS):
		pr_s = np.ones((25))/25
		pr_o_s = np.zeros((5,25))
		for o in range(len(self.O)):
			for s in range(len(self.S)):
				pr_o_s[o,s] = self.observePr(self.S[s],self.O[o],S_iceD, S_iceS)
		joint_pr_o_s = pr_o_s * pr_s
		pr_o = np.sum(joint_pr_o_s,1).reshape((5,1))
		# Probability of particular state given observation
		return joint_pr_o_s / pr_o
	
	''' Visualization Methods '''

	# Draw ASCII art of current state
	def drawState(self, state):
		print('  ┌'+'─'*(2*self.x_dim+1)+'┐')
		for j in range(self.y_dim-1,-1,-1):
			line = str(j) + ' │ '
			for i in range(self.x_dim):
				if state != (i,j):
					if (i,j) in self.S_obs:
						line += 'X ' # Obstacle
					elif (i,j) in self.S_road:
						line += '# ' # Road
					elif (i,j) in self.S_iceD:
						line += 'D ' # Road
					elif (i,j) in self.S_iceS:
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

	# Plot all transition probabilities given particular state and action
	def plotProbability(self, probabilities, time=None, blocking=True, **kwarg):
		probabilities = probabilities.reshape(5,5).transpose()
		fig, ax = plt.subplots(1,1)
		for (j,i),label in np.ndenumerate(probabilities):
		    ax.text(i,j,round(label,4),ha='center',va='center')
		im = ax.imshow(probabilities, origin = 'lower',**kwarg)
		ax.set_xticks(np.arange(-.5, self.x_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		if time != None:
			ax.set_title('Time t = ' + str(time))
		plt.colorbar(im)
		plt.show(block = blocking)

	# Plot optimal value function across all states
	def plotValue(self):
		matrix = self.V.reshape(5,5).transpose()
		fig, ax = plt.subplots(1,1)
		for (j,i),label in np.ndenumerate(matrix):
		    ax.text(i,j,round(label,4),ha='center',va='center')
		im = ax.imshow(matrix, cmap="RdBu", origin = 'lower')
		ax.set_xticks(np.arange(-.5, self.x_dim, 1), minor=True)
		ax.set_yticks(np.arange(-.5, self.y_dim, 1), minor=True)
		ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
		plt.colorbar(im)
		plt.show()

	def bayesFilter(self, est_pr, pr_s_o, action, observation):
		belief_pr = np.matmul(self.P[:,self.A.index(action),:].reshape((25,25)).transpose(),copy(est_pr).reshape((25,1)))
		temp = pr_s_o[observation].reshape((25,1)) * belief_pr
		belief_pr = temp / np.linalg.norm(temp,1)
		return belief_pr

def main():
	print("start")

	# Define Gridworld map via matrix representation
	grid_map = np.zeros((5,5))
	# Obstacles
	grid_map[1,3] = 1
	grid_map[2,3] = 1
	grid_map[1,1] = 1
	grid_map[2,1] = 1
	# Icecream Shops
	grid_map[2,2] = 2
	grid_map[2,0] = 3
	# Road
	grid_map[4,0] = 4
	grid_map[4,1] = 4
	grid_map[4,2] = 4
	grid_map[4,3] = 4
	grid_map[4,4] = 4

	# Define GridWorld object
	world = GridWorld(grid_map,0.4)
	# Plot transition probabilities when starting from state #17 (x=3,y=2) & moving up
	#world.plotProbability(world.P[17,0,:])

	# Value Iteration
	#print(world.valueIteration())
	#world.plotValue()

	# Policy Iteration
	#world.policyIteration()
	#world.plotValue()

	start_state = (2,4)
	
	# Compute Conditional Probability pr(o|s)
	print(world.O)
	print(world.observePr(start_state,2,world.S_iceD, world.S_iceS))
	world.drawState(start_state)

	actions = ['left','left','left','down','down','down','right','right','right']
	observations = [3,3,4,4,3,2,2,1,0]

	# Optimization Problem
	maxConfidence = 0
	actualIceD = [(0,0)]
	actualIceS = [(0,1)]


	for iceD in world.S:
		if iceD not in world.S_obs:
			for iceS in world.S:
				if iceS not in world.S_obs and iceS != iceD:
					pr_s_o = world.conditionalObservePr([iceD], [iceS])

					# Bayes Filtering
					est_pr = np.ones((25,1))/25
					for t in range(len(actions)):
						est_pr = world.bayesFilter(est_pr, pr_s_o, actions[t], observations[t])
					confidence = max(est_pr)
					print(confidence)
					if maxConfidence < confidence:
						maxConfidence = confidence
						actualIceD = iceD
						actualIceS = iceS

	print(actualIceD)
	print(actualIceS)

	pr_r_o_s = np.zeros((25*24,5,25))
	i = 0
	for iceD in world.S:
		if iceD not in world.S_obs:
			for iceS in world.S:
				if iceS not in world.S_obs and iceS != iceD:
					pr_r_o_s[i,:,:] = world.conditionalObservePr([iceD], [iceS])
					i += 1
	print(pr_r_o_s.shape)

	est_pr = np.ones((25,1))/25
	reward_pr = np.ones((25*24,1))/(25*24)
	for t in range(len(actions)):
		i = 0
		for iceD in world.S:
			if iceD not in world.S_obs:
				for iceS in world.S:
					if iceS not in world.S_obs and iceS != iceD:
						belief_pr = world.bayesFilter(est_pr, pr_r_o_s[i,:,:],actions[t], observations[t])
						#est_pr += reward_pr[i]*belief_pr
						reward_pr[i] *= max(belief_pr)
						i += 1
		print(t)
		#est_pr = np.matmul(reward_pr, est_pr)
		reward_pr = reward_pr / np.linalg.norm(reward_pr,1)
		world.plotProbability(est_pr, time=t, blocking=True,vmin=0,vmax=1)

# ???



	'''
	g = Global(grid_map, 0.4, start_state)
	while (g.state != g.S_iceD or g.state != g.S_iceS):
		#initialize observation to be something
		#gridworld agent should propose a move on this line called "agent_action"
		real_action = g.next_action(agent_action)
		true_nextstate = g.next_move(real_action)
		observation = g.return_obs(true_nextstate)
		#this observation feeds back into the gridworld agent, repeat loop
	'''



	#for i in range(5):
	#	world.plotProbability(pr_s_o[i])


if __name__ == '__main__':
	main()