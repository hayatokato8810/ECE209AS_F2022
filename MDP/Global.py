#!usr/bin/env python

import numpy as np
import random
import math

class Global(object):
    def __init__(self, worldmap, p_e, start_state):
        self.x_dim, self.y_dim = worldmap.shape
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                self.S.append((i,j))
        self.S_obs = []
        self.S_iceD = []
        self.S_iceS = []
        for state in self.S:
            if worldmap[state] == 1:
                self.S_obs.append(state)
            elif worldmap[state] == 2:
                self.S_iceD.append(state)
            elif worldmap[state] == 3:
                self.S_iceS.append(state)

        self.A = ['up','down','left','right','center']
        self.p_e = p_e
        self.state = start_state

    #converts action to change state
    def f(self, action):
        desired_state = list(self.state)
        if action == self.A[0]: 
            if self.state[1] != self.y_dim-1:
                desired_state[1] += 1
            else:
                return None
        elif action == self.A[1]: 
            if self.state[1] != 0:
                desired_state[1] -= 1
            else:
                return None
        elif action == self.A[2]:
            if self.state[0] != 0:
                desired_state[0] -= 1
            else:
                return None
        elif action == self.A[3]:
            if self.state[0] != self.x_dim-1:
                desired_state[0] += 1
            else:
                return None
        elif action == self.A[4]:
            pass
        else:
            print("Not a valid action")
            exit()
        return tuple(desired_state)

    #checks if you got the action you wanted or a wrong one
    def next_action(self, action):
        real_action = action
        roll = random.random()
        if roll < (1-self.p_e):
            pass
        else:
            alternate = self.A
            alternate.remove(action)
            real_action = random.choice(alternate)
        return real_action

    def S_adj(self, state):
        space = []
        for action in self.A:
            new_state = self.f(state,action)
            if new_state != None:
                space.append(new_state)
        return space

    #update state with the move
    def next_move(self, real_action):
        next_state = self.f(real_action)
        if next_state not in self.S_obs and self.state not in self.S_obs and next_state in self.S_adj(self.state):
            self.state = next_state
            return next_state
        else: #failed, do nothing
            return self.state

    def return_obs(self, state):
        observation = 0
        dD = min(math.dist(list(s),list(state)) for s in self.S_iceD)
        dS = min(math.dist(list(s),list(state)) for s in self.S_iceS)
        try:
            h = 2/(1/dD+1/dS)
        except:
            h = 0
        if h == 0:
            return observation
        else:
            roll = random.random()
            if roll < 1 - (math.ceil(h) - h):
                observation = math.ceil(h)
            else:
                observation = math.floor(h)
        return observation

    
    #generate observation function o(h) for each state
    #highroll observation
    def oLUT1(self):
        observations = []
        for state in self.S: #(i,j) pairs
            dD = min(math.dist(list(s),list(state)) for s in self.S_iceD)
            dS = min(math.dist(list(s),list(state)) for s in self.S_iceS)
            try:
                h = 2/(1/dD+1/dS)
            except:
                h = 0
            o = 1 - (math.ceil(h)-h)
            observations.append(o)
        return observations
	
    #lowroll observation
    def oLUT2(self):
        observations = []
        for state in self.S: #(i,j) pairs
            dD = min(math.dist(list(s),list(state)) for s in self.S_iceD)
            dS = min(math.dist(list(s),list(state)) for s in self.S_iceS)
            try:
                h = 2/(1/dD+1/dS)
            except:
                h = 0
            o = (math.ceil(h)-h)
            observations.append(o)
        return observations


