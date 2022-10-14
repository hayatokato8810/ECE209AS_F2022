__author__ = "Ankur Kumar, Hayato Kato"

from copy import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from GridWorld import GridWorld

class Policy:
    def __init__(self, grid_world, stochastic=False):
        """
        Assumes we can iterate over state_space and action_space
        """
        self.grid_world = grid_world
        self.gamma = grid_world.gamma
        self.delta = grid_world.converge_threshold
        self.stochastic = stochastic

        self.init()

    def init(self, pi=None):
        if pi is None:
            pi = {}
            n_actions = len(self.grid_world.A)
            for state in self.grid_world.S:
                if not self.stochastic:
                    pi[tuple(state)] = random.choice(list(range(n_actions)))
                else:
                    pi[tuple(state)] = [1. / n_actions] * n_actions
        self.set_policy(pi)

    def set_policy(self, pi):
        self.pi = pi

    def sample(self, state):
        if not self.stochastic:
            return self.pi[tuple(state)]
        else:
            prob = self.pi[tuple(state)]
            return random.choices(self.grid_world.A, prob)

    # assume deterministic policy for policy iteration
    def policy_eval(self, Q, V):
        for state in self.grid_world.S:
            a = self.pi[tuple(state)]
            action = self.grid_world.A[a]
            q_val = 0
            for next_state in self.grid_world.S:
                q_val += self.grid_world.pr(state, action, next_state) * \
                    (self.grid_world.R(state, action, next_state) + self.gamma * V[next_state[0], next_state[1]])
            Q[state[0], state[1], a] = q_val
        return Q

    def policy_update(self, Q):
        actions = np.argmax(Q, axis=-1)
        pi = {
            tuple(state): actions[state[0], state[1]] for state in self.grid_world.S
        }
        self.set_policy(pi)

    def policy_iter(self):
        x_dim = self.grid_world.x_dim
        y_dim = self.grid_world.y_dim
        n_actions = len(self.grid_world.A)

        Q = np.zeros((x_dim, y_dim, n_actions))
        V = np.zeros((x_dim, y_dim))
        iteration = 0
        while True:
            iteration += 1
            # print('iteration', iteration)
            # print(Q)
            Q = self.policy_eval(Q, V)
            self.policy_update(Q)

            # update value
            V_old = V
            V = np.amax(Q, axis=-1)
            if np.linalg.norm(V - V_old) < self.delta:
                self.grid_world.V_star = V
                return iteration

# adapted from GridWorld.py
def main():
    # Define matrix that describes the gridworld
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
    
    # Initial State
    starting_state = [2,2]
    
    # Initialize Gridworld class
    world = GridWorld(grid_map)
    
    # ASCII Visualization of particular state
    world.draw_ASCII(starting_state)
    
    # Displays all transition probabilities for a given state and action pair
    world.compute_P(starting_state,'right',plot=True)
    
    policy = Policy(world)
    
    # Runs Value Iteration and finds the optimal value function
    # converge_k, d = world.compute_V_star(plot=False)
    iteration = policy.policy_iter()
    #print(converge_k)
    world.plot(world.V_star, iteration, blocking=True)

if __name__ == '__main__':
	main()
