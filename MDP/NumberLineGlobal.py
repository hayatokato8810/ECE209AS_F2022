#!usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class NumberLineGlobal(object):
    def __init__(self, start_state, yrange, vrange, frange): #yrange, vrange, frange tuples of min,max
        self.y = start_state[0]; self.v = start_state[1]
        self.ymin = yrange[0]; self.ymax = yrange[1]
        self.vmin = vrange[0]; self.vmax = vrange[1]
        self.fmin = frange[0]; self.fmax = frange[1]
        self.pc = 0
        self.mass = 1
        self.input = 0
        self.observation = 0
        self.particles = self.particle_filter(1000, True)

    def particle_filter(self, Nparticles, random: bool): #only call this while initializing
        particles = np.zeros((Nparticles, 3))
        particles[:,2] = 1/Nparticles
        if random:
            particles[:,0] = np.random.uniform(self.ymin, self.ymax, Nparticles)
            particles[:,1] = np.random.uniform(self.vmin, self.vmax, Nparticles)
        else:
            particles[:,0] = np.ones((1,Nparticles))*self.y
            particles[:,1] = np.ones((1,Nparticles))*self.v
        
        return particles     

    def y_next(self):
        return self.y + self.v

    def v_next(self, input, field):
        roll = np.random.random()
        if roll < (abs(self.v) - self.vmax) * self.pc / self.vmax:
            print("boom")
            velocity = 0
        else:
            self.wobble = np.random.normal(0, abs(0.1*self.v))
            velocity = self.v + (1/self.mass)*(input+field) + self.wobble
        return velocity

    def field(self, amplitude): #assuming field = Acos(y)
        return amplitude * np.sin(self.y)

    def force(self): #random input force
        f = np.random.uniform(self.fmin, self.fmax)
        print(f)
        return f

    def observe(self):
        return self.y + np.random.normal(0, abs(0.5*self.v), 1)

    def next(self):
        self.input = self.force()
        self.y = self.y_next()
        self.v = self.v_next(self.input, self.field(1))
        self.observation = self.observe()
        self.update_particles()
        self.plot2d()
        # print(self.particles[:,0])
        # print(self.particles[:,1])

    def update_particles(self):
        self.particles = self.update_particle_states(self.particles, self.input)
        self.particles = self.update_particle_weights(self.particles, self.observation)

    def update_particle_states(self, particles, action):
        particles[:, 0] += particles[:, 1]
        for i in range(len(particles[:, 1])):
            particles[i, 1] += action + self.field(1) + np.random.normal(0, abs(0.1*particles[i,1]))
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

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        ax.scatter(self.particles[:,0], self.particles[:,1], self.particles[:,2], color = "red")
        ax.scatter(self.y, self.v, 1, color = "black")
        ax.set_zlim(0,1)
        ax.set_zlabel("weight")
        plt.xlim((-40, 40)); plt.ylim((-40, 40))
        plt.xlabel("position"); plt.ylabel("velocity")
        plt.title("input = " + str(self.input))
        plt.show()

    def plot2d(self):
        ids = np.argsort(self.particles[:,2])[-5:]
        print([self.particles[id] for id in ids])
        color = ["red"] * len(self.particles)
        c = ["black", "brown" ,"gold", "orange", "blue"]
        for i,id in enumerate(ids):
            color[id] = c[i]
        plt.scatter(self.particles[:,0], self.particles[:,1], s=self.particles[:,2] * 100, color=color)
        plt.scatter(self.y, self.v, s=100, color="green")
        plt.xlim((self.y - 15, self.y + 15)); plt.ylim((self.v-5, self.v+5)) 
        plt.annotate("y = %f, v = %f, w = %f" % (self.particles[ids[-1]][0], self.particles[ids[-1]][1], self.particles[ids[-1]][2]), (self.particles[ids[-1]][0], self.particles[ids[-1]][1]))
        #for i in temp:
            #plt.annotate(i+1, (self.particles[i,0], self.particles[i,1])) #fontsize=min(10, self.particles[i,2] * 100))
        plt.xlabel("position"); plt.ylabel("velocity")
        plt.title("y = %f, v = %f, force = %f" % (self.y, self.v, self.input+self.field(1)))
        plt.show()
#need to plot weights
#need to make all particle noise independent
def main():
    g = NumberLineGlobal((0, 0), (-10, 10), (-10, 10), (-1, 1))
    for t in range(100):
        g.next()

if __name__ == '__main__':
	main()




    


    
