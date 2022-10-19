#!/usr/bin/env python

import numpy as np

class Graph(object):
    def __init__(self, V, E):
        self.V = V
        self.E = E

    def bfs(self,start,end):
        queue = []
        visited = []
        visited.append(start)
        queue.append(start)
        steps = 0

        while queue:
            steps += 1
            node = queue.pop(0)
            for neighbor in E[node]:
                if neighbor == end:
                    return steps
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)

class Board(object):
    def __init__(self, piece):
        self.piece = piece
        self.vertices = []
        self.edges = {}
        
        self.makeV()
        self.makeE()
    
    def makeV(self):
        for i in range(8):
            for j in range(8):
                self.vertices.append((i,j))
    
    def makeE(self):
        for vertex in self.vertices:
            moves = []
            for action in self.piece:
                move = (vertex[0]+action[0], vertex[1]+action[1])
                if move in self.vertices:
                    moves.append(move)
            self.edges[vertex] = moves

    def dump(self):
        #print(self.vertices)
        #print(len(self.vertices))
        print(self.edges)
        total = 0
        for i in self.edges:
            total += len(self.edges[i])
            print(len(self.edges[i]))
        print(total)


def main():
    piece = ((1,2),(1,-2),(2,1),(2,-1),(-1,2),(-1,-2),(-2,1),(-2,-1))
    board = Board(piece)
    board.dump()
    
main()                
