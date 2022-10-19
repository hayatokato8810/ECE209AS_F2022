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
        parent = {}
        while queue:
            steps += 1
            node = queue.pop(0)
            for neighbor in self.E[node]:
                if neighbor == end:
                    parent[neighbor] = node
                    break
                if neighbor not in visited:
                    parent[neighbor] = node
                    visited.append(neighbor)
                    queue.append(neighbor)
        
        path = []
        while(end != start):
            path.append(end)
            end = parent[end]
        path.append(start)
        path.reverse()
        return path

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
    g = Graph(board.vertices,board.edges)  
    path = g.bfs((0,0),(7,7))
    print(path) 

    
main()                
