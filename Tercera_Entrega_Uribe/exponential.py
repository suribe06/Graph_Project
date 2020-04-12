"""
Proyecto AGRA 2019-1 Santiago Uribe P. Cod: 8925546
"""

import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class Block: #Individual block
    def __init__(self, id, time, dad):
        self.id = id
        self.time = time
        self.dad = dad
        return

    def getId(self):
        return self.id

    def getTime(self):
        return self.time

    def getDad(self):
        return self.dad

class Network: #Block Chain network
    def __init__(self):
        self.network = []
        root = Block(0, 0, None)
        self.network.append([])
        self.network[0].append(root)
        return

    def add_Block(self, b):
        self.network.append([])
        d = b.getDad()
        self.network[d].append(b) #agrego el bloque en la lista que tiene el indice del padre
        return

    def make_Graph(self):
        """
        Esta funcion construye el grafo de la red (network), para asi con el grafo resultante
        poder hallar sus metricas respectivas
        """
        G = [[] for _ in range(len(self.network))]
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                dad = self.network[i][j].getDad()
                id = self.network[i][j].getId()
                if dad != None:
                    G[dad].append(id)

        return G

def delayB(p): #Dp()
    y = random.random()
    delay = -p * math.log(1 - y)
    return delay

def timeB(): #T()
    y = random.random()
    time = -1 * math.log(1 - y)
    return time

def bfs(src, G): #BFS para hallar rama mas larga de la red (la rama valida) y los bloques perdidos
    global vis, depth
    q = deque()
    q.append(src)
    vis[src] = 1
    while len(q):
        u = q.popleft()
        for v in G[u]:
            depth[v] = depth[u] + 1 #asigno la profundidad correspondiete a cada nodo
            if vis[v] == 0:
                vis[v] = 1
                q.append(v)

    valid_branch_length = max(depth) + 1
    lost_blocks = len(G) - valid_branch_length

    return valid_branch_length, lost_blocks

def network_stats(G):
    global vis, depth
    n = len(G)
    vis = [0 for _ in range(n)]
    depth = [0 for _ in range(n)]
    branch_length, lost_blocks = bfs(0, G)

    return branch_length, lost_blocks

def generator(nodes, cant):
    p_small = [i for i in np.arange(0, 2.05, 0.05)]
    length_network = []
    lost_B = []
    time = 0
    for p in p_small:
        cont1, cont2 = 0, 0
        for i in range(cant):
            n = Network()
            cont_blocks = 1 #contador para asignar los ids de los bloques
            block_times = [0] #voy guardando los tiempos para saber en que parte anexar el siguiente bloque dependiendo del delay (encontrar el padre)
            for j in range(1, nodes):
                nt = timeB()
                time += nt
                block_times.append(time)
                delay = delayB(p) #calculo hasta donde ve el nuevo bloque donde va la red
                aware = time - delay
                dad = None
                if aware < 0:
                    dad = 0
                else:
                    for i in range(len(block_times)):
                        if block_times[i] < aware:
                            dad = i
                        elif block_times[i] > aware:
                            break
                block = Block(cont_blocks, time, dad)
                cont_blocks += 1
                n.add_Block(block)

            networkGraph = n.make_Graph()
            v1, v2 = network_stats(networkGraph)
            cont1 += v1
            cont2 += v2

        length_network.append(cont1 / cant)
        lost_B.append(cont2 / cant)

    return length_network, lost_B, p_small

def graficar(m1, m2, p):
    plt.title("Block Chain Stats")
    plt.xlabel("p")
    plt.ylabel("f(p)")
    plt.stem(p, m1, linefmt='blue', markerfmt='blue', label='Valid branch length')
    plt.stem(p, m2, linefmt='green', markerfmt='green', label='Lost blocks')
    plt.legend()
    plt.show()
	#plt.savefig("Stats_3.png", format="PNG")

def main():
    #Exercise 3.1
    m1, m2, p = generator(10000, 1)
    graficar(m1, m2, p)

    return

main()
