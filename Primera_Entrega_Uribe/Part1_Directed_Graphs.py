"""
Proyecto AGRA 2019-1 Santiago Uribe P. Cod: 8925546
"""

import random
import matplotlib.pyplot as plt
import networkx as nx
import csv
import numpy

#Strongly Connected Components with Kosaraju
def dfs(u, num, G):
	global vis, scc, edges
	vis[u] = 1
	scc[u] = num
	for v in G[u]:
		if(vis[v] == 0):
			dfs(v,num, G)
		if scc[v] != -1 and scc[u] != scc[v]:
			edges += 1

def dfs_list(u):
	global L, vis, I
	vis[u] = 1
	for v in I[u]:
		if(vis[v] == 0):
			dfs_list(v)
	L.append(u)

def kosaraju(G):
	global L, I, scc, vis, edges
	n = len(G)
	scc = [-1 for i in range(n)]

	I = [[] for i in range(n)]
	for i in range(n):
		for j in G[i]:
			I[j].append(i)
	vis = [0 for i in range(n)]
	L = []
	for i in range(n):
		if(vis[i] == 0):
			dfs_list(i)
	vis = [0 for i in range(n)]
	cont = 0
	edges = 0
	while(len(L)):
		i = L.pop()
		if(vis[i] == 0):
			dfs(i,cont, G)
			cont +=	1

	return scc, edges

#Semilla
random.seed(23)

def random_graph(n, p):
    Edges = []
    for i in range(n):
        for j in range(n):
            if random.random() < p:
                Edges.append((i, j))

    G2 = [[] for _ in range(n)]
    G = nx.DiGraph(directed=True)
    for i in range(n): G.add_node(i)
    G.add_edges_from(Edges)

    for i,j in Edges: G2[i].append(j)

    return G, G2

def graph_stats(G):
	numberSCC, averageSCC = 0, 0
	label_scc, edgesSCC = kosaraju(G)
	numberSCC = max(label_scc) + 1
	averageSCC = int(len(G) / numberSCC)

	return numberSCC, averageSCC, edgesSCC

#Functions Exercise 1.2
def generator(n, cant):
	p_small = [i for i in numpy.arange(0, 0.1, 0.005)]
	number = []
	averages = []
	edges = []
	i = 0
	for i in p_small:
		cont1, cont2, cont3 = 0, 0, 0
		for j in range(cant):
			G, G2 = random_graph(n, i)
			val1, val2, val3 = graph_stats(G2)
			cont1 += val1
			cont2 += val2
			cont3 += val3
		number.append(cont1/cant)
		averages.append(cont2/cant)
		edges.append(cont3/cant)

	return number, averages, edges, p_small

def graficar(val1, val2, val3, p):
	plt.title("Graph Stats")
	plt.xlabel("p")
	plt.ylabel("f(100, p)")
	plt.plot(p, val1, label='Number Of SCC')
	plt.plot(p, val2, label='Average Size SCC')
	plt.plot(p, val3, label='Edges SCC')
	plt.legend()
	plt.show()
	#plt.savefig("Stats2.png", format="PNG")

#Functions Exercise 1.3
def func_aux(n, p):
	cont1 = 0
	for i in range(1000):
		G, G2 = random_graph(n, p)
		val1, val2, val3 = graph_stats(G2)
		cont1 += val3
	ans =  cont1 / 1000
	return ans

def divideConquer(lo, hi, mx):
	mid = (lo + hi) / 2
	ps = [lo, mid-0.001, mid, mid+0.001, hi]
	newY = []
	for i in ps:
		r = func_aux(100, i)
		newY.append(r)
	m = max(newY)
	ind = newY.index(m)
	if m >= mx:
		if m == newY[1]:
			return divideConquer(mid, hi, m)
		elif m == newY[3]:
			return divideConquer(lo, mid, m)
		elif m == newY[2]:
			return divideConquer(lo+0.001, hi-0.001, m)
	else:
		return ps[ind]

def maxiumOfEdges(edges, p_small):
	largest = max(edges)
	h = edges.index(largest)
	lo = p_small[h-1]
	hi = p_small[h+1]
	ans = divideConquer(lo, hi, largest)
	return ans

def derivative(x, y):
	result = []
	for i in range(3, len(y)):
		x1, y1 = x[i-1], y[i-1]
		x2, y2 = x[i], y[i]
		d = (y2 - y1) / (x2 - x1)
		result.append(d)

	return result

def inflectionPoint(p_small, avergaSize):
	firstDeri = derivative(p_small, avergaSize)
	secondDeri = derivative(p_small, firstDeri)
	ilo, ihi = 0, 0
	i = 1
	flag = True
	while i < len(secondDeri) and flag == True:
		ant = secondDeri[i - 1]
		sig = secondDeri[i]
		if ant > 0 and sig < 0 or ant < 0 and sig > 0:
			ilo = i-1
			ihi = i
			flag = False
		i += 1
	lo = p_small[ilo+3]
	hi = p_small[ihi+3]

	return lo, hi

def main():
	#Exercise 1.1
	"""graph, graph2 = random_graph(15, 0.2)
	nx.draw_circular(graph, with_labels=True, arrows=True)
	plt.savefig("Directed_Graph.png", format="PNG")
	m1, m2, m3 = graph_stats(graph2)
	print("Numero de SCC: {0}, Promedio de nodos en SCC: {1}, Numero de Arcos que conectan SCC: {2}".format(m1, m2, m3))
	"""
	#Exercise 1.2
	v1, v2, v3, ps = generator(100, 200)
	graficar(v1, v2, v3, ps)
	#print(ps)
	#print(v2)
	"""
	#Guardado en archivo csv
	with open('data.csv', 'w', newline = '') as f:
		writerF = csv.writer(f)
		writerF.writerow(['p_small value ', ' Mean number of SCC', ' Mean average size of SCC', ' Mean of edges that connect SCC'  ])
		for i in range(len(ps)) :
			writerF.writerow( [ps[i], v1[i], v2[i], v3[i]])

	#Exercise 1.3
	maxEd = maxiumOfEdges(v3, ps)
	print("El valor de p que da como resultado el mayor numero de arcos que conectan SCC es: {0}".format(maxEd))
	"""
	r1, r2 = inflectionPoint(ps, v2)
	print("El punto de inflexion se encuentra entre el rango [{0}, {1}] de p".format(r1, r2))

main()
