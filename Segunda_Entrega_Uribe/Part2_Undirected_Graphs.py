"""
Proyecto AGRA 2019-1 Santiago Uribe P. Cod: 8925546
"""
import random
import matplotlib.pyplot as plt
import networkx as nx
import csv
import numpy

#Tarjan Algorithm for Bridges and Articulation Points
def dfs(u, G):
    global visited, depth, low, time, parent, ap, bridges
    visited[u] = 1
    depth[u] = low[u] = time
    time += 1
    children = 0
    for v in G[u]:
        if visited[v] == 0:
            parent[v] = u
            children += 1
            dfs(v, G)
            low[u] = min(low[u], low[v])
            if low[v] > depth[u]:
                bridges.append((u, v))
            if (children > 1 and parent[u] == -1) or (parent[u] != -1 and low[v] >= depth[u]):
                ap.append(u)
        elif v != parent[u]:
            low[u] = min(low[u], depth[v])

def tarjan(G):
    global visited, depth, low, time, parent, ap, bridges
    n = len(G)
    visited = [0 for _ in range(n)]
    depth = [None for _ in range(n)]
    low = [None for _ in range(n)]
    parent = [-1 for _ in range(n)]
    time, cc = 0, 0
    ap, bridges = [], []

    for i in range(n):
        if visited[i] == 0:
            dfs(i, G)
            cc += 1

    return ap, bridges, cc

#Semilla
random.seed(23)

def random_graph(n, p):
    Edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                Edges.append((i, j))

    G = nx.Graph()
    for i in range(n): G.add_node(i)
    G.add_edges_from(Edges)

    G2 = [[] for i in range(n)]
    for i, j in Edges:
        G2[i].append(j)
        G2[j].append(i)

    return G, G2

def graph_stats(G):
    n = len(G)
    number_bcc, average_bcc, arti_points, bridges, average_degree, triplets, triangles = 0, 0, 0, 0, 0, 0, 0
    ap, bri, cc = tarjan(G)

    number_cc = cc
    average_cc = n / cc
    arti_points = len(ap)
    bridges = len(bri)
    cont_deg = 0
    for i in range(n):
        cont_deg += len(G[i])
    average_degree = cont_deg / n

    return number_cc, average_cc, arti_points, bridges, average_degree

def triangles_triplets(G):
    """
    Esta funcion es la encargada de calcular las tripletas abiertas y los triangulos que a su vez son tripletas cerradas de un grafo
    Nota: La funcion no cuenta las permutaciones de dichas metricas
    """
    n = len(G)
    triangles, triplets = 0, 0
    for u in range(n):
        for v in G[u]:
            for w in G[v]:
                if u in G[w] and u != v and u != w and v != w:
                    triangles += 1
                if u != v and v != w and w != u and w not in G[u]:
                    triplets += 1
    triangles = triangles / 6
    triplets = triplets / 2 #calculo de tripletas abiertas

    triplets = triplets + triangles #tripletas abiertas + tripletas cerradas(triangulos)

    return triangles, triplets

def generatorTT(n, cant):
    """
    Esta funcion es especial para los triangulos y tripletas debido a que estas 2 metricas se deben realizar
    con los valores de p_small U p_large
    """
    p = [i for i in numpy.arange(0, 0.1, 0.005)] #p_small
    for i in numpy.arange(0.1, 1.05, 0.05): p.append(i)#p_small U p_large
    tria, trip = [], []

    for i in p:
        cont1, cont2 = 0, 0
        for j in range(200):
            G, G2 = random_graph(n, i)
            v1, v2 = triangles_triplets(G2)
            cont1 += v1
            cont2 += v2
        tria.append(cont1/cant)
        trip.append(cont2/cant)
    return tria, trip, p

def generator(n, cant):
    met1, met2, met3, met4, met5 = [], [], [], [], []
    p_small = [i for i in numpy.arange(0, 0.1, 0.005)]

    for i in p_small:
        cont1, cont2, cont3, cont4, cont5 = 0, 0, 0 ,0 , 0
        for j in range(cant):
            G, G2 = random_graph(n, i)
            v1, v2, v3, v4, v5 = graph_stats(G2)
            cont1 += v1
            cont2 += v2
            cont3 += v3
            cont4 += v4
            cont5 += v5
        met1.append(cont1/cant)
        met2.append(cont2/cant)
        met3.append(cont3/cant)
        met4.append(cont4/cant)
        met5.append(cont5/cant)

    return met1, met2, met3, met4, met5, p_small

def graficar(m1, m2, m3, m4, m5, p):
    plt.title("Graph Stats")
    plt.xlabel("p")
    plt.ylabel("f(100, p)")
    plt.plot(p, m1, label="Number Of CC's")
    plt.plot(p, m2, label="Average Of CC's")
    plt.plot(p, m3, label="Articulation Points")
    plt.plot(p, m4, label="Bridges")
    plt.plot(p, m5, label="Degree")
    plt.legend()
    plt.show()

def graficarTT(m1, m2, p):
    """
    Funcion encargada de graficar unicamente los tirangulos, tripletas y el cociente entre dichas metricas
    """
    quotient = []
    for i in range(len(m1)):
        if m2[i] != 0:
            q = m1[i] / m2[i]
            quotient.append(q)
        else:
            quotient.append(0)

    plt.title("Triangles and Triplets")
    plt.plot(p, m1, label="Triangles")
    plt.plot(p, m2, label="Triplets")
    plt.plot(p, quotient, label="#Triangles / #Triplets")
    plt.legend()
    plt.show()

    """
    La relacion que tiene el p con el cociente es que a medida que el valor de p se va a acercando a 1
    el cociente de igual manera va creciendo, es decir, son directamente proporcionales.
    """

#Functions Exercise 2.3 (Maxium AP)
def func_aux_ap(n, p):
    """
    Esta funcion auxiliar se utilizara para obtener unicamente el valor promedio de los AP de los grafos
    adicionales que se generan en la funcion de maxium_ap
    """
    cont = 0
    for i in range(1000):
        G, G2 = random_graph(n, p)
        v1, v2, v3, v4, v5 = graph_stats(G2)#v3 es la metrica de AP del grafo
        cont += v3
    ans =  cont / 1000
    return ans

def divideConquer(lo, hi):
    """
    En esta funcion se generan mas grafos aleatorios para asi suavizar mas la grafica y asi el proceso de
    encontrar el valor maximo sea mas exacto. Dependiendo de los valores que se obtienen en los valores
    de los nuevos grafos generados se ajustan los valor de lo y hi, hasta encontrar el maximo
    """
    global largest
    iteraciones = 0
    while iteraciones < 100:
        nlo = lo
        nhi = hi
        mid = (nlo + nhi) / 2
        ps = [lo, mid-0.001, mid, mid+0.001, hi]
        newY = []
        for i in ps:
            r = func_aux_ap(100, i)
            newY.append(r)
        m = max(newY)
        ind = newY.index(m)
        if m > largest:
            largest = m
            if ind == 0 or ind == len(ps)-1:
                nlo = ps[0]
                nhi = ps[len(ps) -1]
            else:
                nlo = ps[ind - 1]
                nhi = ps[ind + 1]
        else:
            return ps[ind]

        iteraciones += 1

def maxium_ap(AP, p_small):
    """
    En esta funcion se realiza un primer acercamiento al maximo con los primeros 200 grafos obtenidos.
    Despues se procede a llamar a la funcion que realizara el proceso de suavizar la grafica para
    posteriormente encontrar mas acertadamente el valor maximo
    """
    global largest
    largest = max(AP)
    h = AP.index(largest)
    lo = p_small[h-1]
    hi = p_small[h+1]
    ans = divideConquer(lo, hi)
    return ans

#Functions Exercise 2.3 (Maxium Bridges)
def func_aux_b(n, p):
    """
    Esta funcion auxiliar se utilizara para obtener unicamente el valor promedio de los Bridges de los grafos
    adicionales que se generan en la funcion de maxium_b
    """
    cont = 0
    for i in range(1000):
        G, G2 = random_graph(n, p)
        v1, v2, v3, v4, v5 = graph_stats(G2) #v4 es la metrica de bridges del grafo
        cont += v4
    ans =  cont / 1000
    return ans

def divideConquer2(lo, hi):
    """
    En esta funcion se generan mas grafos aleatorios para asi suavizar mas la grafica y asi el proceso de
    encontrar el valor maximo sea mas exacto. Dependiendo de los valores que se obtienen en los valores de
    las metricas de los nuevos grafos generados se ajustan los valor de lo y hi, hasta encontrar el maximo
    """
    global largest
    iteraciones = 0
    while iteraciones < 100:
        nlo = lo
        nhi = hi
        mid = (nlo + nhi) / 2
        ps = [lo, mid-0.001, mid, mid+0.001, hi]
        newY = []
        for i in ps:
            r = func_aux_b(100, i)
            newY.append(r)
        m = max(newY)
        ind = newY.index(m)
        if m > largest:
            largest = m
            if ind == 0 or ind == len(ps)-1:
                nlo = ps[0]
                nhi = ps[len(ps) -1]
            else:
                nlo = ps[ind - 1]
                nhi = ps[ind + 1]
        else:
            return ps[ind]

        iteraciones += 1

def maxium_b(B, p_small):
    """
    En esta funcion se realiza un primer acercamiento al maximo con los primeros 200 grafos obtenidos.
    Despues se procede a llamar a la funcion que realizara el proceso de suavizar la grafica para
    posteriormente encontrar mas acertadamente el valor maximo
    """
    global largest
    largest = max(B)
    h = B.index(largest)
    lo = p_small[h-1]
    hi = p_small[h+1]
    ans = divideConquer2(lo, hi)
    return ans

#Functions Exercise 2.3 (Inflection Points)
def derivative(x, y):
    """
    Esta funcion es la encargada de "derivar". Ya que en este funcion no se tiene una funcion concreta
    para derivar este proceso se lleva a cabo sacando las pendientes de las los resultados de las metricas
    """
    result = []
    for i in range(3, len(y)):
        x1, y1 = x[i-1], y[i-1]
        x2, y2 = x[i], y[i]
        d = (y2 - y1) / (x2 - x1)
        result.append(d)

    return result

def inflectionPoint(p_small, array):
    """
    Esta funcion es la encargada de ya con las segundas derivadas encontrar el rango de valores de p,
    en los cuales la segunda derivada cambia de signo, lo que indica que habria un cambio de concavidad
    en la funcion
    """
    firstDeri = derivative(p_small, array)
    secondDeri = derivative(p_small, firstDeri)
    ilo, ihi = 0, 0
    i = 1
    flag = True
    while i < len(secondDeri) and flag == True:
        ant = secondDeri[i - 1]
        sig = secondDeri[i]
        if (ant > 0 and sig < 0) or (ant < 0 and sig > 0):
            ilo = i-1
            ihi = i
            flag = False
        i += 1
    lo = p_small[ilo+3]
    hi = p_small[ihi+3]

    return lo, hi

def main():
    #Exercise 2.1
    graph, graph2 = random_graph(15, 0.2)
    nx.draw_circular(graph, with_labels=True)
    plt.show()
    #plt.savefig("Undirected_Graph.png", format="PNG")
    m1, m2, m3, m4, m5 = graph_stats(graph2)
    m7, m6 = triangles_triplets(graph2)
    print("Number of CC's: {0}, Average of CC's: {1}, Articulation Points: {2}, Bridges: {3}, Degree: {4}, Triplets: {5}, Triangles: {6}".format(m1, m2, m3, m4, m5, m6, m7))

    #Exercise 2.2.1
    lm1, lm2, lm3, lm4, lm5, p = generator(100, 200)
    graficar(lm1, lm2, lm3, lm4, lm5, p)

    #Guardado de datos en csv
    with open('data2_1.csv', 'w', newline = '') as f:
        writerF = csv.writer(f)
        writerF.writerow(['p_small value ', 'Number of CC', ' Mean average size of CC', ' Mean of Articulation Points', ' Mean of Bridges', 'Mean of node Degree' ])
        for i in range(len(p)) :
            writerF.writerow([p[i], lm1[i], lm2[i], lm3[i], lm4[i], lm5[i]])

    #Exercise 2.2.2 (triplets and triangles)
    ltria, ltrip, p2 = generatorTT(20, 200)
    graficarTT(ltria, ltrip, p2)

    #Guardado de triangulos y tripletas en csv
    with open('data2_2.csv', 'w', newline = '') as f2:
        writerF = csv.writer(f2)
        writerF.writerow(['p value ', ' Number Triangles', ' Number Triplets'])
        for i in range(len(p2)) :
            writerF.writerow([p2[i], ltria[i], ltrip[i]])

    #Exercise 2.3.1 (maxium articulation points)
    m_ap = maxium_ap(lm3, p)
    print("El valor de p que da como resultado el mayor numero de puntos de articulacion es: {0}".format(m_ap))

    #Exercise 2.3.2 (maxium bridges)
    m_b = maxium_b(lm4, p)
    print("El valor de p que da como resultado el mayor numero de puentes es: {0}".format(m_b))

    #Exercise 2.3.3 (Inflection Point)
    r1, r2 = inflectionPoint(p, lm2)
    print("El punto de inflexion se encuentra entre el rango [{0}, {1}] de p".format(r1, r2))

main()
