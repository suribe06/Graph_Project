import random
import matplotlib.pyplot as plt

random.seed(1107)

def random_erdos(n,p):
    E = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random()<p:
                E.append((i,j))
    G = [[] for i in range(n)]
    for i,j in E:
        G[i].append(j)
        G[j].append(i)
    return G

#print(random_erdos(10,0.5))


hists = []
for ronda in range(100):
    G = random_erdos(20, 0.3)

    n = len(G)
    degs = [len(G[i]) for i in range(n)]
    hist = [0]*n
    for d in degs:
        hist[d]+=1
    hists.append(hist)

hist = [0]*n
for h in hists:
    for i in range(n):
        hist[i]+=h[i]

for i in range(n):
    hist[i]/=len(hists)

print(hist)

plt.stem(hist)
plt.title('Degree distribution')
plt.xlabel('degree')
plt.ylabel('# nodes with a given degree')

#plt.plot([1200*k**-3 if k>=3 else 0 for k in range(n)], color='orange')

plt.show()
