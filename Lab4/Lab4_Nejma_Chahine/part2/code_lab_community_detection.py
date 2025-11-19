"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k=3):
    
    ##################
    # your code here #
    ##################
    Adjancecy = nx.to_numpy_array(G)
         
    n = Adjancecy.shape[0]
    
    degrees = Adjancecy.sum(axis=1)
    D_inv = np.diag(1 / degrees)
    L_rw = np.eye(n) - D_inv @ Adjancecy
    eigvals, eigvecs = np.linalg.eigh(L_rw)

    # sort eigenvalues and take first k eigenvectors
    idx = np.argsort(eigvals)
    U = eigvecs[:, idx[:k]]     # shape (m, k)

    # --- Step 4: K-means on rows of U ---
    kmeans = KMeans(n_clusters=k, n_init=10).fit(U)
    labels = kmeans.labels_

    # --- Return mapping: node -> cluster ---
    nodes = list(G.nodes())
    clustering = {nodes[i]: int(labels[i]) for i in range(n)}
    
    return clustering


############## Task 4

##################
# your code here #
##################

print("\n################ TASK 4 #############")
file_path = r'C:\Users\ADMIN\Desktop\ALTEGRAD\ALTEGRAD-MVA\Lab4\datasets\CA-HepTh.txt' 

G = nx.read_edgelist(
    file_path,
    comments='#',
    delimiter='\t',
    create_using=nx.Graph()  # Specify an undirected graph
)

components = sorted(nx.connected_components(G), key=len, reverse=True)
giant_nodes = components[0]
G_giant = G.subgraph(giant_nodes).copy()

clusters = spectral_clustering(G_giant, k=50)

from collections import Counter

counts = Counter(clusters.values())


print("number of nodes in each class:",counts)



############## Task 5
# Compute modularity value from graph G based on clustering
print("\n################ TASK 5 #############")

def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    m = G.number_of_edges()
    # Build community → list of nodes
    comms = {}
    for node, c in clustering.items():
        comms.setdefault(c, []).append(node)

    Q = 0.0

    for c, nodes in comms.items():
        # lc = edges inside the community
        lc = 0
        for u in nodes:
            for v in G.neighbors(u):
                if v in nodes:
                    lc += 1
        lc = lc / 2     # each internal edge counted twice

        # dc = sum of degrees in the community
        dc = sum(G.degree(n) for n in nodes)

        Q += (lc / m) - (dc / (2 * m))**2
    
    
    
    
    return Q



############## Task 6

##################
# your code here #
##################
print("\n################ TASK 6k #############")
Q_spec = modularity(G_giant, clusters)
print("Modularity (Spectral, k=50):", Q_spec)


nodes = list(G_giant.nodes())
clusters_rand = {node: np.random.randint(0, 50) for node in nodes}

Q_rand = modularity(G_giant, clusters_rand)
print("Modularity (Random, k=50):", Q_rand)


