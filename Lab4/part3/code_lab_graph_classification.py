"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():

    ##################
    # your code here #
    ##################
    dataset = TUDataset(root="data/MUTAG", name="MUTAG")

    Gs = []
    for data in dataset:
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        Gs.append(G)
    y = [data.y.item() for data in dataset]
    return Gs, y


Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(Gs_train), 4))
    
    ##################
    # your code here #
    ##################
    
    for i, G in enumerate(Gs_train):
        nodes = list(G.nodes())
        n = len(nodes)
        if n < 3: 
            continue

        for _ in range(n_samples):
            sampled_nodes = np.random.choice(nodes, size=3, replace=False)
            subgraph = G.subgraph(sampled_nodes)
            
            for j in range(4):
                if nx.is_isomorphic(subgraph, graphlets[j]):
                    phi_train[i, j] += 1
                    break

    phi_test = np.zeros((len(Gs_test), 4))
    
    ##################
    # your code here #
    ##################
    for i, G in enumerate(Gs_test):
        nodes = list(G.nodes())
        n = len(nodes)
        if n < 3: 
            continue

        for _ in range(n_samples):
        
            sampled_nodes = np.random.choice(nodes, size=3, replace=False)
            subgraph = G.subgraph(sampled_nodes)
            
            for j in range(4):
                if nx.is_isomorphic(subgraph, graphlets[j]):
                    
                    phi_test[i, j] += 1
                    break

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)



############## Task 9

##################
# your code here #
##################

print("######################## TASK 9 ##################")
Kernell_train , Kernell_test = graphlet_kernel(G_train, G_test)
print(f"we retrive the sampled following kernel {Kernell_train}")


############## Task 10

##################
# your code here #
##################

from sklearn . svm import SVC
# I n i t i a l i z e SVM and t r a i n
clf = SVC( kernel= "precomputed" )
clf.fit (Kernell_train , y_train )
# P r e d i c y_train
y_pred_graphlet = clf . predict ( Kernell_test )
accuracy_graphlet = accuracy_score(y_test, y_pred_graphlet)

kernel_train_shortest,kernel_test_shortest = shortest_path_kernel(G_train,G_test)
clf = SVC( kernel= "precomputed" )
clf.fit (kernel_train_shortest , y_train )
# P r e d i c y_train
y_pred_shortest = clf . predict ( kernel_test_shortest )
accuracy_shortest = accuracy_score(y_test, y_pred_shortest)


#
print("--- Classification Accuracy Comparison ---")
print(f"Shortest Path Kernel Accuracy: {accuracy_shortest:.4f}")
print(f"Graphlet Kernel Accuracy:      {accuracy_graphlet:.4f}")


