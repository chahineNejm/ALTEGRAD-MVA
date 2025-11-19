"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
print("########### TASK 1 ###########")
file_path = r'C:\Users\ADMIN\Desktop\ALTEGRAD\ALTEGRAD-MVA\Lab4\datasets\CA-HepTh.txt' 

G = nx.read_edgelist(
    file_path,
    comments='#',
    delimiter='\t',
    create_using=nx.Graph()  # Specify an undirected graph
)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print("--- Network Characteristics ---")
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

############## Task 2
print("\n########### TASK 2 ###########")
num_cc = nx.number_connected_components(G)
print(f"Number of connected components: {num_cc}")

components = sorted(nx.connected_components(G), key=len, reverse=True)
giant_nodes = components[0]
G_giant = G.subgraph(giant_nodes).copy()

n_total = G.number_of_nodes()
e_total = G.number_of_edges()

n_giant = G_giant.number_of_nodes()
e_giant = G_giant.number_of_edges()

print("\n--- Giant Connected Component ---")
print(f"Nodes in GCC: {n_giant} ({n_giant / n_total:.2%} of all nodes)")
print(f"Edges in GCC: {e_giant} ({e_giant / e_total:.2%} of all edges)")



