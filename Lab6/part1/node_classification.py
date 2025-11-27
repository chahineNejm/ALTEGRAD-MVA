"""
Deep Learning on Graphs - ALTEGRAD - Nov 2025
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist(r'C:\Users\ADMIN\Desktop\ALTEGRAD\ALTEGRAD-MVA\Lab6\data\karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt(r'C:\Users\ADMIN\Desktop\ALTEGRAD\ALTEGRAD-MVA\Lab6\data\karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
# your code here #
##################

pos = nx.spring_layout(G, seed=42)
color_map = ['red' if label == 0 else 'blue' for label in y]
nx.draw_networkx(
    G,
    pos,
    node_color=color_map,
    with_labels=True,

)
# plt.title("Karate Club")
# plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 40
walk_length = 20
model = deepwalk(G=G,num_walks=n_walks,walk_length=walk_length,n_dim=n_dim)

embeddings = np.zeros((n, n_dim))

for i, node in enumerate(G.nodes()):
    
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
# your code here #
##################
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", acc)

############## Task 8
# Generates spectral embeddings

##################
# your code here #
##################
# --- Spectral Embeddings (Task 8) ---


degrees = np.array([G.degree(i) for i in G.nodes()])
D_inv = diags(1.0 / degrees)
A = nx.to_scipy_sparse_array(G, dtype=float)
L_rw = eye(n) - D_inv @ A

vals, vecs = eigs(L_rw, k=2, which='SM')
spectral_embeddings = vecs.real  

X_train_spectral = spectral_embeddings[idx_train]
X_test_spectral  = spectral_embeddings[idx_test]

y_train_spectral = y_train
y_test_spectral  = y_test

# Train logistic regression
clf_spec = LogisticRegression(max_iter=500)
clf_spec.fit(X_train_spectral, y_train_spectral)

y_pred_spec = clf_spec.predict(X_test_spectral)
acc_spec = accuracy_score(y_test_spectral, y_pred_spec)

print("Spectral accuracy:", acc_spec)
print("DeepWalk accuracy:", acc)

