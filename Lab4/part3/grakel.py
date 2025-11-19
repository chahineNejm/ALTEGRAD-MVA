import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = r'C:/Users/ADMIN/Desktop/ALTEGRAD/ALTEGRAD-MVA/Lab4/datasets/train_5500_coarse.label'
path_to_test_set = r'C:/Users/ADMIN/Desktop/ALTEGRAD/ALTEGRAD-MVA/Lab4/datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx,doc in enumerate(docs):
        G = nx.Graph()
    
        ##################
        # your code here #
        ##################
        
        G.add_nodes_from(vocab)
        
        for i in range(len(doc)):
            term_i = doc[i]
            
            for j in range(i + 1, min(i + 1 + window_size, len(doc))):
                term_j = doc[j]
                G.add_edge(term_i, term_j)
                        
        graphs.append(G)
        
    
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
#nx.draw_networkx(G_train_nx[3], with_labels=True)
#plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sys


# Task 12

# Transform networkx graphs to grakel representations
G_train = graph_from_networkx(G_train_nx)
G_test = graph_from_networkx(G_test_nx)

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman()

# Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.fit_transform(G_test)

#Task 13

# Train an SVM classifier and make predictions

##################
# your code here #
##################

clf = SVC( kernel= "precomputed" )
clf.fit (K_train , y_train )
# P r e d i c y_train
y_pred = clf . predict ( K_train )

# Evaluate the predictions
print("Accuracy:", accuracy_score(y_pred, y_test))


#Task 14


##################
# your code here #
##################
from grakel.kernels import RandomWalk
import pandas as pd

# --- Define Kernels ---
kernels = [
    {
        "name": ">>> WL Subtree (n_iter=1) <<<",
        "kernel": WeisfeilerLehman(n_iter=1, normalize=False, base_kernel=VertexHistogram)
    },
    {
        "name": ">>> WL Subtree (n_iter=5) <<<",
        "kernel": WeisfeilerLehman(n_iter=5, normalize=True, base_kernel=VertexHistogram)
    },
    {
        "name": ">>> Random Walk (p=0.01) <<<",
        "kernel": RandomWalk(kernel_type='geometric', p=0.01, normalize=True)
    },
    {
        "name": ">>> Vertex Histogram (Baseline) <<<",
        "kernel": VertexHistogram(normalize=True)
    }
]

results = []

print("=" * 60)
print("             STARTING GRAPH KERNEL EXPERIMENT")
print("=" * 60)

for item in kernels:
    kernel_name = item["name"]
    gk = item["kernel"]
    
    print(f"\n{kernel_name}")
    print("-" * len(kernel_name))
    

    # Construct Kernel Matrices
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)
    
    # Train SVM and Predict
    svm_classifier = SVC(kernel='precomputed')
    svm_classifier.fit(K_train, y_train)
    y_pred = svm_classifier.predict(K_test)
    
    # Evaluate Performance
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        "Kernel Name": kernel_name.strip('> <'),
        "Accuracy": accuracy,
        "K_train Shape": K_train.shape
    })
        
    print(f"  ACCURACY: {accuracy:.4f}")
    print(f"  K_train Shape: {K_train.shape}")



# --- Performance Matrix ---
performance_df = pd.DataFrame(results)
performance_df = performance_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
performance_df.index.name = "Rank"

print("\n" + "=" * 60)
print("             FINAL PERFORMANCE MATRIX")
print("=" * 60)

# Display the performance table using markdown format
print(performance_df[['Kernel Name', 'Accuracy', 'K_train Shape']].to_markdown(index=True))
print("=" * 60)