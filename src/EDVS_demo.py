import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Load data
# load the gml file
G = nx.read_gml("celegansneural.gml")
# load the edgelist file
# G = nx.read_edgelist("sub_citesnet.edgelist", nodetype=int,data=(("weight", int),),create_using=nx.DiGraph)
# load the mtx file
# G = nx.read_edgelist("polblogs.mtx", nodetype=int,data=(("weight", int),),create_using=nx.DiGraph)

# print nx.info
print(nx.info(G,n=None))

# get the list of nodes
nodes = list(G.nodes)
# two cite networks need special treatment
# nodes = pointspsy['id'].tolist()

# get adjacency matrix
def getAdjacencyMatrix(network, nodes):
    matrix =  np.zeros([len(nodes), len(nodes)])
    
    for i in range(len(nodes)):
        neighs = list(network.predecessors(nodes[i]))
        for j in neighs:

            # if load edgelist file
            # matrix[i,nodes.index(j)] = network[nodes[i]][j]['weight']
            # if load blog mtx file
            # matrix[i,nodes.index(j)] = network[j][nodes[i]]['weight']
            # load neural network gml file
            matrix[i,nodes.index(j)] = int(network[j][nodes[i]][0]['value'])
    return matrix

# get connective matrix
def getConnectiveMatrix(network, nodes):
    matrix =  np.zeros([len(nodes), len(nodes)])
    
    for i in range(len(nodes)):
        neighs = list(network.predecessors(nodes[i]))
        for j in neighs:
            matrix[i,nodes.index(j)] = 1
    return matrix

adjacency_matrix = getAdjacencyMatrix(G, nodes)
connective_matrix = getConnectiveMatrix(G, nodes)

# calculate the EDVS diversity
EDVS_matrix = np.matmul(connective_matrix, adjacency_matrix)
neural_diversity = pd.DataFrame()
neural_diversity['node'] = nodes
diversity = []
for i in range(len(nodes)):
    diversity.append(entropy(EDVS_matrix[i,:]))
neural_diversity['EdVS'] = diversity

neural_diversity.to_csv('neural_diversity.csv', index=False)