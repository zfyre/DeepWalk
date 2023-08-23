import os
import networkx as nx
from scipy.io import mmread
from scanpy import read_mtx

DatasetName = "rand"

""" Change the dataset path heree!!"""
Data_Dir = os.path.join("Data",DatasetName + ".mtx")

# Open the txt file
with open(Data_Dir) as f:
    dim = f.readline()
    edges = f.readlines()

# Create a graph
graph = nx.Graph()

# Add edges to the graph
for edge in edges:
    edge = edge.strip("\n")
    u, v = (int(val) for val in edge.split())
    graph.add_edge(u, v)

# Saving The Graph
import pickle
pickle.dump(graph, open(os.path.join("Data",DatasetName + ".pickle"), 'wb'))



