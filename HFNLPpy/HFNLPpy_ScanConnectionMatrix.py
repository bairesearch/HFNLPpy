"""HFNLPpy_ScanConnectionMatrix.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Connection Matrix

"""

import numpy as np
import torch as pt
import csv
from torch_geometric.data import Data

from HFNLPpy_ScanGlobalDefs import *
from ANNtf2_loadDataset import datasetFolderRelative
	
def readHFconnectionMatrix():
	if(HFreadSavedConnectionsMatrix):
		HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName
		HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName

		neuronNamelist = readConceptNeuronList(HFconceptNeuronListPathName)
		HFconnectionMatrix = readGraphFromCsv(HFconnectionMatrixPathName)
	else:
		neuronNamelist = []
		HFconnectionMatrix = Data(edge_index=None, edge_attr=None)
		
	return neuronNamelist, HFconnectionMatrix

def writeHFconnectionMatrix(neuronNamelist, HFconnectionMatrix):
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName
	HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName

	writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
	writeGraphToCsv(HFconnectionMatrix, HFconnectionMatrixPathName)

def readGraphFromCsv(file_path):
	"""
	Reads a graph from a CSV file and returns a PyG Data object representing the graph.
	The CSV file should have three columns: source, target, weight.
	"""
	connections = []
	with open(file_path, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			source, target, weight = map(int, row)
			connections.append((source, target, weight))
	edge_index = pt.tensor([[c[0], c[1]] for c in connections], dtype=pt.long).t()
	edge_attr = pt.tensor([c[2] for c in connections], dtype=pt.float)
	graph = generateGraphFromEdgeLists(edge_index, edge_attr)
	return graph

def generateGraphFromEdgeLists(edge_index, edge_attr):
	graph = Data(edge_index=edge_index, edge_attr=edge_attr)
	return graph

def writeGraphToCsv(graph, file_path):
	"""
	Writes a graph represented by a PyG Data object to a CSV file.
	The CSV file will have three columns: source, target, weight.
	"""
	edge_index = graph.edge_index.t().tolist()
	edge_attr = graph.edge_attr.tolist()
	connections = [(edge_index[i][0], edge_index[i][1], edge_attr[i]) for i in range(len(edge_attr))]
	with open(file_path, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(connections)
		
def readConceptNeuronList(file_path):
	names = []
	try:
		with open(file_path, 'r') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if row:
					names.append(row[0])
	except FileNotFoundError:
		print("File not found.")
	return names

def writeConceptNeuronList(names, file_path):
	try:
		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for name in names:
				writer.writerow([name])
		print("Names written to file successfully.")
	except Exception as e:
		print("Error:", e)

def updateOrAddConnectionToGraph(neuronNamelist, graph, sourceNeuronID, targetNeuronID):
	if(edgeExists(graph.edge_index, sourceNeuronID, targetNeuronID)):
		edgeIndex = getEdgeIndex(graph.edge_index, sourceNeuronID, targetNeuronID)
		graph.edge_attr[edgeIndex] += HFconnectionWeightObs
	else:
		# Define the new edge to add as a tuple (neuronAid, neuronBid, connectionWeight)
		new_edge = (sourceNeuronID, targetNeuronID, HFconnectionWeightObs)
		# Append the new edge to the edge_index and edge_attr attributes of the Data object
		edge_indexAdd = pt.tensor([[new_edge[0]], [new_edge[1]]], dtype=pt.long)
		edge_attrAdd = pt.tensor([new_edge[2]], dtype=pt.float)
		if(graphExists(graph)):
			graph.edge_index = pt.cat([graph.edge_index, edge_indexAdd], dim=1)
			graph.edge_attr = pt.cat([graph.edge_attr, edge_attrAdd])
		else:
			graph.edge_index = edge_indexAdd
			graph.edge_attr = edge_attrAdd
		
def edgeExists(graph, source, target):
	if(graphExists(graph)):
		# Find the indices of all edges with the given source node
		source_edges = (graph.edge_index[0] == source).nonzero(as_tuple=True)[0]
		# Check if any of these edges have the given target node
		result = (graph.edge_index[1][source_edges] == target).any()
	else:
		result = False
	return result

def graphExists(graph):
	if(hasattr(graph, 'edge_index')):
		if(graph.edge_index is not None):
			result = True
		else:
			result = False
	else:
		result = False
	return result
	
def getEdgeIndex(graph, source, target):
	# Find the indices of all edges with the given source node
	source_edges = (graph.edge_index[0] == source).nonzero(as_tuple=True)[0]
	# Check if any of these edges have the given target node
	target_edges = (graph.edge_index[1][source_edges] == target).nonzero(as_tuple=True)[0]
	edgeIndex = None
	if len(target_edges) > 0:
		edgeIndex = source_edges[target_edges[0]].item()
	return edgeIndex
		
