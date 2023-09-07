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
HFNLP Scan Connection Matrix

"""

import numpy as np
import torch as pt
import csv
from torch_geometric.data import Data

from HFNLPpy_ScanGlobalDefs import *
from ANNtf2_loadDataset import datasetFolderRelative

def updateOrAddConnectionToGraph(neuronNamelist, HFconnectionGraph, sourceNeuronID, targetNeuronID):
	#contextConnection=False
	if(edgeExists(HFconnectionGraph.edge_index, sourceNeuronID, targetNeuronID)):
		edge_index = getEdgeIndex(HFconnectionGraph.edge_index, sourceNeuronID, targetNeuronID)
		HFconnectionGraph.edge_attr[edge_index] += HFconnectionWeightObs
	else:
		# Define the new edge to add as a tuple (neuronAid, neuronBid, connectionWeight)
		new_edge = (sourceNeuronID, targetNeuronID, HFconnectionWeightObs)
		# Append the new edge to the edge_index and edge_attr attributes of the Data object
		edge_indexAdd = pt.tensor([[new_edge[0]], [new_edge[1]]], dtype=pt.long)
		edge_attrAdd = pt.tensor([new_edge[2]], dtype=pt.float)
		if(graphExists(HFconnectionGraph)):
			HFconnectionGraph.edge_index = pt.cat([HFconnectionGraph.edge_index, edge_indexAdd], dim=1)
			HFconnectionGraph.edge_attr = pt.cat([HFconnectionGraph.edge_attr, edge_attrAdd])
		else:
			HFconnectionGraph.edge_index = edge_indexAdd
			HFconnectionGraph.edge_attr = edge_attrAdd

def readHFconnectionMatrix():
	if(HFreadSavedConnectionsMatrix):
		HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName
		HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName
		neuronNamelist = readConceptNeuronList(HFconceptNeuronListPathName)
		HFconnectionGraph = readGraphFromCsv(HFconnectionMatrixPathName)
	else:
		neuronNamelist = []
		HFconnectionGraph = Data(edge_index=None, edge_attr=None)
	return neuronNamelist, HFconnectionGraph

def writeHFconnectionMatrix(neuronNamelist, HFconnectionGraph):
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName
	HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName
	writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
	writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)

def readGraphFromCsv(filePath):
	"""
	Reads a graph from a CSV file and returns a PyG Data object representing the graph.
	The CSV file should have three columns: source, target, weight.
	"""
	connections = []
	with open(filePath, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			source, target, weight = map(int, row)
			connections.append((source, target, weight))
	edge_index = pt.tensor([[c[0], c[1]] for c in connections], dtype=pt.long).t()
	edge_attr = pt.tensor([c[2] for c in connections], dtype=pt.float)
	HFconnectionGraph = generateGraphFromEdgeLists(edge_index, edge_attr)
	return HFconnectionGraph

def generateGraphFromEdgeLists(edge_index, edge_attr):
	HFconnectionGraph = Data(edge_index=edge_index, edge_attr=edge_attr)
	return HFconnectionGraph

def writeGraphToCsv(graph, filePath):
	"""
	Writes a graph represented by a PyG Data object to a CSV file.
	The CSV file will have three columns: source, target, weight.
	"""
	edge_index = graph.edge_index.t().tolist()
	edge_attr = graph.edge_attr.tolist()
	connections = [(edge_index[i][0], edge_index[i][1], edge_attr[i]) for i in range(len(edge_attr))]
	with open(filePath, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(connections)
		
def readConceptNeuronList(filePath):
	names = []
	try:
		with open(filePath, 'r') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if row:
					names.append(row[0])
	except FileNotFoundError:
		print("File not found.")
	return names

def writeConceptNeuronList(names, filePath):
	try:
		with open(filePath, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for name in names:
				writer.writerow([name])
		print("Names written to file successfully.")
	except Exception as e:
		print("Error:", e)
		
def edgeExists(graph, source, target):
	if(graphExists(graph)):
		# Find the indices of all edges with the given source node
		sourceEdges = (graph.edge_index[0] == source).nonzero(as_tuple=True)[0]
		# Check if any of these edges have the given target node
		result = (graph.edge_index[1][sourceEdges] == target).any()
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
	sourceEdges = (graph.edge_index[0] == source).nonzero(as_tuple=True)[0]
	# Check if any of these edges have the given target node
	targetEdges = (graph.edge_index[1][sourceEdges] == target).nonzero(as_tuple=True)[0]
	edge_index = None
	if len(targetEdges) > 0:
		edge_index = sourceEdges[targetEdges[0]].item()
	return edge_index
		
