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
import torch.nn.functional as F
import csv
from torch_geometric.data import Data

from HFNLPpy_ScanGlobalDefs import *
from ANNtf2_loadDataset import datasetFolderRelative

if(useHFconnectionMatrixBasicBool):
	HFconnectionsMatrixType = pt.bool
else:
	HFconnectionsMatrixType = pt.long
	
def addContextConnectionsToGraph(HFconnectionGraph, neuronID, contextConnectionVector):
	conceptsSize = contextConnectionVector.shape[0]
	spareConceptsSize = HFconnectionMatrixBasicMaxConcepts-conceptsSize
	contextConnectionVectorPadded = F.pad(contextConnectionVector, (0, spareConceptsSize), mode='constant', value=0)
	#contextConnectionVectorPadded = pt.nn.ZeroPad1d(spareConceptsSize)(contextConnectionVector)	#requires later version of pytorch
	if(useHFconnectionMatrixBasicBool):
		pt.logical_and(HFconnectionGraph[neuronID], contextConnectionVectorPadded)
	else:
		HFconnectionGraph[neuronID] += contextConnectionVectorPadded

def readHFconnectionMatrix():
	if(HFreadSavedConnectionsMatrixBasic):
		HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName
		HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName
		neuronNamelist = readConceptNeuronList(HFconceptNeuronListPathName)
		HFconnectionGraph = readGraphFromCsv(HFconnectionMatrixPathName)
		conceptsSize = HFconnectionGraph.shape[0]
		spareConceptsSize = HFconnectionMatrixBasicMaxConcepts-conceptsSize
		print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
		HFconnectionGraph = pt.nn.ZeroPad2d((0, spareConceptsSize, 0, spareConceptsSize))(HFconnectionGraph)
		print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	else:
		neuronNamelist = []
		HFconnectionGraph = pt.zeros([HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts], dtype=HFconnectionsMatrixType)
	return neuronNamelist, HFconnectionGraph

def writeHFconnectionMatrix(neuronNamelist, HFconnectionGraph):
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName
	HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName
	writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
	writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)

def readGraphFromCsv(filePath):
	connections = []
	with open(filePath, 'r') as f:
		reader = csv.reader(f)
		for row in (reader):
			connections.append(row)
	HFconnectionGraph = np.array(connections, dtype=HFconnectionsMatrixType)
	return HFconnectionGraph

def writeGraphToCsv(graph, filePath):
	connections = graph.numpy()
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
		
