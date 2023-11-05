"""HFNLPpy_ConnectionMatrixBasic.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Connection Matrix Basic

"""

import numpy as np
import torch as pt
import torch.nn.functional as F
import csv
from torch_geometric.data import Data

from HFNLPpy_globalDefs import *
from ANNtf2_loadDataset import datasetFolderRelative

epsilon = 1e-8  # Small epsilon value

if(pt.cuda.is_available()):
	device = pt.device("cuda")
else:
	device = pt.device("cpu")

def addContextConnectionsToGraphBasic(HFconnectionGraph, contextConnectionVector):
	if(useHFconnectionMatrixBasicBool):
		HFconnectionGraph = pt.logical_and(HFconnectionGraph, contextConnectionVector)
	else:
		HFconnectionGraph += contextConnectionVector
	return HFconnectionGraph

def padContextConnectionVector(contextConnectionVector):
	conceptsSize = contextConnectionVector.shape[0]
	spareConceptsSize = HFconnectionMatrixBasicMaxConcepts-conceptsSize
	contextConnectionVectorPadded = F.pad(contextConnectionVector, (0, spareConceptsSize), mode='constant', value=0)
	#contextConnectionVectorPadded = pt.nn.ZeroPad1d(spareConceptsSize)(contextConnectionVector)	#requires later version of pytorch
	return contextConnectionVectorPadded

def extendConceptNeuronContextVector(conceptNeuronContextVector):
	conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVector, dim=0)
	conceptNeuronContextVectorExtended = conceptNeuronContextVectorExtended.repeat(HFconnectionMatrixBasicMaxConcepts, 1)	#len(HFconnectionGraphObject.neuronNamelist)	
	return conceptNeuronContextVectorExtended
	
def createDiagonalMatrix(squareMatrix, width):
	diagonalMatrix = pt.tril(squareMatrix, diagonal=0) - pt.tril(squareMatrix, diagonal=-width)
	#diagonalMatrix = pt.tril(squareMatrix, diagonal=0) * torch.triu(squareMatrix, diagonal=width)
	return diagonalMatrix

def initialiseNeuronNameList(HFconnectionGraphObject, readSavedConceptList):
	if(readSavedConceptList):
		HFconceptNeuronListPathName = generateConceptListFileName()
		HFconnectionGraphObject.neuronNamelist = readConceptNeuronList(HFconceptNeuronListPathName)
	else:
		HFconnectionGraphObject.neuronNamelist = []
	createNeuronIDdictFromNameList(HFconnectionGraphObject)
	
def initialiseHFconnectionMatrixBasic(dendriticBranchIndex="", contextSizeIndex=""):
	if(HFreadSavedConnectionsMatrixBasic):
		HFconnectionGraph = readHFconnectionMatrixBasic(dendriticBranchIndex, contextSizeIndex)
	else:
		HFconnectionGraph = pt.zeros([HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts], dtype=HFconnectionsMatrixBasicType)
	if(HFconnectionMatrixBasicGPU):
		HFconnectionGraph = HFconnectionGraph.to(device)
	return HFconnectionGraph

def readHFconnectionMatrixBasic():
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixBasicFileName + HFconnectionMatriBasicExtensionName
	HFconnectionGraph = readGraphFromCsv(HFconnectionMatrixPathName)
	conceptsSize = HFconnectionGraph.shape[0]
	spareConceptsSize = HFconnectionMatrixBasicMaxConcepts-conceptsSize
	print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	HFconnectionGraph = pt.nn.ZeroPad2d((0, spareConceptsSize, 0, spareConceptsSize))(HFconnectionGraph)
	print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	return HFconnectionGraph
		
def createContextVectorTensorBasic():
	contextVectorLength = HFconnectionMatrixBasicMaxConcepts
	contextConnectionVector = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixBasicType)
	return contextConnectionVector
	
def writeHFconnectionMatrixBasic(HFconnectionGraph, neuronNamelist):
	HFconnectionMatrixPathName = generateHFconnectionMatrixBasicFileName()
	writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)

def generateHFconnectionMatrixBasicFileName():
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixBasicFileName + HFconnectionMatrixBasicExtensionName
	return HFconnectionMatrixPathName

def writeHFConceptListBasic(neuronNamelist):
	HFconceptNeuronListPathName = generateConceptListFileName()
	writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
	
def generateConceptListFileName():
	HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsBasicFileName + HFconceptNeuronsBasicExtensionName
	return HFconceptNeuronListPathName
	
def readGraphFromCsv(filePath):
	connections = []
	with open(filePath, 'r') as f:
		reader = csv.reader(f)
		for row in (reader):
			connections.append(row)
	connections = [[int(value) for value in row] for row in connections]
	graph = pt.tensor(connections, dtype=HFconnectionsMatrixAlgorithmType)
	return graph

def writeGraphToCsv(graph, filePath):
	graph = graph.cpu()		
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

		
def writeHFconnectionMatrixBasicWrapper(HFconnectionGraphObject):
	if(HFwriteSavedConnectionsMatrixBasic):
		if(linkSimilarConceptNodesBagOfWords):
			writeHFconnectionMatrixBasic(HFconnectionGraphObject.HFconnectionGraphBasic)
	if(HFwriteSavedConceptListBasic):
		writeHFConceptListBasic(HFconnectionGraphObject.neuronNamelist)

def normaliseBatchedTensor(HFconnectionGraph):
	HFconnectionGraph = HFconnectionGraph.float()
	if(useHFconnectionMatrixBasicBool):	#OLD: if(not weightStore)
		HFconnectionGraphNormalised = HFconnectionGraph
	else:
		#calculate a temporary normalised version of the HFconnectionGraph	#CHECKTHIS
		if(HFconnectionMatrixBasicNormaliseSoftmax):
			HFconnectionGraphNormalised = pt.nn.functional.softmax(HFconnectionGraph, dim=1)
		else:
			min_vals, _ = pt.min(HFconnectionGraph, dim=-1, keepdim=True)
			max_vals, _ = pt.max(HFconnectionGraph, dim=-1, keepdim=True)
			HFconnectionGraphNormalised = (HFconnectionGraph - min_vals) / (max_vals - min_vals + epsilon)
	return HFconnectionGraphNormalised
	
def createNeuronIDdictFromNameList(HFconnectionGraphObject):
	for neuronID, neuronName in enumerate(HFconnectionGraphObject.neuronNamelist):
		HFconnectionGraphObject.neuronIDdict[neuronName] = neuronID 
