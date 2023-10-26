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

from HFNLPpy_ScanGlobalDefs import *
from HFNLPpy_MatrixGlobalDefs import *
from ANNtf2_loadDataset import datasetFolderRelative
if(useAlgorithmMatrix):
	import HFNLPpy_MatrixOperations

if(pt.cuda.is_available()):
	device = pt.device("cuda")
else:
	device = pt.device("cpu")
	
def addContextConnectionsToGraph(HFconnectionGraph, neuronID, contextConnectionVector, matrixSingleTensor):
	if(matrixSingleTensor):
		printe("addContextConnectionsToGraph error: algorithmMatrixSingleTensorEfficientAdd incomplete")
		#maskSummedTopKindices = maskSummedTopKindices.squeeze(-1).gather(dim=1, index=maskSummedTopK.indices)
		HFconnectionGraph[neuronID] += contextConnectionVector
	else:
		contextConnectionVectorPadded = padContextConnectionVector(contextConnectionVector)
		if(useHFconnectionMatrixBasicBool):
			pt.logical_and(HFconnectionGraph[neuronID], contextConnectionVectorPadded)
		else:
			#print("contextConnectionVectorPadded.shape = ", contextConnectionVectorPadded.shape)
			#print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
			#print("neuronID = ", neuronID)
			HFconnectionGraph[neuronID] += contextConnectionVectorPadded
	#print("HFconnectionGraph[neuronID] = ", HFconnectionGraph[neuronID])

def padContextConnectionVector(contextConnectionVector):
	conceptsSize = contextConnectionVector.shape[0]
	spareConceptsSize = HFconnectionMatrixBasicMaxConcepts-conceptsSize
	contextConnectionVectorPadded = F.pad(contextConnectionVector, (0, spareConceptsSize), mode='constant', value=0)
	#contextConnectionVectorPadded = pt.nn.ZeroPad1d(spareConceptsSize)(contextConnectionVector)	#requires later version of pytorch
	return contextConnectionVectorPadded

def extendConceptNeuronContextVector(conceptNeuronContextVector, matrixSingleTensor):
	if(matrixSingleTensor):
		conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVector, dim=1)
		conceptNeuronContextVectorExtended = conceptNeuronContextVectorExtended.repeat(1, HFconnectionMatrixBasicMaxConcepts, 1)	#len(HFconnectionGraphObject.neuronNamelist)	
		conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVectorExtended, dim=0)
	else:
		conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVector, dim=0)
		conceptNeuronContextVectorExtended = conceptNeuronContextVectorExtended.repeat(HFconnectionMatrixBasicMaxConcepts, 1)	#len(HFconnectionGraphObject.neuronNamelist)	
	return conceptNeuronContextVectorExtended
	
def createDiagonalMatrix(squareMatrix, width):
	diagonalMatrix = pt.tril(squareMatrix, diagonal=0) - pt.tril(squareMatrix, diagonal=-width)
	#diagonalMatrix = pt.tril(squareMatrix, diagonal=0) * torch.triu(squareMatrix, diagonal=width)
	return diagonalMatrix
	
def readHFconnectionMatrix(dendriticBranchIndex="", contextSizeIndex=""):
	if(HFreadSavedConnectionsMatrixBasic):
		HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName + dendriticBranchIndex + contextSizeIndex + HFconnectionMatrixExtensionName
		HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName + dendriticBranchIndex + contextSizeIndex + HFconceptNeuronsExtensionName
		neuronNamelist = readConceptNeuronList(HFconceptNeuronListPathName)
		HFconnectionGraph = readGraphFromCsv(HFconnectionMatrixPathName)
		conceptsSize = HFconnectionGraph.shape[0]
		spareConceptsSize = HFconnectionMatrixBasicMaxConcepts-conceptsSize
		print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
		HFconnectionGraph = pt.nn.ZeroPad2d((0, spareConceptsSize, 0, spareConceptsSize))(HFconnectionGraph)
		print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	else:
		neuronNamelist = []
		if(useAlgorithmMatrix):
			secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
			if(algorithmMatrixSingleTensor):
				tensorShape = (numberOfIndependentDendriticBranches, secondDataIndexMax, HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
			elif(algorithmMatrixSANImethodUseActivationAcrossSegments):
				tensorShape = (secondDataIndexMax, HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
			else:
				tensorShape = (HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
		else:
			tensorShape = (HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
		if(simulatedDendriticBranchesInitialisation):
			HFconnectionGraph = createRandomisedTensor(tensorShape)
		else:
			HFconnectionGraph = createEmptyTensor(tensorShape)
			
	return neuronNamelist, HFconnectionGraph

def createRandomisedTensor(tensorShape):
	if(useHFconnectionMatrixBasicSparse):
		printe("useHFconnectionMatrixBasicSparse:simulatedDendriticBranchesInitialisation not currently supported")
	else:
		emptyTensor = pt.rand(tensorShape, dtype=HFconnectionsMatrixType)*simulatedDendriticBranchesInitialisationWeight
	return emptyTensor

	
def createEmptyTensor(tensorShape):
	if(useHFconnectionMatrixBasicSparse):
		tensorDims = len(tensorShape)
		emptyTensor = pt.sparse_coo_tensor(pt.empty((tensorDims, 0), dtype=pt.int64), pt.empty(0), tensorShape)
	else:
		emptyTensor = pt.zeros(tensorShape, dtype=HFconnectionsMatrixType)
	return emptyTensor

def writeHFconnectionMatrix(neuronNamelist, HFconnectionGraph, dendriticBranchIndex="", contextSizeIndex=""):
	if(HFwriteSavedConnectionsMatrixBasic):
		HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixFileName + dendriticBranchIndex + contextSizeIndex + HFconnectionMatrixExtensionName
		HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsFileName + dendriticBranchIndex + contextSizeIndex + HFconceptNeuronsExtensionName
		writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
		writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)

def readGraphFromCsv(filePath):
	connections = []
	with open(filePath, 'r') as f:
		reader = csv.reader(f)
		for row in (reader):
			connections.append(row)
	HFconnectionGraph = np.array(connections, dtype=HFconnectionsMatrixType)
	
	if(useAlgorithmMatrix and algorithmMatrixSingleTensor):
		numberOfConcepts = graph.shape[0]
		originalShape = (numberOfIndependentDendriticBranches, contextSizeMax, numberOfConcepts, numberOfConcepts)
		graph = graph.view(originalShape)
	if(useHFconnectionMatrixBasicSparse):
		graph = graph.to_sparse()
	graph = graph.to(device)
	
	return HFconnectionGraph

def writeGraphToCsv(graph, filePath):

	graph = graph.cpu()
	if(useHFconnectionMatrixBasicSparse):
		graph = graph.to_dense()
	if(useAlgorithmMatrix and algorithmMatrixSingleTensor):
		graph = graph.view(tensor.size(0), -1)	# Flatten the ND tensor into a 2D tensor
		
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
		
def readHFconnectionMatrixWrapper(HFconnectionGraphObject):
	if(useAlgorithmMatrix):
		if(algorithmMatrixSingleTensor):
			HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.HFconnectionGraphMatrix = readHFconnectionMatrix()
			HFconnectionGraphObject.HFconnectionGraphMatrixNormalised = HFNLPpy_MatrixOperations.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix)
		else:
			secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
			for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
				if(algorithmMatrixSANImethodUseActivationAcrossSegments):
					HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex]= readHFconnectionMatrix(createIndexStringDendriticBranch(dendriticBranchIndex))
					HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex] = HFNLPpy_MatrixOperations.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex])
				else:
					for secondDataIndex in range(secondDataIndexMax):
						HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex] = readHFconnectionMatrix(createIndexStringDendriticBranch(dendriticBranchIndex), createIndexStringSecondDataIndex(secondDataIndex))
						HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][secondDataIndex] = HFNLPpy_MatrixOperations.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex])
	else:
		HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.HFconnectionGraphBasic = readHFconnectionMatrix()

def writeHFconnectionMatrixWrapper(HFconnectionGraphObject):
	if(useAlgorithmMatrix):
		if(algorithmMatrixSingleTensor):
			writeHFconnectionMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix, HFconnectionGraphObject.neuronNamelist)
		else:
			secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
			for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
				if(algorithmMatrixSANImethodUseActivationAcrossSegments):
					writeHFconnectionMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex], HFconnectionGraphObject.neuronNamelist, createIndexStringDendriticBranch(dendriticBranchIndex))
				else:
					for secondDataIndex in range(secondDataIndexMax):
						writeHFconnectionMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex], HFconnectionGraphObject.neuronNamelist, createIndexStringDendriticBranch(dendriticBranchIndex), createIndexStringSecondDataIndex(secondDataIndex))
	else:
		writeHFconnectionMatrix(HFconnectionGraphObject.HFconnectionGraphBasic, HFconnectionGraphObject.neuronNamelist)

def createIndexStringDendriticBranch(dendriticBranchIndex):
	return "dendriticBranchIndex" + str(dendriticBranchIndex)
def createIndexStringSecondDataIndex(secondDataIndex):
	return "secondDataIndex" + str(secondDataIndex)		
'''
def createIndexStringContextSizeIndex(contextSizeIndex):
	return "contextSizeIndex" + str(contextSizeIndex)	
def createIndexStringSequentialSegmentIndex(sequentialSegmentIndex):
	return "sequentialSegmentIndex" + str(sequentialSegmentIndex)	
'''

