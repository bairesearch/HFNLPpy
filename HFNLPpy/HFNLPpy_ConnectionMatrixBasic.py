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

epsilon = 1e-8  # Small epsilon value

if(pt.cuda.is_available()):
	device = pt.device("cuda")
else:
	device = pt.device("cpu")

def addContextConnectionsToGraphNeuronIDWrapper(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID):
	if(useHFconnectionMatrixBasicSplit):
		for conceptNeuronContextVectorIndex in range(contextConnectionVector.contextVectorLength):
			contextVectorSourceNeuronID = contextConnectionVector.indices[conceptNeuronContextVectorIndex]
			if(contextVectorSourceNeuronID != HFcontextVectorSparseNull):
				setConnectionGraphContextIndexNeuronID(HFconnectionGraphObject, contextConnectionVector.values[conceptNeuronContextVectorIndex], firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID, contextVectorSourceNeuronID)
	else:
		setConnectionGraphNeuronID(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID)
		
def setConnectionGraphContextIndexNeuronID(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID, contextVectorSourceNeuronID):
	if(HFconnectionMatrixBasicSplitRAM):
		#print("HFconnectionGraphObject.HFconnectionGraphMatrix = ", HFconnectionGraphObject.HFconnectionGraphMatrix)
		#print("HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID].shape = ", HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID].shape)
		HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = addContextConnectionsToGraph(HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], contextConnectionVector)
		HFconnectionGraphObject.HFconnectionGraphMatrixMax[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = pt.max(HFconnectionGraphObject.HFconnectionGraphMatrixMax[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID])
		HFconnectionGraphObject.HFconnectionGraphMatrixMin[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = pt.min(HFconnectionGraphObject.HFconnectionGraphMatrixMin[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID])				
		if(algorithmMatrixTensorDim == 4):
			assert (HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] == HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID])	#verify pytorch assignment supports both [][] and [, ] syntaxes for multidimensional assignment
	else:
		printe("setConnectionGraphContextIndex error: !useHFconnectionMatrixBasicSplitRAM has not been coded")

def setConnectionGraphNeuronID(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID):
	HFconnectionGraphObject.HFconnectionGraphMatrix[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = addContextConnectionsToGraph(HFconnectionGraphObject.HFconnectionGraphMatrix[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], contextConnectionVector)
	HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID])
	
#algorithmMatrix
def addContextConnectionsToGraph(HFconnectionGraph, contextConnectionVector):
	if(useHFconnectionMatrixBasicBool):
		HFconnectionGraph = pt.logical_and(HFconnectionGraph, contextConnectionVector)
	else:
		if(useHFconnectionMatrixBasicSparse):
			printe("addContextConnectionsToGraph error: useHFconnectionMatrixBasicSparse is incomplete")
			contextConnectionVector = contextConnectionVector.to_sparse()
		HFconnectionGraph += contextConnectionVector
	return HFconnectionGraph
	
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

def extendConceptNeuronContextVector(conceptNeuronContextVector, matrixTensorDim4):
	if(matrixTensorDim4):
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

def initialiseNeuronNameList():
	if(HFreadSavedConnectionsMatrixBasic):
		neuronNamelist = readConceptNeuronList(HFconceptNeuronListPathName)
	else:
		neuronNamelist = []
	return neuronNamelist
	


def initialiseHFconnectionMatrixBasic(dendriticBranchIndex="", contextSizeIndex=""):
	if(HFreadSavedConnectionsMatrixBasic):
		HFconnectionGraph = readHFconnectionMatrixBasic(dendriticBranchIndex, contextSizeIndex)
	else:
		HFconnectionGraph = pt.zeros([HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts], dtype=HFconnectionsMatrixType)
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
		
def initialiseHFconnectionMatrixAlgorithmMatrix(dendriticBranchIndex="", contextSizeIndex=""):
	if(HFreadSavedConnectionsMatrixBasic):
		HFconnectionGraph = readHFconnectionMatrixAlgorithmMatrix(dendriticBranchIndex, contextSizeIndex)
	else:
		HFconnectionGraph = createHFconnectionMatrixAlgorithmMatrix()
	return HFconnectionGraph
	
def readHFconnectionMatrixAlgorithmMatrix(dendriticBranchIndex="", secondDataIndex=""):
	if(useAlgorithmMatrix and not algorithmMatrixTensorDim2):
		printe("initialiseHFconnectionMatrix error: HFreadSavedConnectionsMatrixBasic does not currently support useAlgorithmMatrix and not algorithmMatrixTensorDim2")
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixAlgorithmMatrixFileName + dendriticBranchIndex + secondDataIndex + HFconceptNeuronsAlgorithmMatrixExtensionName
	HFconnectionGraph = readGraphFromCsv(HFconnectionMatrixPathName)
	conceptsSize = HFconnectionGraph.shape[0]
	spareConceptsSize = HFconnectionMatrixBasicMaxConcepts-conceptsSize
	print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	HFconnectionGraph = pt.nn.ZeroPad2d((0, spareConceptsSize, 0, spareConceptsSize))(HFconnectionGraph)
	print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	return HFconnectionGraph
		
def createHFconnectionMatrixAlgorithmMatrix():
	if(useAlgorithmMatrix):
		secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
		if(useHFconnectionMatrixBasicSplit):
			#create connection matrix columns only (for a single source context neuron ID);
			if(algorithmMatrixTensorDim == 4):
				tensorShape = (numberOfIndependentDendriticBranches, secondDataIndexMax, HFconnectionMatrixBasicMaxConcepts)
			elif(algorithmMatrixTensorDim == 3):
				tensorShape = (secondDataIndexMax, HFconnectionMatrixBasicMaxConcepts)
			else:
				tensorShape = (HFconnectionMatrixBasicMaxConcepts)
		else:
			if(algorithmMatrixTensorDim == 4):
				tensorShape = (numberOfIndependentDendriticBranches, secondDataIndexMax, HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
			elif(algorithmMatrixTensorDim == 3):
				tensorShape = (secondDataIndexMax, HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
			else:
				tensorShape = (HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
	else:
		tensorShape = (HFconnectionMatrixBasicMaxConcepts, HFconnectionMatrixBasicMaxConcepts)
	if(simulatedDendriticBranchesInitialisation):
		HFconnectionGraph = createRandomisedTensor(tensorShape)
	else:
		HFconnectionGraph = createEmptyTensor(tensorShape, useHFconnectionMatrixBasicSparse)
	if(HFconnectionMatrixGPU):
		HFconnectionGraph = HFconnectionGraph.to(device)
	return HFconnectionGraph
		
def createRandomisedTensor(tensorShape):
	if(useHFconnectionMatrixBasicSparse):
		printe("useHFconnectionMatrixBasicSparse:simulatedDendriticBranchesInitialisation not currently supported")
	else:
		emptyTensor = pt.rand(tensorShape, dtype=HFconnectionsMatrixType)*simulatedDendriticBranchesInitialisationWeight
	return emptyTensor


def getConnectionGraph(HFconnectionGraphObject, conceptNeuronContextVector, firstDataIndex, secondDataIndex, matrixTensorDim4):
	if(useHFconnectionMatrixBasicSplit):
		HFconnectionGraph = getConnectionGraphContextSubset(HFconnectionGraphObject, conceptNeuronContextVector, firstDataIndex, secondDataIndex, matrixTensorDim4)
	else:		
		HFconnectionGraph = getConnectionGraphFull(HFconnectionGraphObject, firstDataIndex, secondDataIndex)
	return HFconnectionGraph
	
if(useHFconnectionMatrixBasicSplit):
	def getConnectionGraphContextSubset(HFconnectionGraphObject, conceptNeuronContextVector, firstDataIndex, secondDataIndex, matrixTensorDim4):
		if(matrixTensorDim4):
			#gather HFconnectionGraph subset based on secondDataIndex
			if(len(conceptNeuronContextVector) == 0):
				printe("getConnectionGraphContextSubset error: len(conceptNeuronContextVector) == 0")
			HFconnectionGraphContextSubset2List = []
			for secondDataIndex, conceptNeuronContextVector2 in enumerate(conceptNeuronContextVector):	#overwrite secondDataIndex (argument is None)
				HFconnectionGraphContextSubset2 = getConnectionGraphContextSubset2(HFconnectionGraphObject, conceptNeuronContextVector2, firstDataIndex, secondDataIndex, matrixTensorDim4)
				HFconnectionGraphContextSubset2List.append(HFconnectionGraphContextSubset2)
			HFconnectionGraphContextSubset = pt.stack(HFconnectionGraphContextSubset2List, dim=1)
			#print("1 HFconnectionGraphContextSubset.shape = ", HFconnectionGraphContextSubset.shape)
		else:
			HFconnectionGraphContextSubset = getConnectionGraphContextSubset2(HFconnectionGraphObject, conceptNeuronContextVector,  firstDataIndex, secondDataIndex, matrixTensorDim4)
			#print("2 HFconnectionGraphContextSubset.shape = ", HFconnectionGraphContextSubset.shape)		
		return HFconnectionGraphContextSubset
	
	def getConnectionGraphContextSubset2(HFconnectionGraphObject, conceptNeuronContextVector, firstDataIndex, secondDataIndex, matrixTensorDim4):
		#gather HFconnectionGraph subset based on conceptNeuronContextVector.indices
		HFconnectionGraphContextIndexList = []
		if(len(conceptNeuronContextVector.indices) == 0):
			printe("getConnectionGraphContextSubset2 error: len(conceptNeuronContextVector.indices) == 0")
		for conceptNeuronContextVectorIndex in range(len(conceptNeuronContextVector.indices)):
			contextVectorSourceNeuronID = conceptNeuronContextVector.indices[conceptNeuronContextVectorIndex]
			if(contextVectorSourceNeuronID == HFcontextVectorSparseNull):
				HFconnectionGraphContextIndex = pt.zeros_like(getConnectionGraphContextIndex(HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, 0))
			else:
				HFconnectionGraphContextIndex = getConnectionGraphContextIndex(HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, contextVectorSourceNeuronID)
				HFconnectionGraphContextIndex = normaliseConnectionGraphContextIndex(HFconnectionGraphContextIndex, HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, contextVectorSourceNeuronID)
			HFconnectionGraphContextIndexList.append(HFconnectionGraphContextIndex)
		HFconnectionGraphContextSubset = pt.stack(HFconnectionGraphContextIndexList, dim=-1)
		#print("3 HFconnectionGraphContextSubset.shape = ", HFconnectionGraphContextSubset.shape)
		return HFconnectionGraphContextSubset
							
	def getConnectionGraphContextIndex(HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, contextVectorSourceNeuronID):
		if(HFconnectionMatrixBasicSplitRAM):
			if(matrixTensorDim4):
				if(algorithmMatrixTensorDim == 4):
					HFconnectionGraphContextIndex = HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][:, secondDataIndex]	#[:][secondDataIndex] syntax unsupported by pytorch
				elif(algorithmMatrixTensorDim == 3):
					HFconnectionGraphContextIndex = HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex].unsqueeze(dim=0)	#[contextVectorSourceNeuronID][firstDataIndex].unsqueeze(dim=0)
			else:
				HFconnectionGraphContextIndex = HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex]
		else:
			printe("getConnectionGraphContextIndex error: !HFconnectionMatrixBasicSplitRAM has not been coded")
		return HFconnectionGraphContextIndex
		
	def normaliseConnectionGraphContextIndex(HFconnectionGraphContextIndex, HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, contextVectorSourceNeuronID):
		#this may not be efficient to do this every time here (low memory but still relatively high computational cost; aim for low memory and low computational cost); see addContextWordsToConnectionGraphNeuronID:normaliseBatchedTensorWrapper for alternate even higher cost implementation
		if(matrixTensorDim4):
			if(algorithmMatrixTensorDim == 4):
				HFconnectionGraphMin = HFconnectionGraphObject.HFconnectionGraphMatrixMin[:, secondDataIndex]	#[:][secondDataIndex] syntax unsupported by pytorch
				HFconnectionGraphMax = HFconnectionGraphObject.HFconnectionGraphMatrixMax[:, secondDataIndex]	#[:][secondDataIndex] syntax unsupported by pytorch
			elif(algorithmMatrixTensorDim == 3):
				HFconnectionGraphMin = HFconnectionGraphObject.HFconnectionGraphMatrixMin[firstDataIndex][secondDataIndex].unsqueeze(dim=0)	#[firstDataIndex].unsqueeze(dim=0)
				HFconnectionGraphMax = HFconnectionGraphObject.HFconnectionGraphMatrixMax[firstDataIndex][secondDataIndex].unsqueeze(dim=0)	#[firstDataIndex].unsqueeze(dim=0)
		else:
			HFconnectionGraphMin = HFconnectionGraphObject.HFconnectionGraphMatrixMin[firstDataIndex][secondDataIndex]
			HFconnectionGraphMax = HFconnectionGraphObject.HFconnectionGraphMatrixMax[firstDataIndex][secondDataIndex]
		if(useHFconnectionMatrixNormaliseSoftmax):
			printe("normaliseConnectionGraphContextIndex does not support useHFconnectionMatrixNormaliseSoftmax")
		else:
			HFconnectionGraphContextIndexNormalised = (HFconnectionGraphContextIndex - HFconnectionGraphMin) / (HFconnectionGraphMax - HFconnectionGraphMin + epsilon)
		return HFconnectionGraphContextIndexNormalised
else:
	def getConnectionGraphFull(HFconnectionGraphObject, firstDataIndex, secondDataIndex):
		if(algorithmMatrixTensorDim == 4):
			HFconnectionGraph = HFconnectionGraphObject.HFconnectionGraphMatrixNormalised
		elif(algorithmMatrixTensorDim == 3):
			HFconnectionGraph = HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[firstDataIndex].unsqueeze(dim=0)	#[firstDataIndex].unsqueeze(dim=0)
		else:
			HFconnectionGraph = HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[firstDataIndex][secondDataIndex]
		return HFconnectionGraph

def convertContextVectorSparseListToDense(contextConnectionVectorData, matrixTensorDim4):
	if(useHFconnectionMatrixBasicSplit):
		if(matrixTensorDim4):
			contextConnectionVectorList = []
			for contextConnectionVectorSparse in contextConnectionVectorData:
				contextConnectionVectorList.append(contextConnectionVectorSparse.values)
			contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
			#print("contextConnectionVector.shape = ", contextConnectionVector.shape)
		else:
			contextConnectionVector = contextConnectionVectorData.values
	if(HFconnectionMatrixGPU):
		contextConnectionVector = contextConnectionVector.to(device)
	return contextConnectionVector
		
if(HFcontextVectorSparse):
	class ContextVectorTensorSparse:
		def __init__(self, contextVectorLength, indicesTensor, valuesTensor):
			self.indices = indicesTensor
			self.values = valuesTensor
			self.contextVectorLength = contextVectorLength

def createContextVectorTensorBasic():
	contextVectorLength = HFconnectionMatrixBasicMaxConcepts
	contextConnectionVector = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixType)
	return contextConnectionVector
			
def createContextVectorTensor(contextSize):
	if(HFcontextVectorSparse):
		contextVectorLength = contextSize
	else:
		contextVectorLength = HFconnectionMatrixBasicMaxConcepts
	if(HFcontextVectorSparse):
		#print("contextVectorLength = ", contextVectorLength)
		#print("HFconnectionsMatrixType = ", HFconnectionsMatrixType)
		valuesTensor = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixType)
		indicesTensor = pt.ones(contextVectorLength, dtype=pt.int64) * HFcontextVectorSparseNull
		contextConnectionVector = ContextVectorTensorSparse(contextVectorLength, indicesTensor, valuesTensor)
	else:
		contextConnectionVector = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixType)
	return contextConnectionVector
	
def createEmptyTensor(tensorShape, sparse):
	if(sparse):
		tensorDims = len(tensorShape)
		valuesTensor = pt.empty(0, dtype=HFconnectionsMatrixType)
		indicesTensor = pt.empty((tensorDims, 0), dtype=pt.int64)
		emptyTensor = pt.sparse_coo_tensor(indicesTensor, valuesTensor, tensorShape)
	else:
		emptyTensor = pt.zeros(tensorShape, dtype=HFconnectionsMatrixType)
	return emptyTensor

def writeHFconnectionMatrixBasic(HFconnectionGraph, neuronNamelist):
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixBasicFileName + HFconnectionMatrixBasicExtensionName
	HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsBasicFileName + HFconceptNeuronsBasicExtensionName
	writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
	writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)
	
def writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraph, neuronNamelist, dendriticBranchIndex="", contextSizeIndex=""):
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixAlgorithmMatrixFileName + dendriticBranchIndex + contextSizeIndex + HFconnectionMatrixAlgorithmMatrixExtensionName
	HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsAlgorithmMatrixFileName + dendriticBranchIndex + contextSizeIndex + HFconceptNeuronsAlgorithmMatrixExtensionName
	writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
	writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)

def readGraphFromCsv(filePath):
	connections = []
	with open(filePath, 'r') as f:
		reader = csv.reader(f)
		for row in (reader):
			connections.append(row)
	HFconnectionGraph = np.array(connections, dtype=HFconnectionsMatrixType)
	
	if(useAlgorithmMatrix and algorithmMatrixTensorDim==4):
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
	if(useAlgorithmMatrix and algorithmMatrixTensorDim==4):
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

		
def writeHFconnectionMatrixWrapper(HFconnectionGraphObject):
	if(HFwriteSavedConnectionsMatrixBasic):
		if(useAlgorithmMatrix):
			if(useHFconnectionMatrixBasicSplit):
				printe("writeHFconnectionMatrixWrapper error: useHFconnectionMatrixBasicSplit has not been coded")
			else:
				writeHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject)
		if(linkSimilarConceptNodesBagOfWords):
			writeHFconnectionMatrixBasic(HFconnectionGraphObject.HFconnectionGraphBasic, HFconnectionGraphObject.neuronNamelist)

def writeHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject):
	if(algorithmMatrixTensorDim==4):
		writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix, HFconnectionGraphObject.neuronNamelist)
	else:
		secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			if(algorithmMatrixTensorDim==3):
				writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex], HFconnectionGraphObject.neuronNamelist, createIndexStringDendriticBranch(dendriticBranchIndex))
			else:
				#print("secondDataIndexMax = ", secondDataIndexMax)
				#print("HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex] = ", len(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex]))
				for secondDataIndex in range(secondDataIndexMax):
					writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex], HFconnectionGraphObject.neuronNamelist, createIndexStringDendriticBranch(dendriticBranchIndex), createIndexStringSecondDataIndex(secondDataIndex))


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

def normaliseBatchedTensorWrapper(HFconnectionGraphObject, firstDataIndex, secondDataIndex):
	#this may not be efficient to do this every time here (high memory and high computational cost); see normaliseConnectionGraphContextIndex for lower cost implementation
	HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[firstDataIndex][secondDataIndex] = normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[firstDataIndex][secondDataIndex])
	
def normaliseBatchedTensor(HFconnectionGraph):
	HFconnectionGraph = HFconnectionGraph.float()
	if(useHFconnectionMatrixBasicSparse):
		printe("normaliseBatchedTensor error: useHFconnectionMatrixBasicSparse is not currently supported")
	else:
		if(useHFconnectionMatrixBasicBool):	#OLD: if(not weightStore)
			HFconnectionGraphNormalised = HFconnectionGraph
		else:
			#calculate a temporary normalised version of the HFconnectionGraph	#CHECKTHIS
			if(useHFconnectionMatrixNormaliseSoftmax):
				HFconnectionGraphNormalised = pt.nn.functional.softmax(HFconnectionGraph, dim=1)
			else:
				if(useHFconnectionMatrixBasicSparse):
					printe("normaliseBatchedTensor does not yet support useHFconnectionMatrixBasicSparse")
				else:
					min_vals, _ = pt.min(HFconnectionGraph, dim=-1, keepdim=True)
					max_vals, _ = pt.max(HFconnectionGraph, dim=-1, keepdim=True)
					HFconnectionGraphNormalised = (HFconnectionGraph - min_vals) / (max_vals - min_vals + epsilon)
	return HFconnectionGraphNormalised
	
