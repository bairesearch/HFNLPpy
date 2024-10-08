"""HFNLPpy_ConnectionMatrixAlgorithm.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Connection Matrix Algorithm

"""

import numpy as np
import torch as pt
import csv

from HFNLPpy_MatrixGlobalDefs import *
from ANNtf2_loadDataset import datasetFolderRelative
import HFNLPpy_MatrixOperations
import HFNLPpy_ConnectionMatrixBasic
import HFNLPpy_ConnectionMatrixOperations

epsilon = 1e-8  # Small epsilon value

def addContextConnectionsToGraphNeuronIDWrapper(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID):
	if(HFconnectionMatrixAlgorithmSplit):
		for conceptNeuronContextVectorIndex in range(contextConnectionVector.contextVectorLength):
			contextVectorSourceNeuronID = contextConnectionVector.indices[conceptNeuronContextVectorIndex]
			if(contextVectorSourceNeuronID != HFcontextVectorSparseNull):
				setConnectionGraphContextIndexNeuronID(HFconnectionGraphObject, contextConnectionVector.values[conceptNeuronContextVectorIndex], firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID, contextVectorSourceNeuronID)
	else:
		setConnectionGraphNeuronID(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID)
		
def setConnectionGraphContextIndexNeuronID(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID, contextVectorSourceNeuronID):
	#preconditions: if(HFconnectionMatrixAlgorithmSplitDatabase): assume HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID] has already been loaded into RAM from hard drive
	#print("contextConnectionVector = ", contextConnectionVector)
	#print("setConnectionGraphContextIndexNeuronID; contextVectorSourceNeuronID = ", contextVectorSourceNeuronID)
	#print("1 HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID].shape = ", HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID].shape)
	HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = addContextConnectionsToGraph(HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], contextConnectionVector)
	HFconnectionGraphObject.HFconnectionGraphMatrixMax[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = pt.max(HFconnectionGraphObject.HFconnectionGraphMatrixMax[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID])
	HFconnectionGraphObject.HFconnectionGraphMatrixMin[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = pt.min(HFconnectionGraphObject.HFconnectionGraphMatrixMin[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID])				
	if(algorithmMatrixTensorDim == 4):
		assert (HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] == HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID])	#verify pytorch assignment supports both [][] and [, ] syntaxes for multidimensional assignment

def setConnectionGraphNeuronID(HFconnectionGraphObject, contextConnectionVector, firstDataIndex, secondDataIndex, HFconnectionGraphNeuronID):
	HFconnectionGraphObject.HFconnectionGraphMatrix[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = addContextConnectionsToGraph(HFconnectionGraphObject.HFconnectionGraphMatrix[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID], contextConnectionVector)
	HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID] = normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[firstDataIndex][secondDataIndex][HFconnectionGraphNeuronID])
	
def addContextConnectionsToGraph(HFconnectionGraph, contextConnectionVector):
	if(useHFconnectionMatrixAlgorithmBool):
		HFconnectionGraph = pt.logical_and(HFconnectionGraph, contextConnectionVector)
	else:
		if(HFconnectionMatrixAlgorithmSparse):
			printe("addContextConnectionsToGraph error: HFconnectionMatrixAlgorithmSparse is incomplete")
			contextConnectionVector = contextConnectionVector.to_sparse()
		HFconnectionGraph += contextConnectionVector
	return HFconnectionGraph

def extendConceptNeuronContextVector(HFconnectionGraphObject, conceptNeuronContextVector, matrixTensorDim4):
	if(matrixTensorDim4):
		conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVector, dim=1)
		conceptNeuronContextVectorExtended = conceptNeuronContextVectorExtended.repeat(1, HFconnectionGraphObject.connectionMatrixMaxConcepts, 1)	#len(HFconnectionGraphObject.neuronNamelist)	#repeat across number of target neuron candidates in database
		conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVectorExtended, dim=0)
	else:
		conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVector, dim=0)
		conceptNeuronContextVectorExtended = conceptNeuronContextVectorExtended.repeat(HFconnectionGraphObject.connectionMatrixMaxConcepts, 1)	#len(HFconnectionGraphObject.neuronNamelist)	#repeat across number of target neuron candidates in database
	return conceptNeuronContextVectorExtended
	
def initialiseHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, readSavedConnectionsMatrixAlgorithm, connectionMatrixAlgorithmSplit, dtype, dendriticBranchIndex="", contextSizeIndex=""):
	if(readSavedConnectionsMatrixAlgorithm):
		HFconnectionGraph = readHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, dendriticBranchIndex, contextSizeIndex)
	else:
		HFconnectionGraph = createHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, connectionMatrixAlgorithmSplit, dtype)
	return HFconnectionGraph
	
def readHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, dendriticBranchIndex="", secondDataIndex=""):
	if(useAlgorithmMatrix and not algorithmMatrixTensorDim2):
		printe("initialiseHFconnectionMatrix error: HFreadSavedConnectionsMatrixAlgorithm does not currently support useAlgorithmMatrix and not algorithmMatrixTensorDim2")
	HFconnectionMatrixPathName = generateHFconnectionMatrixAlgorithmMatrixFileName(dendriticBranchIndex, contextSizeIndex)
	HFconnectionGraph = readGraphFromCsv(HFconnectionMatrixPathName)
	HFconnectionGraph = padHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraph)
	return HFconnectionGraph

def squeezeHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraph):
	networkSize = HFNLPpy_ConnectionMatrixOperations.getNetworkSize(HFconnectionGraphObject)
	maxConceptsSize = HFconnectionGraphSourceIndex.shape[-1]
	if(algorithmMatrixTensorDim == 4):
		HFconnectionGraphSourceIndexSqueezed = HFconnectionGraph[:, :, 0:networkSize, 0:networkSize]	#squeeze to length of database
	elif(algorithmMatrixTensorDim == 3):
		HFconnectionGraphSourceIndexSqueezed = HFconnectionGraph[:, 0:networkSize, 0:networkSize]	#squeeze to length of database
	elif(algorithmMatrixTensorDim == 2):
		HFconnectionGraphSourceIndexSqueezed = HFconnectionGraph[0:networkSize, 0:networkSize]	#squeeze to length of database
	else:
		printe("padHFconnectionGraph error: illegal algorithmMatrixTensorDim")
	return HFconnectionGraphSourceIndexSqueezed
	
def padHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraphSourceIndex):
	networkSize = HFNLPpy_ConnectionMatrixOperations.getNetworkSize(HFconnectionGraphObject)
	conceptsSize = HFconnectionGraphSourceIndex.shape[-1]
	assert (networkSize == conceptsSize)
	spareConceptsSize = HFconnectionGraphObject.connectionMatrixMaxConcepts-networkSize
	if(algorithmMatrixTensorDim == 4):
		HFconnectionGraphSourceIndexPadded = F.pad(HFconnectionGraph, (0, spareConceptsSize, 0, spareConceptsSize, 0, 0, 0, 0), mode='constant', value=0)	#pad the the end of the last dimensions of the tensor (note that F.pad padding dimensions are reversed ordered)
	elif(algorithmMatrixTensorDim == 3):
		HFconnectionGraphSourceIndexPadded = F.pad(HFconnectionGraph, (0, spareConceptsSize, 0, spareConceptsSize, 0, 0), mode='constant', value=0)	#pad the the end of the last dimensions of the tensor (note that F.pad padding dimensions are reversed ordered)
	elif(algorithmMatrixTensorDim == 2):
		HFconnectionGraphSourceIndexPadded = F.pad(HFconnectionGraph, (0, spareConceptsSize, 0, spareConceptsSize), mode='constant', value=0)	#pad the the end of the last dimensions of the tensor (note that F.pad padding dimensions are reversed ordered)
		#HFconnectionGraph = pt.nn.ZeroPad2d((0, spareConceptsSize, 0, spareConceptsSize))(HFconnectionGraph)
	else:
		printe("padHFconnectionGraph error: illegal algorithmMatrixTensorDim")
	return HFconnectionGraphSourceIndexPadded
	
def createHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, connectionMatrixAlgorithmSplit, dtype):
	secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
	if(connectionMatrixAlgorithmSplit):
		#create connection matrix columns only (for a single source context neuron ID);
		if(algorithmMatrixTensorDim == 4):
			tensorShape = (numberOfIndependentDendriticBranches, secondDataIndexMax, HFconnectionGraphObject.connectionMatrixMaxConcepts)
		elif(algorithmMatrixTensorDim == 3):
			tensorShape = (secondDataIndexMax, HFconnectionGraphObject.connectionMatrixMaxConcepts)
		elif(algorithmMatrixTensorDim == 2):
			tensorShape = (HFconnectionGraphObject.connectionMatrixMaxConcepts)
		else:
			printe("createHFconnectionMatrixAlgorithmMatrix error: invalid algorithmMatrixTensorDim")
	else:
		if(algorithmMatrixTensorDim == 4):
			tensorShape = (numberOfIndependentDendriticBranches, secondDataIndexMax, HFconnectionGraphObject.connectionMatrixMaxConcepts, HFconnectionGraphObject.connectionMatrixMaxConcepts)
		elif(algorithmMatrixTensorDim == 3):
			tensorShape = (secondDataIndexMax, HFconnectionGraphObject.connectionMatrixMaxConcepts, HFconnectionGraphObject.connectionMatrixMaxConcepts)
		elif(algorithmMatrixTensorDim == 2):
			tensorShape = (HFconnectionGraphObject.connectionMatrixMaxConcepts, HFconnectionGraphObject.connectionMatrixMaxConcepts)
		else:
			printe("createHFconnectionMatrixAlgorithmMatrix error: invalid algorithmMatrixTensorDim")
	if(simulatedDendriticBranchesInitialisation):
		HFconnectionGraph = createRandomisedTensor(tensorShape, dtype)
	else:
		HFconnectionGraph = createEmptyTensor(tensorShape, HFconnectionMatrixAlgorithmSparse, dtype)
	if(HFconnectionMatrixAlgorithmGPU):
		HFconnectionGraph = HFconnectionGraph.to(HFNLPpy_ConnectionMatrixOperations.device)
	return HFconnectionGraph
		
def createRandomisedTensor(tensorShape, dtype=HFconnectionsMatrixAlgorithmType):
	if(HFconnectionMatrixAlgorithmSparse):
		printe("HFconnectionMatrixAlgorithmSparse:simulatedDendriticBranchesInitialisation not currently supported")
	else:
		emptyTensor = pt.rand(tensorShape, dtype=dtype)*simulatedDendriticBranchesInitialisationWeight
	return emptyTensor

def getConnectionGraph(HFconnectionGraphObject, conceptNeuronContextVector, firstDataIndex, secondDataIndex, matrixTensorDim4):
	if(HFconnectionMatrixAlgorithmSplit):
		HFconnectionGraph = getConnectionGraphContextSubset(HFconnectionGraphObject, conceptNeuronContextVector, firstDataIndex, secondDataIndex, matrixTensorDim4)
	else:		
		HFconnectionGraph = getConnectionGraphFull(HFconnectionGraphObject, firstDataIndex, secondDataIndex)
	return HFconnectionGraph
	
if(HFconnectionMatrixAlgorithmSplit):
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
		else:
			HFconnectionGraphContextSubset = getConnectionGraphContextSubset2(HFconnectionGraphObject, conceptNeuronContextVector,  firstDataIndex, secondDataIndex, matrixTensorDim4)
		return HFconnectionGraphContextSubset
	
	def getConnectionGraphContextSubset2(HFconnectionGraphObject, conceptNeuronContextVector, firstDataIndex, secondDataIndex, matrixTensorDim4):
		#gather HFconnectionGraph subset based on conceptNeuronContextVector.indices
		HFconnectionGraphContextIndexList = []
		if(len(conceptNeuronContextVector.indices) == 0):
			printe("getConnectionGraphContextSubset2 error: len(conceptNeuronContextVector.indices) == 0")
		for conceptNeuronContextVectorIndex in range(len(conceptNeuronContextVector.indices)):
			contextVectorSourceNeuronID = conceptNeuronContextVector.indices[conceptNeuronContextVectorIndex]
			if(contextVectorSourceNeuronID == HFcontextVectorSparseNull):
				HFNLPpy_MatrixOperations.getSecondDataIndexMax()
				HFconnectionGraphContextIndex = pt.zeros([numberOfIndependentDendriticBranches , HFconnectionGraphObject.connectionMatrixMaxConcepts])
				if(HFconnectionMatrixAlgorithmGPU):
					HFconnectionGraphContextIndex = HFconnectionGraphContextIndex.to(HFNLPpy_ConnectionMatrixOperations.device)
			else:
				HFconnectionGraphContextIndex = getConnectionGraphContextIndex(HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, contextVectorSourceNeuronID)
				HFconnectionGraphContextIndex = normaliseConnectionGraphContextIndex(HFconnectionGraphContextIndex, HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, contextVectorSourceNeuronID)
			HFconnectionGraphContextIndexList.append(HFconnectionGraphContextIndex)
		HFconnectionGraphContextSubset = pt.stack(HFconnectionGraphContextIndexList, dim=-1)
		return HFconnectionGraphContextSubset
							
	def getConnectionGraphContextIndex(HFconnectionGraphObject, firstDataIndex, secondDataIndex, matrixTensorDim4, contextVectorSourceNeuronID):
		#preconditions: if(HFconnectionMatrixAlgorithmSplitDatabase): assume HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID] has already been loaded into RAM from hard drive
		if(matrixTensorDim4):
			if(algorithmMatrixTensorDim == 4):
				HFconnectionGraphContextIndex = HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][:, secondDataIndex]	#[:][secondDataIndex] syntax unsupported by pytorch
			elif(algorithmMatrixTensorDim == 3):
				HFconnectionGraphContextIndex = HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex].unsqueeze(dim=0)	#[contextVectorSourceNeuronID][firstDataIndex].unsqueeze(dim=0)
		else:
			HFconnectionGraphContextIndex = HFconnectionGraphObject.HFconnectionGraphMatrix[contextVectorSourceNeuronID][firstDataIndex][secondDataIndex]
		if(HFconnectionMatrixAlgorithmGPU):
			HFconnectionGraphContextIndex = HFconnectionGraphContextIndex.to(HFNLPpy_ConnectionMatrixOperations.device)
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
			
		if(HFconnectionMatrixAlgorithmNormalise=="softmax"):
			printe("normaliseConnectionGraphContextIndex does not support HFconnectionMatrixAlgorithmNormalise==softmax")
		elif(HFconnectionMatrixAlgorithmNormalise=="tanh"):
			HFconnectionGraphContextIndexNormalised = HFconnectionGraphContextIndex.tanh()
		#elif(HFconnectionMatrixAlgorithmNormalise=="xsech"):
		#	HFconnectionGraphContextIndexNormalised = HFconnectionGraphContextIndex * 1/HFconnectionGraphContextIndex.cosh()		
		elif(HFconnectionMatrixAlgorithmNormalise=="linear"):
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
	if(HFconnectionMatrixAlgorithmSplit):
		if(matrixTensorDim4):
			contextConnectionVectorList = []
			for contextConnectionVectorSparse in contextConnectionVectorData:
				contextConnectionVectorList.append(contextConnectionVectorSparse.values)
			contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
			#print("contextConnectionVector.shape = ", contextConnectionVector.shape)
		else:
			contextConnectionVector = contextConnectionVectorData.values
	if(HFconnectionMatrixAlgorithmGPU):
		contextConnectionVector = contextConnectionVector.to(HFNLPpy_ConnectionMatrixOperations.device)
	return contextConnectionVector
		
if(HFconnectionMatrixAlgorithmContextVectorSparse):
	class ContextVectorTensorSparse:
		def __init__(self, contextVectorLength, indicesTensor, valuesTensor):
			self.indices = indicesTensor
			self.values = valuesTensor
			self.contextVectorLength = contextVectorLength
			
def createContextVectorTensor(HFconnectionGraphObject, contextSize):
	if(HFconnectionMatrixAlgorithmContextVectorSparse):
		contextVectorLength = contextSize
	else:
		contextVectorLength = HFconnectionGraphObject.connectionMatrixMaxConcepts
	if(HFconnectionMatrixAlgorithmContextVectorSparse):
		#print("contextVectorLength = ", contextVectorLength)
		#print("HFconnectionsMatrixAlgorithmType = ", HFconnectionsMatrixAlgorithmType)
		valuesTensor = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixAlgorithmType)
		indicesTensor = pt.ones(contextVectorLength, dtype=pt.int64) * HFcontextVectorSparseNull
		contextConnectionVector = ContextVectorTensorSparse(contextVectorLength, indicesTensor, valuesTensor)
	else:
		contextConnectionVector = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixAlgorithmType)
	return contextConnectionVector
	
def createEmptyTensor(tensorShape, sparse, dtype=HFconnectionsMatrixAlgorithmType):
	if(sparse):
		tensorDims = len(tensorShape)
		valuesTensor = pt.empty(0, dtype=dtype)
		indicesTensor = pt.empty((tensorDims, 0), dtype=pt.int64)
		emptyTensor = pt.sparse_coo_tensor(indicesTensor, valuesTensor, tensorShape)
	else:
		emptyTensor = pt.zeros(tensorShape, dtype=dtype)
	return emptyTensor

def createContextInputTensor(HFconnectionGraphObject):	#tensor representing single input
	contextInputLength = 1
	if(HFconnectionMatrixAlgorithmContextVectorSparse):
		#print("contextInputLength = ", contextInputLength)
		#print("HFconnectionsMatrixAlgorithmType = ", HFconnectionsMatrixAlgorithmType)
		valuesTensor = pt.zeros(contextInputLength, dtype=HFconnectionsMatrixAlgorithmType)
		indicesTensor = pt.ones(contextInputLength, dtype=pt.int64) * HFcontextVectorSparseNull
		contextConnectionScalar = ContextVectorTensorSparse(contextInputLength, indicesTensor, valuesTensor)
	else:
		contextConnectionScalar = pt.zeros(contextInputLength, dtype=HFconnectionsMatrixAlgorithmType)
	return contextConnectionScalar
	
def createEmptyInputTensor(sparse):
	tensorShape = (1)
	if(sparse):
		tensorDims = len(tensorShape)
		valuesTensor = pt.empty(0, dtype=HFconnectionsMatrixAlgorithmType)
		indicesTensor = pt.empty((tensorDims, 0), dtype=pt.int64)
		emptyTensor = pt.sparse_coo_tensor(indicesTensor, valuesTensor, tensorShape)
	else:
		emptyTensor = pt.zeros(tensorShape, dtype=HFconnectionsMatrixAlgorithmType)
	return emptyTensor
	
def writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraph, dendriticBranchIndex="", contextSizeIndex=""):
	HFconnectionGraph = squeezeHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraph)
	HFconnectionMatrixPathName = generateHFconnectionMatrixAlgorithmMatrixFileName(dendriticBranchIndex, contextSizeIndex)
	writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)

def generateHFconnectionMatrixAlgorithmMatrixFileName(dendriticBranchIndex="", contextSizeIndex=""):
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixAlgorithmMatrixFileName + dendriticBranchIndex + contextSizeIndex + HFconnectionMatrixAlgorithmMatrixExtensionName
	return HFconnectionMatrixPathName
	
def writeHFconnectionMatrixAlgorithmWrapper(HFconnectionGraphObject):
	#if(useAlgorithmMatrix):
	if(HFwriteSavedConnectionsMatrixAlgorithm):
		if(HFconnectionMatrixAlgorithmSplit):
			printe("writeHFconnectionMatrixAlgorithmWrapper error: HFconnectionMatrixAlgorithmSplit has not been coded")
		else:
			writeHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject)
	if(HFwriteSavedConceptList):
		HFNLPpy_ConnectionMatrixOperations.writeHFConceptList(HFconnectionGraphObject.neuronNamelist)

def writeHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject):
	if(algorithmMatrixTensorDim==4):
		writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix, HFconnectionGraphObject.neuronNamelist)
	else:
		secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			if(algorithmMatrixTensorDim==3):
				writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex], HFconnectionGraphObject.neuronNamelist, createIndexStringDendriticBranch(dendriticBranchIndex))
			else:
				#print("secondDataIndexMax = ", secondDataIndexMax)
				#print("HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex] = ", len(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex]))
				for secondDataIndex in range(secondDataIndexMax):
					writeHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex], HFconnectionGraphObject.neuronNamelist, createIndexStringDendriticBranch(dendriticBranchIndex), createIndexStringSecondDataIndex(secondDataIndex))

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
	if(HFconnectionMatrixAlgorithmSparse):
		printe("normaliseBatchedTensor error: HFconnectionMatrixAlgorithmSparse is not currently supported")
	else:
		if(useHFconnectionMatrixAlgorithmBool):	#OLD: if(not weightStore)
			HFconnectionGraphNormalised = HFconnectionGraph
		else:
			#calculate a temporary normalised version of the HFconnectionGraph	#CHECKTHIS
			if(HFconnectionMatrixAlgorithmNormalise=="softmax"):
				HFconnectionGraphNormalised = pt.nn.functional.softmax(HFconnectionGraph, dim=1)
			elif(HFconnectionMatrixAlgorithmNormalise=="tanh"):
				HFconnectionGraphNormalised = HFconnectionGraph.tanh()
			#elif(HFconnectionMatrixAlgorithmNormalise=="xsech"):
			#	HFconnectionGraphNormalised = HFconnectionGraph * 1/HFconnectionGraph.cosh()		
			elif(HFconnectionMatrixAlgorithmNormalise=="linear"):
				if(HFconnectionMatrixAlgorithmSparse):
					printe("normaliseBatchedTensor does not yet support HFconnectionMatrixAlgorithmSparse")
				else:
					min_vals, _ = pt.min(HFconnectionGraph, dim=-1, keepdim=True)
					max_vals, _ = pt.max(HFconnectionGraph, dim=-1, keepdim=True)
					HFconnectionGraphNormalised = (HFconnectionGraph - min_vals) / (max_vals - min_vals + epsilon)
	return HFconnectionGraphNormalised
	

def readGraphFromCsv(filePath):
	graph = HFNLPpy_ConnectionMatrixOperations.readGraphFromCsv(filePath)
	
	if(algorithmMatrixTensorDim==4):
		secondDataIndexMax = getSecondDataIndexMax()
		numberOfConcepts = graph.shape[0]
		originalShape = (numberOfIndependentDendriticBranches, secondDataIndexMax, numberOfConcepts, numberOfConcepts)
		graph = graph.view(originalShape)
	elif(algorithmMatrixTensorDim==3):
		printe("HFNLPpy_ConnectionMatrixAlgorithm:readGraphFromCsv error: HFwriteSavedConnectionsMatrixAlgorithm/HFreadSavedConnectionsMatrixAlgorithm currently requires algorithmMatrixTensorDim=2 or algorithmMatrixTensorDim=4 such that the file i/o code can be simplified")
	
	if(HFconnectionMatrixAlgorithmSparse):
		graph = graph.to_sparse()
	
	return graph

def writeGraphToCsv(graph, filePath):
	graph = graph.cpu()
	if(HFconnectionMatrixAlgorithmSparse):
		graph = graph.to_dense()
		
	if(algorithmMatrixTensorDim==4):
		graph = graph.view(graph.shape[2], -1)	# Flatten the ND tensor into a 2D tensor
	elif(algorithmMatrixTensorDim==3):
		printe("HFNLPpy_ConnectionMatrixAlgorithm:writeGraphToCsv error: HFwriteSavedConnectionsMatrixAlgorithm/HFreadSavedConnectionsMatrixAlgorithm currently requires algorithmMatrixTensorDim=2 or algorithmMatrixTensorDim=4 such that the file i/o code can be simplified")
		
	HFNLPpy_ConnectionMatrixOperations.writeGraphToCsv(graph, filePath)

def updateSynapsesMemoryForget(HFconnectionGraph, sentenceConceptNodeList, connectionMatrixAlgorithmSplit):
	if(connectionMatrixAlgorithmSplit):
		#forget every synapse in every neuron in sentence
		for w1, sentence in enumerate(sentenceConceptNodeList):
			if(forgetSynapsesUncorrelated):
				HFconnectionGraph[w1][HFconnectionGraph[w1]<=forgetSynapsesUncorrelatedThreshold] = pt.clamp(HFconnectionGraph[w1][HFconnectionGraph[w1]<=forgetSynapsesUncorrelatedThreshold] - forgetSynapsesRate, min=0)
				#print("HFconnectionGraph[w1] = ", HFconnectionGraph[w1])
			else:
				HFconnectionGraph[w1] = pt.clamp(HFconnectionGraph[w1] - forgetSynapsesRate, min=0)
	else:
		#forget every synapse in network
		if(forgetSynapsesUncorrelated):
			HFconnectionGraph[HFconnectionGraph<forgetSynapsesUncorrelatedThreshold] = pt.clamp(HFconnectionGraph[HFconnectionGraph<forgetSynapsesUncorrelatedThreshold] - forgetSynapsesRate, min=0)
		else:
			HFconnectionGraph[:] = pt.clamp(HFconnectionGraph[:] - forgetSynapsesRate, min=0)
	
