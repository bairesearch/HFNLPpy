"""HFNLPpy_MatrixDatabase.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Matrix Database

"""

import numpy as np
import torch as pt
import torch.nn.functional as F

from HFNLPpy_MatrixGlobalDefs import *
import HFNLPpy_ConnectionMatrixBasic
from HFNLPpy_MatrixOperations import getSecondDataIndexMax
import HFNLPpy_ConnectionMatrixOperations

def generateMatrixDatabaseFileName(sourceNeuronID):
	#sourceNeuronID corresponds to the column index of a 2D connections matrix
	filePath = matrixDatabasePathName + "/" + matrixDatabaseFileNameStart + str(sourceNeuronID) + matrixDatabaseFileNameEnd
	return filePath
	
def saveMatrixDatabaseFile(HFconnectionGraphObject, sourceNeuronID, HFconnectionGraphSourceIndex):
	HFconnectionGraphSourceIndex = squeezeHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraphSourceIndex)
	filePath = generateMatrixDatabaseFileName(sourceNeuronID)
	writeGraphToCsv(HFconnectionGraphSourceIndex, filePath)
	
def loadMatrixDatabaseFile(HFconnectionGraphObject, sourceNeuronID):
	filePath = generateMatrixDatabaseFileName(sourceNeuronID)
	HFconnectionGraphSourceIndex = readGraphFromCsv(filePath)
	HFconnectionGraphSourceIndex = padHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraphSourceIndex)
	if(HFconnectionMatrixAlgorithmGPU):
		HFconnectionGraphSourceIndex = HFconnectionGraphSourceIndex.to(HFNLPpy_ConnectionMatrixOperations.device)
	return HFconnectionGraphSourceIndex

def squeezeHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraphSourceIndex):
	networkSize = HFNLPpy_ConnectionMatrixOperations.getNetworkSize(HFconnectionGraphObject)
	maxConceptsSize = HFconnectionGraphSourceIndex.shape[-1]
	HFconnectionGraphSourceIndexSqueezed = HFconnectionGraphSourceIndex[:, :, 0:networkSize]	#squeeze to length of database
	return HFconnectionGraphSourceIndexSqueezed
	
def padHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraphSourceIndex):
	networkSize = HFNLPpy_ConnectionMatrixOperations.getNetworkSize(HFconnectionGraphObject)
	conceptsSize = HFconnectionGraphSourceIndex.shape[-1]
	#print("networkSize = ", networkSize)
	#print("conceptsSize = ", conceptsSize)
	#assert (networkSize == conceptsSize)
	#note networkSize will not necessarily equal conceptsSize (as array could have been previously written using a)
	spareConceptsSize = HFconnectionGraphObject.connectionMatrixMaxConcepts-conceptsSize
	HFconnectionGraphSourceIndexPadded = F.pad(HFconnectionGraphSourceIndex, (0, spareConceptsSize, 0, 0, 0, 0), mode='constant', value=0)	#pad the the end of the last dimension of the tensor (note that F.pad padding dimensions are reversed ordered)
	return HFconnectionGraphSourceIndexPadded
	
def initialiseMatrixDatabase(HFconnectionGraphObject):
	HFconceptNeuronListPathName = HFNLPpy_ConnectionMatrixOperations.generateConceptListFileName() 
	matrixDatabaseFileNameMinFilePath = generateMatrixDatabaseFileName(matrixDatabaseFileNameMin)

	HFNLPpy_ConnectionMatrixOperations.initialiseNeuronNameList(HFconnectionGraphObject)
		
	if(os.path.exists(matrixDatabaseFileNameMinFilePath) and os.path.exists(HFconceptNeuronListPathName)):
		print("initialiseMatrixDatabase warning: matrixDatabaseAlreadyInitialised; using existing HFconnectionGraphMatrixMin/HFconnectionGraphMatrixMax and conceptList files")
	elif(os.path.exists(matrixDatabaseFileNameMinFilePath) or os.path.exists(HFconceptNeuronListPathName)):
		printe("matrixDatabase incorrectly initialised os.path.exists(matrixDatabaseFileNameMinFilePath) XOR os.path.exists(HFconceptNeuronListPathName)")
	
	if(os.path.exists(matrixDatabaseFileNameMinFilePath)):
		HFconnectionGraphObject.HFconnectionGraphMatrixMin = loadMatrixDatabaseFile(HFconnectionGraphObject, matrixDatabaseFileNameMin)
		HFconnectionGraphObject.HFconnectionGraphMatrixMax = loadMatrixDatabaseFile(HFconnectionGraphObject, matrixDatabaseFileNameMax)

def finaliseMatrixDatabaseSentence(HFconnectionGraphObject, sentenceConceptNodeList):
	#save all tensors to drive and clear all tensors from RAM
	for conceptNode in sentenceConceptNodeList:
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		if(HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID] != None):	#ensure node has not already been deleted (in the case of replica nodes in sentence)
			saveMatrixDatabaseFile(HFconnectionGraphObject, neuronID, HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID])
			HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID] = None
	
def finaliseMatrixDatabase(HFconnectionGraphObject):
	#save all tensors to drive and clear all tensors from RAM
	saveMatrixDatabaseFile(HFconnectionGraphObject, matrixDatabaseFileNameMin, HFconnectionGraphObject.HFconnectionGraphMatrixMin)
	saveMatrixDatabaseFile(HFconnectionGraphObject, matrixDatabaseFileNameMax, HFconnectionGraphObject.HFconnectionGraphMatrixMax)
	HFNLPpy_ConnectionMatrixOperations.writeHFConceptList(HFconnectionGraphObject.neuronNamelist)
	
def readGraphFromCsv(filePath):
	graph = HFNLPpy_ConnectionMatrixOperations.readGraphFromCsv(filePath)
	
	if(algorithmMatrixTensorDim==4):
		secondDataIndexMax = getSecondDataIndexMax()
		numberOfConcepts = graph.shape[0]
		originalShape = (numberOfIndependentDendriticBranches, secondDataIndexMax, numberOfConcepts)
		graph = graph.view(originalShape)
	else:
		printe("HFNLPpy_MatrixDatabase:readGraphFromCsv error: HFconnectionMatrixAlgorithmSplitDatabase currently requires algorithmMatrixTensorDim=4 such that the file i/o code can be simplified")
	
	return graph

def writeGraphToCsv(graph, filePath):
	graph = graph.cpu()
	
	if(algorithmMatrixTensorDim==4):
		graph = graph.reshape(graph.shape[2], -1)	# Flatten the ND tensor into a 2D tensor
	else:
		printe("HFNLPpy_MatrixDatabase:writeGraphToCsv error: HFconnectionMatrixAlgorithmSplitDatabase currently requires algorithmMatrixTensorDim=4 such that the file i/o code can be simplified")
		
	HFNLPpy_ConnectionMatrixOperations.writeGraphToCsv(graph, filePath)

