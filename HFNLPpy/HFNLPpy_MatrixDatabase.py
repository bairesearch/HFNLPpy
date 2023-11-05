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
import csv
import os

from HFNLPpy_MatrixGlobalDefs import *
import HFNLPpy_ConnectionMatrixBasic
from HFNLPpy_MatrixOperations import getSecondDataIndexMax

def generateMatrixDatabaseFileName(sourceNeuronID):
	#sourceNeuronID corresponds to the column index of a 2D connections matrix
	filePath = matrixDatabasePathName + matrixDatabaseFileNameStart + str(sourceNeuronID) + matrixDatabaseFileNameEnd
	return filePath
	
def saveMatrixDatabaseFile(sourceNeuronID, HFconnectionGraphSourceIndex):
	filePath = generateMatrixDatabaseFileName(sourceNeuronID)
	writeGraphToCsv(HFconnectionGraphSourceIndex, filePath)
	
def loadMatrixDatabaseFile(HFconnectionGraphObject, sourceNeuronID):
	filePath = generateMatrixDatabaseFileName(sourceNeuronID)
	HFconnectionGraphSourceIndex = readGraphFromCsv(filePath)
	#HFconnectionGraphSourceIndex = padHFconnectionGraph(HFconnectionGraphSourceIndex, HFconnectionGraphObject)	#not currently used as HFconnectionGraphObject.HFconnectionGraph is intialised with many blank rows (HFconnectionMatrixBasicMaxConcepts)
	return HFconnectionGraphSourceIndex

def getDatabaseSize(HFconnectionGraphObject):
	databaseSize = len(HFconnectionGraphObject.neuronNamelist)	#or len(HFconnectionGraphObject.neuronIDdict)
	return databaseSize

def padHFconnectionGraph(HFconnectionGraphSourceIndex, HFconnectionGraphObject):
	print("HFconnectionGraphSourceIndex.shape = ", HFconnectionGraphSourceIndex.shape)
	databaseSize = getDatabaseSize(HFconnectionGraphObject)
	conceptsSize = HFconnectionGraphSourceIndex.shape[-1]
	spareConceptsSize = databaseSize-conceptsSize
	HFconnectionGraphSourceIndexPadded = F.pad(HFconnectionGraphSourceIndex, (0, spareConceptsSize, 0, 0, 0, 0), mode='constant', value=0)
	print("HFconnectionGraphSourceIndexPadded.shape = ", HFconnectionGraphSourceIndexPadded.shape)
	return HFconnectionGraphSourceIndexPadded
	
def initialiseMatrixDatabase(HFconnectionGraphObject):
	HFconceptNeuronListPathName = HFNLPpy_ConnectionMatrixBasic.generateConceptListFileName() 
	matrixDatabaseFileNameMinFilePath = generateMatrixDatabaseFileName(matrixDatabaseFileNameMin)

	if(os.path.exists(HFconceptNeuronListPathName)):
		HFNLPpy_ConnectionMatrixBasic.initialiseNeuronNameList(HFconnectionGraphObject, True)
		
	if(os.path.exists(matrixDatabaseFileNameMinFilePath) and os.path.exists(HFconceptNeuronListPathName)):
		print("initialiseMatrixDatabase warning: matrixDatabaseAlreadyInitialised; using existing HFconnectionGraphMatrixMin/HFconnectionGraphMatrixMax and conceptList files")
	elif(os.path.exists(matrixDatabaseFileNameMinFilePath) or os.path.exists(HFconceptNeuronListPathName)):
		printe("matrixDatabase incorrectly initialised os.path.exists(matrixDatabaseFileNameMinFilePath) XOR os.path.exists(HFconceptNeuronListPathName)")
	
	if(os.path.exists(matrixDatabaseFileNameMinFilePath)):
		HFconnectionGraphObject.HFconnectionGraphMatrixMin = loadMatrixDatabaseFile(HFconnectionGraphObject, matrixDatabaseFileNameMin)
		HFconnectionGraphObject.HFconnectionGraphMatrixMax = loadMatrixDatabaseFile(HFconnectionGraphObject, matrixDatabaseFileNameMax)

def finaliseMatrixDatabaseSentence(HFconnectionGraphObject, sentenceConceptNodeList):
	#save all tensors to drive and clear all tensors from RAM
	saveMatrixDatabaseFile(matrixDatabaseFileNameMin, HFconnectionGraphObject.HFconnectionGraphMatrixMin)
	saveMatrixDatabaseFile(matrixDatabaseFileNameMax, HFconnectionGraphObject.HFconnectionGraphMatrixMax)
	#del HFconnectionGraphObject.HFconnectionGraphMatrixMin	#deletion not required - can retain HFconnectionGraphMatrixMin/Max in RAM across sentences [assuming sufficient initialisation padding for new concepts]
	#del HFconnectionGraphObject.HFconnectionGraphMatrixMax #deletion not required - can retain HFconnectionGraphMatrixMin/Max in RAM across sentences [assuming sufficient initialisation padding for new concepts]
	for conceptNode in sentenceConceptNodeList:
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		saveMatrixDatabaseFile(neuronID, HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID])
		#del HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID]	#TODO: resolve this issue - this inadvertently deletes other matrices within the list (at different indices)
	HFNLPpy_ConnectionMatrixBasic.writeHFConceptListBasic(HFconnectionGraphObject.neuronNamelist)
	
def readGraphFromCsv(filePath):
	connections = []
	with open(filePath, 'r') as f:
		reader = csv.reader(f)
		for row in (reader):
			connections.append(row)
	#print("connections = ", connections)
	connections = [[int(value) for value in row] for row in connections]
	graph = pt.tensor(connections, dtype=HFconnectionsMatrixAlgorithmType)
	
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
		graph = graph.view(graph.shape[2], -1)	# Flatten the ND tensor into a 2D tensor
	else:
		printe("HFNLPpy_MatrixDatabase:writeGraphToCsv error: HFconnectionMatrixAlgorithmSplitDatabase currently requires algorithmMatrixTensorDim=4 such that the file i/o code can be simplified")
		
	connections = graph.numpy()
	with open(filePath, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(connections)

