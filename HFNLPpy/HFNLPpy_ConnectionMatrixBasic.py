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
from torch_geometric.data import Data

from HFNLPpy_globalDefs import *
from ANNtf2_loadDataset import datasetFolderRelative
import HFNLPpy_ConnectionMatrixOperations

epsilon = 1e-8  # Small epsilon value

def addContextConnectionsToGraphBasic(HFconnectionGraph, contextConnectionVector):
	if(HFconnectionMatrixBasicBool):
		HFconnectionGraph = pt.logical_and(HFconnectionGraph, contextConnectionVector)
	else:
		HFconnectionGraph += contextConnectionVector
	return HFconnectionGraph

def padContextConnectionVector(HFconnectionGraphObject, contextConnectionVector):
	conceptsSize = contextConnectionVector.shape[0]
	spareConceptsSize = HFconnectionGraphObject.connectionMatrixMaxConcepts-conceptsSize
	contextConnectionVectorPadded = F.pad(contextConnectionVector, (0, spareConceptsSize), mode='constant', value=0)
	#contextConnectionVectorPadded = pt.nn.ZeroPad1d(spareConceptsSize)(contextConnectionVector)	#requires later version of pytorch
	return contextConnectionVectorPadded

def extendConceptNeuronContextVector(HFconnectionGraphObject, conceptNeuronContextVector):
	conceptNeuronContextVectorExtended = pt.unsqueeze(conceptNeuronContextVector, dim=0)
	conceptNeuronContextVectorExtended = conceptNeuronContextVectorExtended.repeat(HFconnectionGraphObject.connectionMatrixMaxConcepts, 1)	#len(HFconnectionGraphObject.neuronNamelist)	
	return conceptNeuronContextVectorExtended
	
def createDiagonalMatrix(squareMatrix, width):
	diagonalMatrix = pt.tril(squareMatrix, diagonal=0) - pt.tril(squareMatrix, diagonal=-width)
	#diagonalMatrix = pt.tril(squareMatrix, diagonal=0) * torch.triu(squareMatrix, diagonal=width)
	return diagonalMatrix
	
def initialiseHFconnectionMatrixBasic(HFconnectionGraphObject, dendriticBranchIndex="", contextSizeIndex=""):
	if(HFreadSavedConnectionsMatrixBasic):
		HFconnectionGraph = readHFconnectionMatrixBasic(HFconnectionGraphObject, dendriticBranchIndex, contextSizeIndex)
	else:
		print("HFconnectionGraphObject.connectionMatrixMaxConcepts = ", HFconnectionGraphObject.connectionMatrixMaxConcepts)
		print("HFconnectionsMatrixBasicType = ", HFconnectionsMatrixBasicType)
		HFconnectionGraph = pt.zeros([HFconnectionGraphObject.connectionMatrixMaxConcepts, HFconnectionGraphObject.connectionMatrixMaxConcepts], dtype=HFconnectionsMatrixBasicType)
	if(HFconnectionMatrixBasicGPU):
		HFconnectionGraph = HFconnectionGraph.to(HFNLPpy_ConnectionMatrixOperations.device)
	return HFconnectionGraph

def readHFconnectionMatrixBasic(HFconnectionGraphObject):
	HFconnectionMatrixPathName = generateHFconnectionMatrixBasicFileName()
	HFconnectionGraph = HFNLPpy_ConnectionMatrixOperations.readGraphFromCsv(HFconnectionMatrixPathName)
	HFconnectionGraph = padHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraph)
	return HFconnectionGraph

def squeezeHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraph):
	networkSize = HFNLPpy_ConnectionMatrixOperations.getNetworkSize(HFconnectionGraphObject)
	maxConceptsSize = HFconnectionGraphSourceIndex.shape[-1]
	HFconnectionGraphSourceIndexSqueezed = HFconnectionGraph[0:networkSize, 0:networkSize]	#squeeze to length of database
	return HFconnectionGraphSourceIndexSqueezed
	
def padHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraphSourceIndex):
	networkSize = HFNLPpy_ConnectionMatrixOperations.getNetworkSize(HFconnectionGraphObject)
	conceptsSize = HFconnectionGraphSourceIndex.shape[-1]
	assert (networkSize == conceptsSize)
	#HFconnectionGraphSourceIndexPadded = F.pad(HFconnectionGraph, (0, spareConceptsSize, 0, spareConceptsSize), mode='constant', value=0)	#pad the the end of the last dimensions of the tensor (note that F.pad padding dimensions are reversed ordered)
	HFconnectionGraph = pt.nn.ZeroPad2d((0, spareConceptsSize, 0, spareConceptsSize))(HFconnectionGraph)
	return HFconnectionGraphSourceIndexPadded

def createContextVectorTensorBasic(HFconnectionGraphObject):
	contextVectorLength = HFconnectionGraphObject.connectionMatrixMaxConcepts
	contextConnectionVector = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixBasicType)
	return contextConnectionVector
	
def writeHFconnectionMatrixBasic(HFconnectionGraphObject, HFconnectionGraph, neuronNamelist):
	HFconnectionGraph = squeezeHFconnectionGraph(HFconnectionGraphObject, HFconnectionGraph)
	HFconnectionMatrixPathName = generateHFconnectionMatrixBasicFileName()
	HFNLPpy_ConnectionMatrixOperations.writeGraphToCsv(HFconnectionGraph, HFconnectionMatrixPathName)

def generateHFconnectionMatrixBasicFileName():
	HFconnectionMatrixPathName = datasetFolderRelative + "/" + HFconnectionMatrixBasicFileName + HFconnectionMatrixBasicExtensionName
	return HFconnectionMatrixPathName
		
def writeHFconnectionMatrixBasicWrapper(HFconnectionGraphObject):
	if(HFwriteSavedConnectionsMatrixBasic):
		writeHFconnectionMatrixBasic(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphBasic)
	if(HFwriteSavedConceptList):
		HFNLPpy_ConnectionMatrixOperations.writeHFConceptList(HFconnectionGraphObject.neuronNamelist)

def normaliseBatchedTensor(HFconnectionGraph):
	HFconnectionGraph = HFconnectionGraph.float()
	if(HFconnectionMatrixBasicBool):	#OLD: if(not weightStore)
		HFconnectionGraphNormalised = HFconnectionGraph
	else:
		#calculate a temporary normalised version of the HFconnectionGraph	#CHECKTHIS
		if(HFconnectionMatrixBasicNormalise=="softmax"):
			HFconnectionGraphNormalised = pt.nn.functional.softmax(HFconnectionGraph, dim=1)
		elif(HFconnectionMatrixBasicNormalise=="tanh"):
			HFconnectionGraphNormalised = HFconnectionGraph.tanh()
		elif(HFconnectionMatrixBasicNormalise=="xsech"):
			HFconnectionGraphNormalised = HFconnectionGraph * 1/HFconnectionGraph.cosh()	
		elif(HFconnectionMatrixBasicNormalise=="linear"):
			min_vals, _ = pt.min(HFconnectionGraph, dim=-1, keepdim=True)
			max_vals, _ = pt.max(HFconnectionGraph, dim=-1, keepdim=True)
			HFconnectionGraphNormalised = (HFconnectionGraph - min_vals) / (max_vals - min_vals + epsilon)
	return HFconnectionGraphNormalised
	
