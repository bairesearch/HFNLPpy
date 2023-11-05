"""HFNLPpy_ConceptsMatrixOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Matrix Operations Basic

"""

import numpy as np
import torch as pt
from HFNLPpy_globalDefs import *

import torch.nn.functional as F
import HFNLPpy_ConnectionMatrixBasic
import HFNLPpy_hopfieldOperations

def retrieveSimilarConceptsBagOfWords(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject=None):
	for conceptNeuron in connectionTargetNeuronSet:
		if(linkSimilarConceptNodesBagOfWordsContextual):
			conceptNeuronContextVector = createContextVectorBasic(wSource, sentenceConceptNodeList, HFconnectionGraphObject, linkSimilarConceptNodesBagOfWordsDistanceMax, linkSimilarConceptNodesBagOfWordsWeightRetrieval, False)
			connectionTargetNeuronSetExtended, _, _ = connectionMatrixCalculateConnectionTargetSetBasic(HFconnectionGraphObject.HFconnectionGraphBasicNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, linkSimilarConceptNodesBagOfWordsTopK)
		else:
			conceptNeuronID = HFconnectionGraphObject.neuronIDdict[conceptNeuron.nodeName]
			conceptNeuronContextVector = HFconnectionGraphObject.HFconnectionGraphNormalised[conceptNeuronID]
			connectionTargetNeuronSetExtended, _, _ = connectionMatrixCalculateConnectionTargetSetBasic(HFconnectionGraphObject.HFconnectionGraphBasicNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, linkSimilarConceptNodesBagOfWordsTopK)
	return connectionTargetNeuronSetExtended

def connectionMatrixCalculateConnectionTargetSetBasic(HFconnectionGraph, neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, k):
	connectionTargetNeuronList = []
	conceptNeuronContextVectorExtended = HFNLPpy_ConnectionMatrixBasic.extendConceptNeuronContextVector(conceptNeuronContextVector)
	
	mask = HFconnectionGraph * conceptNeuronContextVectorExtended
	array = mask
	
	neuronIndexList = []
	arraySummedTopK = performSumTopK(array, k, 1)
	for i in range(len(arraySummedTopK.values)):
		value = arraySummedTopK.values[i]
		if(value > HFconnectionMatrixBasicMinValue):
			neuronIndexList.append(arraySummedTopK.indices[i])
	
	for i in neuronIndexList:
		conceptName = neuronNamelist[i]
		conceptNeuron, conceptInDict = HFNLPpy_hopfieldOperations.convertLemmaToConcept(networkConceptNodeDict, conceptName)
		if(conceptInDict):
			connectionTargetNeuronList.append(conceptNeuron)
	connectionTargetNeuronSet = set(connectionTargetNeuronList)	
	
	#not used;
	connectionIndex = None
	connectionStrength = pt.sum(arraySummedTopK.values)
		
	return connectionTargetNeuronSet, connectionStrength, connectionIndex

def performSumTopK(array, k, dim):
	arraySummed = pt.sum(array, dim=dim)
	arraySummedTopK = pt.topk(arraySummed, k, dim=dim-1)
	return arraySummedTopK

def createContextVectorBasic(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext):
	contextConnectionVector = HFNLPpy_ConnectionMatrixBasic.createContextVectorTensorBasic()
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(bidirectionalContext or (w2 < w1)):
				if(w1-w2 <= getContextSize(contextSizeIndex)):
					conceptNodeContext = sentenceConceptNodeList[w2]
					neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
					contextConnectionVector[neuronIDcontext] = calculateContextVectorValue(weightStore, w1, w2)
	if(HFconnectionMatrixBasicGPU):
		contextConnectionVector = contextConnectionVector.to(HFNLPpy_ConnectionMatrixBasic.device)
	return contextConnectionVector

def calculateContextVectorValue(weightStore, w1, w2):
	if(weightStore):
		weight = 1.0/(abs(w1 - w2))
		contextVectorValue = weight
	else:
		if(useHFconnectionMatrixBasicBool):
			contextVectorValue = True
		else:
			contextVectorValue = 1.0
	return contextVectorValue

def getContextSize(contextSizeIndex):
	contextSize = contextSizeIndex+1	#min contextSize = 1
	return contextSize
	
