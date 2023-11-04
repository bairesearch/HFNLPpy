"""HFNLPpy_MatrixOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Matrix Operations

"""

import numpy as np
from HFNLPpy_globalDefs import *

import torch as pt
from HFNLPpy_MatrixGlobalDefs import *
import torch.nn.functional as F
import HFNLPpy_ConnectionMatrixBasic
import HFNLPpy_hopfieldOperations

def createConnectionGraphMatrixHolder():
	if(algorithmMatrixTensorDim==4):
		HFconnectionGraphMatrix = None
	else:
		if(algorithmMatrixTensorDim==3):
			HFconnectionGraphMatrix = [None for _ in range(numberOfIndependentDendriticBranches)]
		else:
			secondDataIndexMax = getSecondDataIndexMax()
			HFconnectionGraphMatrix = [[None for _ in range(secondDataIndexMax)] for _ in range(numberOfIndependentDendriticBranches)]	#[[None]*contextSizeMax]*numberOfIndependentDendriticBranches	#[[[None]*contextSizeMax] for i in range(numberOfIndependentDendriticBranches)]
	return HFconnectionGraphMatrix
	
def retrieveSimilarConceptsBagOfWords(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject=None):
	for conceptNeuron in connectionTargetNeuronSet:
		if(linkSimilarConceptNodesBagOfWordsContextual):
			conceptNeuronContextVector = createContextVector(wSource, sentenceConceptNodeList, HFconnectionGraphObject, bagOfWordsDistanceMax, linkSimilarConceptNodesBagOfWordsWeightRetrieval, False)
			connectionTargetNeuronSetExtended, _, _ = connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, linkSimilarConceptNodesBagOfWordsTopK, False)
		else:
			conceptNeuronID = HFconnectionGraphObject.neuronIDdict[conceptNeuron.nodeName]
			conceptNeuronContextVector = HFconnectionGraphObject.HFconnectionGraphNormalised[conceptNeuronID]
			connectionTargetNeuronSetExtended, _, _ = connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, linkSimilarConceptNodesBagOfWordsTopK, False)
	return connectionTargetNeuronSetExtended

def connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4):
	conceptNeuronContextVector = createContextVectorWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4)	#len(HFconnectionGraphObject.neuronNamelist)
	HFconnectionGraph = HFNLPpy_ConnectionMatrixBasic.getConnectionGraph(HFconnectionGraphObject, conceptNeuronContextVector, dendriticBranchIndex, secondDataIndex, matrixTensorDim4)
	if(useHFconnectionMatrixBasicSplit):
		conceptNeuronContextVector = HFNLPpy_ConnectionMatrixBasic.convertContextVectorSparseListToDense(conceptNeuronContextVector, matrixTensorDim4)
	connectionTargetNeuronSet, connectionStrength, connectionIndex = connectionMatrixCalculateConnectionTargetSet(HFconnectionGraph, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, matrixPropagateTopKconceptNodes, matrixTensorDim4)
	return connectionTargetNeuronSet, connectionStrength, connectionIndex

def connectionMatrixCalculateConnectionTargetSet(HFconnectionGraph, neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, k, matrixTensorDim4):
	connectionTargetNeuronList = []
	conceptNeuronContextVectorExtended = HFNLPpy_ConnectionMatrixBasic.extendConceptNeuronContextVector(conceptNeuronContextVector, matrixTensorDim4)
	#print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	#print("conceptNeuronContextVectorExtended.shape = ", conceptNeuronContextVectorExtended.shape)
	if(matrixTensorDim4):
		if(not algorithmMatrixSANI):
			HFconnectionGraph = HFconnectionGraph[:, 0:conceptNeuronContextVectorExtended.shape[1], :, :]	#only compare selected contextIndices
	#print("conceptNeuronContextVectorExtendedsum() = ", conceptNeuronContextVectorExtended.sum())
	#print("HFconnectionGraph.sum() = ", HFconnectionGraph.sum())
	mask = HFconnectionGraph * conceptNeuronContextVectorExtended
	array = mask
	
	neuronIndexList = []
	if(matrixTensorDim4):
		if(algorithmMatrixSANImethodAddActivationAcrossSegments):
			array = pt.sum(array, dim=1, keepdim=True)
		elif(algorithmMatrixSANImethodEnforceSequentialActivationAcrossSegments):
			printe("connectionMatrixCalculateConnectionTargetSet error: algorithmMatrixSANImethodEnforceSequentialActivationAcrossSegments not yet coded")
				
		arraySummedTopK = performSumTopK(array, matrixPropagateTopKconceptNodes, 3)
		arraySummedTopKindices = arraySummedTopK.indices
		arraySummedTopK = performSumTopK(arraySummedTopK.values, matrixPropagateTopKsecondIndex, 2)
		if(simulatedDendriticBranches):
			arraySummedTopKindices = arraySummedTopKindices.squeeze(-1).gather(dim=1, index=arraySummedTopK.indices)
		else:
			arraySummedTopKindices = arraySummedTopKindices[:, arraySummedTopK.indices]
			
		arraySummedTopK = performSumTopK(arraySummedTopK.values, matrixPropagateTopKdendriticBranches, 1)
		for i in range(len(arraySummedTopK.values)):
			value = arraySummedTopK.values[i]
			if(value > HFconnectionMatrixMinValue):
				neuronIndexList.append(arraySummedTopKindices[arraySummedTopK.indices[i]])
	else:
		arraySummedTopK = performSumTopK(array, k, 1)
		for i in range(len(arraySummedTopK.values)):
			value = arraySummedTopK.values[i]
			if(value > HFconnectionMatrixMinValue):
				neuronIndexList.append(arraySummedTopK.indices[i])
	
	for i in neuronIndexList:
		conceptName = neuronNamelist[i]
		conceptNeuron, conceptInDict = HFNLPpy_hopfieldOperations.convertLemmaToConcept(networkConceptNodeDict, conceptName)
		if(conceptInDict):
			connectionTargetNeuronList.append(conceptNeuron)
	connectionTargetNeuronSet = set(connectionTargetNeuronList)	
	
	if(matrixTensorDim4):
		connectionIndex = arraySummedTopK.indices[0]	#only valid for matrixPropagateTopKdendriticBranches=1
		connectionStrength = pt.sum(arraySummedTopK.values)
	else:
		connectionIndex = None
		connectionStrength = pt.sum(arraySummedTopK.values)
		
	return connectionTargetNeuronSet, connectionStrength, connectionIndex

def calculateSequentialSegmentActivation(connectionStrength):
	if(algorithmMatrixSANImethodEnforceSequentialActivationAcrossSegments):
		printe("calculateSequentialSegmentActivation error: algorithmMatrixSANImethodEnforceSequentialActivationAcrossSegments not coded")
	else:
		activationValue = connectionStrength
	return activationValue

def performSumTopK(array, k, dim):
	arraySummed = pt.sum(array, dim=dim)
	arraySummedTopK = pt.topk(arraySummed, k, dim=dim-1)
	return arraySummedTopK
		
def createContextVectorWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, contextSizeMaxSource, weightStore, bidirectionalContext, matrixTensorDim4):
	if(algorithmMatrixSANI):
		if(matrixTensorDim4):
			contextConnectionVector = createContextVectorsSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeMaxSource, weightStore)
		else:
			contextConnectionVector = createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore)
	else:
		if(matrixTensorDim4):
			contextConnectionVector = createContextVectors(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeMaxSource, weightStore, bidirectionalContext)
		else:
			contextConnectionVector = createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext)
	if(not useHFconnectionMatrixBasicSplit):
		if(HFconnectionMatrixGPU):
			contextConnectionVector = contextConnectionVector.to(HFNLPpy_ConnectionMatrixBasic.device)
			
	return contextConnectionVector
	
def createContextVectorsSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentMax, weightStore):
	contextConnectionVectorList = []
	for sequentialSegmentIndex in range(sequentialSegmentMax):	#numberOfBranchSequentialSegments
		contextConnectionVector = createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore)
		contextConnectionVectorList.append(contextConnectionVector)
	if(useHFconnectionMatrixBasicSplit):
		contextConnectionVector = contextConnectionVectorList
	else:
		contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector
	
def createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore):
	if(sequentialSegmentContextEncodingRelativeExponential):
		expRange = createExponentialRange(0, w1, numberOfBranchSequentialSegments)
	contextSequenceLength = w1
	validSequentialSegment = True
	if(sequentialSegmentContextEncodingAbsoluteLinear):
		contextSequenceSegmentLength = w1/numberOfBranchSequentialSegments
		w2Min = w1-(numberOfBranchSequentialSegments*sequentialSegmentContextEncodingAbsoluteLinearSize)+(sequentialSegmentIndex*sequentialSegmentContextEncodingAbsoluteLinearSize)
		w2Max = w2Min+(sequentialSegmentIndex*sequentialSegmentContextEncodingAbsoluteLinearSize)
		if(w2Min < 0):
			validSequentialSegment = False
	elif(sequentialSegmentContextEncodingRelativeLinear):
		contextSequenceSegmentLength = w1/numberOfBranchSequentialSegments
		w2Min = sequentialSegmentIndex*contextSequenceSegmentLength
		w2Max = w2Min + contextSequenceSegmentLength
	elif(sequentialSegmentContextEncodingRelativeExponential):
		w2Min = sum(expRange[0:sequentialSegmentIndex+1])
		w2Max = sum(expRange[0:sequentialSegmentIndex+2])
	w2Min = int(w2Min)
	w2Max = int(w2Max)
	if(validSequentialSegment):
		contextConnectionVector = createContextVectorSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, w2Min, w2Max, weightStore)	#len(HFconnectionGraphObject.neuronNamelist)
	else:
		emptyTensor = pt.zeros(HFconnectionMatrixBasicMaxConcepts)
		contextConnectionVector = emptyTensor
	return contextConnectionVector
	
def createExponentialRange(minVal, maxVal, size):
	#rate = s / (maxVal - minVal)
	#expRange = [random.expovariate(rate) + minVal for _ in range(s)]
	expRange = []
	val_range = maxVal - minVal
	for i in range(s):
		rate = 1 / (val_range - (i * val_range / s))
		exponential_factor = 1 + random.expovariate(rate)
		if i == 0:
			value = minVal
		else:
			value = exponential_values[i - 1] * exponential_factor
		value = max(minVal, min(maxVal, value))
		exponential_values.append(value)
	return expRange
			
def createContextVectors(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeMaxSource, weightStore, bidirectionalContext):
	contextConnectionVectorList = []
	for contextSizeIndex in range(contextSizeMaxSource):
		contextConnectionVector = createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext)	#len(HFconnectionGraphObject.neuronNamelist)
		contextConnectionVectorList.append(contextConnectionVector)
	if(useHFconnectionMatrixBasicSplit):
		contextConnectionVector = contextConnectionVectorList
	else:
		contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector
	
def createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext):
	if(bidirectionalContext):
		contextLength = len(sentenceConceptNodeList)	#contextSizeMax
	else:
		contextLength = w1	#len(sentenceConceptNodeList) [too large]	#int(w2Max-w2Min) [not possible as will vary across secondDataIndex]	#contextSizeMax [too large]
	contextConnectionVector = HFNLPpy_ConnectionMatrixBasic.createContextVectorTensor(contextLength)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(bidirectionalContext or (w2 < w1)):
				if(w1-w2 <= getContextSize(contextSizeIndex)):
					conceptNodeContext = sentenceConceptNodeList[w2]
					neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
					if(HFcontextVectorSparse):
						if(bidirectionalContext):
							contextConnectionVectorIndex = w2
						else:
							contextConnectionVectorIndex = w2	#w2-w2Min
						contextConnectionVector.values[contextConnectionVectorIndex] = calculateContextVectorValue(weightStore, w1, w2)
						contextConnectionVector.indices[contextConnectionVectorIndex] = neuronIDcontext
					else:
						contextConnectionVector[neuronIDcontext] = calculateContextVectorValue(weightStore, w1, w2)
	return contextConnectionVector

def createContextVectorSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, w2Min, w2Max, weightStore):
	contextLength = w1	#len(sentenceConceptNodeList) [too large]	#int(w2Max-w2Min) [not possible as will vary across secondDataIndex]	#contextSizeMax [too large]
	contextConnectionVector = HFNLPpy_ConnectionMatrixBasic.createContextVectorTensor(contextLength)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(w2 >= w2Min and w2 < w2Max):
				conceptNodeContext = sentenceConceptNodeList[w2]
				neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
				if(HFcontextVectorSparse):
					contextConnectionVectorIndex = w2	#w2-w2Min
					contextConnectionVector.values[contextConnectionVectorIndex] = calculateContextVectorValue(weightStore, w1, w2)
					contextConnectionVector.indices[contextConnectionVectorIndex] = neuronIDcontext
				else:
					contextConnectionVector[neuronIDcontext] = calculateContextVectorValue(weightStore, w1, w2)
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

def getSecondDataIndexMax(getContextSizeSource=False, wSource=None):
	if(algorithmMatrixSANI):
		secondDataIndexMax = numberOfBranchSequentialSegments
	else:
		if(getContextSizeSource):
			contextSizeMaxSource = min(contextSizeMax, wSource)
			secondDataIndexMax = contextSizeMaxSource
		else:
			secondDataIndexMax = contextSizeMax
	return secondDataIndexMax


def updateDendriticBranchClosestValue(foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, closestDendriticBranchIndex, targetSet, connectionStrength, dendriticBranchIndex, threshold=False, connectionStrengthNormalised=None):
	if(connectionStrength > closestConnectionStrength):
		if((not threshold) or (connectionStrengthNormalised > simulatedDendriticBranchesMinMatchStrength)):
			foundClosest = True
			if(dendriticBranchClosestTargetSet is not None):
				dendriticBranchClosestTargetSet = targetSet
			if(closestConnectionStrength is not None):
				closestConnectionStrength = connectionStrength
			if(closestDendriticBranchIndex is not None):
				closestDendriticBranchIndex = dendriticBranchIndex
	return foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, closestDendriticBranchIndex
	
