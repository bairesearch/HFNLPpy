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

def retrieveSimilarConceptsBagOfWords(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject=None):
	for conceptNeuron in connectionTargetNeuronSet:
		if(linkSimilarConceptNodesBagOfWordsContextual):
			conceptNeuronContextVector = createContextVector(wSource, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, bagOfWordsDistanceMax, linkSimilarConceptNodesBagOfWordsWeightRetrieval, False)
			connectionTargetNeuronSetExtended, _, _ = connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, linkSimilarConceptNodesBagOfWordsTopK, False)
		else:
			conceptNeuronID = HFconnectionGraphObject.neuronIDdict[conceptNeuron.nodeName]
			conceptNeuronContextVector = HFconnectionGraphObject.HFconnectionGraphNormalised[conceptNeuronID]
			connectionTargetNeuronSetExtended, _, _ = connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, linkSimilarConceptNodesBagOfWordsTopK, False)

	return connectionTargetNeuronSetExtended

def connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraph, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4):
	contextConnectionVector = createContextVectorWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4)	#len(HFconnectionGraphObject.neuronNamelist)
	connectionTargetNeuronSet, connectionStrength, connectionIndex = connectionMatrixCalculateConnectionTargetSet(HFconnectionGraph, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, contextConnectionVector, matrixPropagateTopKconceptNodes, matrixTensorDim4)
	return connectionTargetNeuronSet, connectionStrength, connectionIndex

def connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphNormalised, neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, k, matrixTensorDim4):
	connectionTargetNeuronList = []
	conceptNeuronContextVectorExtended = HFNLPpy_ConnectionMatrixBasic.extendConceptNeuronContextVector(conceptNeuronContextVector, matrixTensorDim4)
	if(matrixTensorDim4):
		if(not algorithmMatrixSANI):
			HFconnectionGraphNormalised = HFconnectionGraphNormalised[:, 0:conceptNeuronContextVectorExtended.shape[1], :, :]	#only compare selected contextIndices
	#print("conceptNeuronContextVectorExtendedsum() = ", conceptNeuronContextVectorExtended.sum())
	#print("HFconnectionGraphNormalised.sum() = ", HFconnectionGraphNormalised.sum())
	mask = HFconnectionGraphNormalised * conceptNeuronContextVectorExtended
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
		
def createContextVectorWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeIndex, contextSizeMaxSource, weightStore, bidirectionalContext, matrixTensorDim4):
	if(algorithmMatrixSANI):
		if(matrixTensorDim4):
			return createContextVectorsSANIPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeMaxSource, weightStore)
		else:
			return createContextVectorSANIPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeIndex, weightStore)
	else:
		if(matrixTensorDim4):
			return createContextVectorsPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeMaxSource, weightStore, bidirectionalContext)
		else:
			return createContextVectorPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeIndex, weightStore, bidirectionalContext)
		
def createContextVectorsSANIPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, sequentialSegmentMax, weightStore):
	contextConnectionVectorList = []
	for sequentialSegmentIndex in range(sequentialSegmentMax):	#numberOfBranchSequentialSegments
		contextConnectionVector = createContextVectorSANIPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, sequentialSegmentIndex, weightStore)
		contextConnectionVectorList.append(contextConnectionVector)
	contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector
	
def createContextVectorSANIPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, sequentialSegmentIndex, weightStore):
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
	if(validSequentialSegment):
		contextConnectionVector = createContextVectorSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, w2Min, w2Max, weightStore)	#len(HFconnectionGraphObject.neuronNamelist)
		contextConnectionVector = HFNLPpy_ConnectionMatrixBasic.padContextConnectionVector(contextConnectionVector)
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
			
def createContextVectorsPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeMaxSource, weightStore, bidirectionalContext):
	contextConnectionVectorList = []
	for contextSizeIndex in range(contextSizeMaxSource):
		contextConnectionVector = createContextVectorPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, contextSizeIndex, weightStore, bidirectionalContext)	#len(HFconnectionGraphObject.neuronNamelist)
		contextConnectionVectorList.append(contextConnectionVector)
	contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector
	
def createContextVectorPadded(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeIndex, weightStore, bidirectionalContext):
	contextConnectionVector = createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeIndex, weightStore, bidirectionalContext)
	padContextConnectionVector = HFNLPpy_ConnectionMatrixBasic.padContextConnectionVector(contextConnectionVector)
	return padContextConnectionVector
	
def createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, contextSizeIndex, weightStore, bidirectionalContext):
	contextConnectionVector = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixType)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(bidirectionalContext or (w2 < w1)):
				if(w1-w2 <= getContextSize(contextSizeIndex)):
					conceptNodeContext = sentenceConceptNodeList[w2]
					neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
					contextConnectionVector[neuronIDcontext] = calculateContextVectorValue(weightStore, w1, w2)
	return contextConnectionVector

def createContextVectorSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextVectorLength, w2Min, w2Max, weightStore):
	contextConnectionVector = pt.zeros(contextVectorLength, dtype=HFconnectionsMatrixType)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(w2 >= w2Min and w2 < w2Max):
				conceptNodeContext = sentenceConceptNodeList[w2]
				neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
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

def normaliseBatchedTensor(HFconnectionGraph):
	if(useHFconnectionMatrixBasicBool):	#OLD: if(not weightStore)
		HFconnectionGraphNormalised = HFconnectionGraphFloat
	else:
		HFconnectionGraphFloat = (HFconnectionGraph).float()
		#calculate a temporary normalised version of the HFconnectionGraph	#CHECKTHIS
		if(useHFconnectionMatrixNormaliseSoftmax):
			HFconnectionGraphNormalised = pt.nn.functional.softmax(HFconnectionGraphFloat, dim=1)
		else:
			if(useHFconnectionMatrixBasicSparse):
				printe("normaliseBatchedTensor does not yet support useHFconnectionMatrixBasicSparse")
			else:
				min_vals, _ = pt.min(HFconnectionGraphFloat, dim=-1, keepdim=True)
				max_vals, _ = pt.max(HFconnectionGraphFloat, dim=-1, keepdim=True)
				epsilon = 1e-8  # Small epsilon value
				HFconnectionGraphNormalised = (HFconnectionGraphFloat - min_vals) / (max_vals - min_vals + epsilon)
	return HFconnectionGraphNormalised

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
	
