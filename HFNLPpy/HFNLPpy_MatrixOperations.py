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
import HFNLPpy_ConnectionMatrixAlgorithm
import HFNLPpy_hopfieldOperations
import HFNLPpy_ConnectionMatrixOperations

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

def connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4, k=matrixPropagateTopKconceptNodes):
	conceptNeuronContextVector = createContextVectorWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4)	#len(HFconnectionGraphObject.neuronNamelist)
	validContextVector, connectionTargetNeuronSet, connectionStrength, connectionIndex = isValidContextVector(conceptNeuronContextVector, matrixTensorDim4)
	if(validContextVector):
		HFconnectionGraph = HFNLPpy_ConnectionMatrixAlgorithm.getConnectionGraph(HFconnectionGraphObject, conceptNeuronContextVector, dendriticBranchIndex, secondDataIndex, matrixTensorDim4)
		if(HFconnectionMatrixAlgorithmSplit):
			conceptNeuronContextVector = HFNLPpy_ConnectionMatrixAlgorithm.convertContextVectorSparseListToDense(conceptNeuronContextVector, matrixTensorDim4)
		connectionTargetNeuronSet, connectionStrength, connectionIndex = connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject, HFconnectionGraph, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, k, matrixTensorDim4)
	return connectionTargetNeuronSet, connectionStrength, connectionIndex

def isValidContextVector(conceptNeuronContextVector, matrixTensorDim4):
	validContextVector = True
	connectionTargetNeuronSet = set()
	connectionStrength = 0
	connectionIndex = -1
	if(HFconnectionMatrixAlgorithmSplit):
		if(matrixTensorDim4):
			if(len(conceptNeuronContextVector[0].indices) == 0):
				validContextVector = False
		else:
			if(len(conceptNeuronContextVector) == 0):
				validContextVector = False
	return validContextVector, connectionTargetNeuronSet, connectionStrength, connectionIndex

def SANImethodAddActivationAcrossSegments(array):
	array = pt.sum(array, dim=1, keepdim=True)
	return array
	
def SANImethodSupportSequentialActivationAcrossSegments(HFconnectionGraph, conceptNeuronContextVector):
	activations = pt.zeros([HFconnectionGraph.shape[0], HFconnectionGraph.shape[1], HFconnectionGraph.shape[2], HFconnectionGraph.shape[3]]).to(HFNLPpy_ConnectionMatrixOperations.device)
	conceptNeuronContextVectorSegmentPrevious = pt.zeros([HFconnectionGraph.shape[0], HFconnectionGraph.shape[2], HFconnectionGraph.shape[3]]).to(HFNLPpy_ConnectionMatrixOperations.device)
	for sequentialSegmentIndex in range(numberOfBranchSequentialSegments):
		#calculate current segment activations;
		HFconnectionGraphSegment = HFconnectionGraph[:, sequentialSegmentIndex]
		conceptNeuronContextVectorSegment = conceptNeuronContextVector[:, sequentialSegmentIndex] + conceptNeuronContextVectorSegmentPrevious
		conceptNeuronContextVectorSegmentCumulative = conceptNeuronContextVectorSegment+ conceptNeuronContextVectorSegmentPrevious
		activationsSegment = HFconnectionGraphSegment * conceptNeuronContextVectorSegmentCumulative
		activations[:, sequentialSegmentIndex] = activationsSegment
		
		#only retain previously unactivated context if all activations for current segment are 0 (without this constraint the method is equivalent to addActivationAcrossSegments);
			#alternate method; only bias based on num of previous active states for unique dendritic inputs
		activationsSegmentOff =	(pt.sum(activationsSegment, dim=2) == 0).int().unsqueeze(dim=-1)
		conceptNeuronContextVectorSegmentPrevious = conceptNeuronContextVectorSegmentPrevious *	activationsSegmentOff	#broadcast last dim
	
		#retain previously unactivated context, to be fed into next segment;
		HFconnectionGraphSegmentOff = (HFconnectionGraphSegment == 0).int()
		conceptNeuronContextVectorSegmentInactive = conceptNeuronContextVectorSegment * HFconnectionGraphSegmentOff
		conceptNeuronContextVectorSegmentPrevious = conceptNeuronContextVectorSegmentPrevious + conceptNeuronContextVectorSegmentInactive
		
	activations = pt.sum(activations, dim=1, keepdim=True)
	return activations
	
def connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject, HFconnectionGraph, neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, k, matrixTensorDim4):
	connectionTargetNeuronList = []
	conceptNeuronContextVectorExtended = HFNLPpy_ConnectionMatrixAlgorithm.extendConceptNeuronContextVector(HFconnectionGraphObject, conceptNeuronContextVector, matrixTensorDim4)
	if(matrixTensorDim4):
		if(not algorithmMatrixSANI):
			HFconnectionGraph = HFconnectionGraph[:, 0:conceptNeuronContextVectorExtended.shape[1], :, :]	#only compare selected contextIndices
	#print("conceptNeuronContextVectorExtended.shape = ", conceptNeuronContextVectorExtended.shape)
	#print("conceptNeuronContextVectorExtended.sum() = ", conceptNeuronContextVectorExtended.sum())
	#print("HFconnectionGraph.shape = ", HFconnectionGraph.shape)
	#print("HFconnectionGraph.sum() = ", HFconnectionGraph.sum())
	activations = HFconnectionGraph * conceptNeuronContextVectorExtended
	array = activations
	
	neuronIndexList = []
	if(matrixTensorDim4):
		if(not matrixPropagateTopCommonSegmentPredictions):
			if(algorithmMatrixSANImethod=="addActivationAcrossSegments"):
				array = SANImethodAddActivationAcrossSegments(array)
			elif(algorithmMatrixSANImethod=="supportSequentialActivationAcrossSegments"):
				array = SANImethodSupportSequentialActivationAcrossSegments(HFconnectionGraph, conceptNeuronContextVectorExtended)
			elif(algorithmMatrixSANImethod=="enforceSequentialActivationAcrossSegments"):
				printe("connectionMatrixCalculateConnectionTargetSet error: algorithmMatrixSANImethod==enforceSequentialActivationAcrossSegments not coded")

		arraySummedTopK = performSumTopK(array, matrixPropagateTopKconceptNodes, 3)
		arraySummedTopKindices = arraySummedTopK.indices
		arraySummedTopKvalues = arraySummedTopK.values
		if(matrixPropagateTopCommonSegmentPredictions):
			arraySummedTopKindices, arraySummedTopKvalues = getTopCommonSegmentPredictions(arraySummedTopKindices, arraySummedTopKvalues)
		arraySummedTopK = performSumTopK(arraySummedTopKvalues, matrixPropagateTopKsecondIndex, 2)
		if(matrixPropagateTopKconceptNodes > 1):
			arraySummedTopKindices = multiIndex(arraySummedTopKindices, arraySummedTopK.indices, 3)		
		else:
			if(simulatedDendriticBranches):
				arraySummedTopKindices = arraySummedTopKindices.squeeze(-1).gather(dim=1, index=arraySummedTopK.indices)
			else:
				arraySummedTopKindices = arraySummedTopKindices[:, arraySummedTopK.indices]
		
		arraySummedTopK = performSumTopK(arraySummedTopK.values, matrixPropagateTopKdendriticBranches, 1)
		for i in range(len(arraySummedTopK.values)):
			value = arraySummedTopK.values[i]
			if(value > HFconnectionMatrixAlgorithmMinValue):
				neuronIndexList.append(arraySummedTopKindices[arraySummedTopK.indices[i]])
	else:
		arraySummedTopK = performSumTopK(array, k, 1)
		for i in range(len(arraySummedTopK.values)):
			value = arraySummedTopK.values[i]
			if(value > HFconnectionMatrixAlgorithmMinValue):
				neuronIndexList.append(arraySummedTopK.indices[i])
	
	for i in neuronIndexList:
		#print("i = ", i)
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

def multiIndex(data, index, numberDimensions):
	#precondition: index tensor dim = data tensor dim - 1, where each cell in index tensor is an index
	if(numberDimensions == 4):
		indexedData = data[torch.arange(data.size(0)).unsqueeze(1).unsqueeze(2), pt.arange(data.size(1)).unsqueeze(1), pt.arange(data.size(2)), index]
	elif(numberDimensions == 3):
		indexedData = data[pt.arange(data.size(0)).unsqueeze(1), pt.arange(data.size(1)), index]
	else:
		printe("multiIndex error: numberDimensions != 4  has not been coded")
	return indexedData

def getTopCommonSegmentPredictions(arraySummedTopKindices, arraySummedTopKvalues):
	if(matrixPropagateTopCommonSegmentPredictionsVectorised):
		return getTopCommonSegmentPredictionsVectorised(arraySummedTopKindices, arraySummedTopKvalues)
	else:
		return getTopCommonSegmentPredictionsStandard(arraySummedTopKindices, arraySummedTopKvalues)

def getTopCommonSegmentPredictionsVectorised(arraySummedTopKindices, arraySummedTopKvalues):
	#for every index in first sequential segment, calculate the total value for this index
	arraySummedTopKindex = pt.permute(arraySummedTopKindices, (2, 0, 1))	#only consider indices from first segment (0)
	arraySummedTopKindex = arraySummedTopKindex[:, :, 0]	#only consider indices from first segment (0)
	arraySummedTopKvaluesBatch = getTopCommonSegmentPredictionsBatch(arraySummedTopKindices, arraySummedTopKvalues, arraySummedTopKindex, 1)

	arraySummedTopKvalues = pt.permute(arraySummedTopKvaluesBatch, (1, 2, 0)) #shape = numberOfIndependentDendriticBranches, 1, matrixPropagateTopKconceptNodes
	arraySummedTopKindices = arraySummedTopKindices[:, 0]	#use indices from first segment (0)	#shape = numberOfIndependentDendriticBranches, matrixPropagateTopKconceptNodes
	arraySummedTopKindices = arraySummedTopKindices.unsqueeze(1)	#restore blank segment dimension
	
	return arraySummedTopKindices, arraySummedTopKvalues
	
def getTopCommonSegmentPredictionsStandard(arraySummedTopKindices, arraySummedTopKvalues):
	#print("arraySummedTopKvalues = ", arraySummedTopKvalues)
	#print("arraySummedTopKindices = ", arraySummedTopKindices)
	
	#arraySummedTopKindices/arraySummedTopKvalues shape = numberOfIndependentDendriticBranches, numberOfBranchSequentialSegments, matrixPropagateTopKconceptNodes
	#only take predictions that are common across all segments
	arraySummedTopKvaluesList = []
	for i in range(matrixPropagateTopKconceptNodes):
		#print("i = ", i)
		#for every index in first sequential segment, calculate the total value for this index
		arraySummedTopKindex = arraySummedTopKindices[:, 0, i]	#only consider indices from first segment (0)
		arraySummedTopKindexSelCommonValues = getTopCommonSegmentPredictionsBatch(arraySummedTopKindices, arraySummedTopKvalues, arraySummedTopKindex, 0)
		arraySummedTopKvaluesList.append(arraySummedTopKindexSelCommonValues)
		
	arraySummedTopKvalues = pt.stack(arraySummedTopKvaluesList, dim=2)	#shape = numberOfIndependentDendriticBranches, 1, matrixPropagateTopKconceptNodes
	arraySummedTopKindices = arraySummedTopKindices[:, 0]	#use indices from first segment (0)	#shape = numberOfIndependentDendriticBranches, matrixPropagateTopKconceptNodes
	arraySummedTopKindices = arraySummedTopKindices.unsqueeze(1)	#restore blank segment dimension
	
	#print("arraySummedTopKvalues = ", arraySummedTopKvalues)
	#print("arraySummedTopKindices = ", arraySummedTopKindices)
	
	return arraySummedTopKindices, arraySummedTopKvalues

def getTopCommonSegmentPredictionsBatch(arraySummedTopKindices, arraySummedTopKvalues, arraySummedTopKindex, batchDims):
	#arraySummedTopKindices/arraySummedTopKvalues shape = numberOfIndependentDendriticBranches, numberOfBranchSequentialSegments, matrixPropagateTopKconceptNodes
	if(batchDims==1):
		arraySummedTopKindices = arraySummedTopKindices.unsqueeze(0)	#.repeat(matrixPropagateTopKconceptNodes, 1, 1, 1) - not necessary with broadcasting
		arraySummedTopKvalues = arraySummedTopKvalues.unsqueeze(0)	#.repeat(matrixPropagateTopKconceptNodes, 1, 1, 1) - not necessary with broadcasting

	arraySummedTopKindex = arraySummedTopKindex.unsqueeze(batchDims+1).unsqueeze(batchDims+2)
	if(batchDims==1):
		arraySummedTopKindex = arraySummedTopKindex.repeat(batchDims, 1, numberOfBranchSequentialSegments, matrixPropagateTopKconceptNodes)	#len(HFconnectionGraphObject.neuronNamelist)	
	else:
		arraySummedTopKindex = arraySummedTopKindex.repeat(1, numberOfBranchSequentialSegments, matrixPropagateTopKconceptNodes)	#len(HFconnectionGraphObject.neuronNamelist)	
	arraySummedTopKindexSel = (arraySummedTopKindices == arraySummedTopKindex)
	arraySummedTopKindexMask = pt.sum(arraySummedTopKindexSel, dim=batchDims+2)	#along matrixPropagateTopKconceptNodes	#not necessary as arraySummedTopKindices should be unique for each sequential segment: (pt.sum(arraySummedTopKindexSel, dim=2) > 0).int()
	arraySummedTopKindexMask = pt.sum(arraySummedTopKindexMask, dim=batchDims+1)	#along sequential segments
	arraySummedTopKindexMaskCommon = (arraySummedTopKindexMask >= matrixPropagateTopCommonSegmentPredictionsRequired) 	#temp: (arraySummedTopKindexMask >= 0)

	#print("arraySummedTopKindices = ", arraySummedTopKindices)
	#print("arraySummedTopKvalues = ", arraySummedTopKvalues)
	#print("arraySummedTopKindex = ", arraySummedTopKindex)
	#print("arraySummedTopKindexSel = ", arraySummedTopKindexSel)
	#print("arraySummedTopKindexMask = ", arraySummedTopKindexMask)
	#print("arraySummedTopKindexMaskCommon = ", arraySummedTopKindexMaskCommon)

	arraySummedTopKindexMaskCommon = arraySummedTopKindexMaskCommon.unsqueeze(batchDims+1).unsqueeze(batchDims+2)
	if(batchDims==1):
		arraySummedTopKindexMaskCommon = arraySummedTopKindexMaskCommon.repeat(1, 1, numberOfBranchSequentialSegments, matrixPropagateTopKconceptNodes)
	else:
		arraySummedTopKindexMaskCommon = arraySummedTopKindexMaskCommon.repeat(1, numberOfBranchSequentialSegments, matrixPropagateTopKconceptNodes)
	arraySummedTopKindexSelCommon = pt.logical_and(arraySummedTopKindexSel, arraySummedTopKindexMaskCommon)

	#print("arraySummedTopKindexMaskCommon = ", arraySummedTopKindexMaskCommon)
	#print("arraySummedTopKindexSelCommon = ", arraySummedTopKindexSelCommon)

	arraySummedTopKindexSelCommonValues = arraySummedTopKvalues * arraySummedTopKindexSelCommon.float()
	arraySummedTopKindexSelCommonValues = pt.sum(arraySummedTopKindexSelCommonValues, dim=batchDims+2)	#along matrixPropagateTopKconceptNodes
	if(algorithmMatrixSANImethod=="addActivationAcrossSegments"):
		arraySummedTopKindexSelCommonValues = pt.sum(arraySummedTopKindexSelCommonValues, dim=batchDims+1, keepdims=True)	#along sequential segments
	elif(algorithmMatrixSANImethod=="supportSequentialActivationAcrossSegments"):
		printe("matrixPropagateTopCommonSegmentPredictions+supportSequentialActivationAcrossSegments not currently supported")
	elif(algorithmMatrixSANImethod=="enforceSequentialActivationAcrossSegments"):
		printe("connectionMatrixCalculateConnectionTargetSet error: algorithmMatrixSANImethod==enforceSequentialActivationAcrossSegments not coded")
		
	return arraySummedTopKindexSelCommonValues
	
	
def calculateSequentialSegmentActivation(connectionStrength):
	if(algorithmMatrixSANImethod=="addActivationAcrossSegments"):
		activationValue = connectionStrength
	elif(algorithmMatrixSANImethod=="supportSequentialActivationAcrossSegments"):
		printe("calculateSequentialSegmentActivation error: algorithmMatrixSANImethod==supportSequentialActivationAcrossSegments not coded")
	elif(algorithmMatrixSANImethod=="enforceSequentialActivationAcrossSegments"):
		printe("connectionMatrixCalculateConnectionTargetSet error: algorithmMatrixSANImethod==enforceSequentialActivationAcrossSegments not coded")
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
	if(not HFconnectionMatrixAlgorithmSplit):
		if(HFconnectionMatrixAlgorithmGPU):
			contextConnectionVector = contextConnectionVector.to(HFNLPpy_ConnectionMatrixOperations.device)
			
	return contextConnectionVector
	
def createContextVectorsSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentMax, weightStore):
	contextConnectionVectorList = []
	for sequentialSegmentIndex in range(sequentialSegmentMax):	#numberOfBranchSequentialSegments
		contextConnectionVector = createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore)
		contextConnectionVectorList.append(contextConnectionVector)
	if(HFconnectionMatrixAlgorithmSplit):
		contextConnectionVector = contextConnectionVectorList
	else:
		contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector
	
def createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore):
	if(sequentialSegmentContextEncoding=="relativeExponential"):
		expRange = createExponentialRange(0, w1, numberOfBranchSequentialSegments)
	contextSequenceLength = w1
	validSequentialSegment = True
	if(sequentialSegmentContextEncoding=="linear"):
		w2Min = w1-(numberOfBranchSequentialSegments*sequentialSegmentContextEncodingSize)+(sequentialSegmentIndex*sequentialSegmentContextEncodingSize)
		w2Max = w2Min+sequentialSegmentContextEncodingSize
		if(w2Min < 0):
			validSequentialSegment = False
	elif(sequentialSegmentContextEncoding=="relativeLinear"):
		contextSequenceSegmentLength = w1/numberOfBranchSequentialSegments
		w2Min = sequentialSegmentIndex*contextSequenceSegmentLength
		w2Max = w2Min + contextSequenceSegmentLength
	elif(sequentialSegmentContextEncoding=="relativeExponential"):
		w2Min = sum(expRange[0:sequentialSegmentIndex+1])
		w2Max = sum(expRange[0:sequentialSegmentIndex+2])
	w2Min = int(w2Min)
	w2Max = int(w2Max)
	if(validSequentialSegment):
		contextConnectionVector = createContextVectorSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, w2Min, w2Max, weightStore)	#len(HFconnectionGraphObject.neuronNamelist)
	else:
		contextConnectionVector = createContextVectorSANIempty(w1, sentenceConceptNodeList, HFconnectionGraphObject, w2Min, w2Max, weightStore)	#len(HFconnectionGraphObject.neuronNamelist)
		#contextConnectionVector = pt.zeros(HFconnectionGraphObject.connectionMatrixMaxConcepts)
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
	if(HFconnectionMatrixAlgorithmSplit):
		contextConnectionVector = contextConnectionVectorList
	else:
		contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector
	
def createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext):
	if(HFconnectionMatrixAlgorithmContextVectorSparse):
		if(bidirectionalContext):
			contextLength = len(sentenceConceptNodeList)
		else:
			contextLength = w1	#len(sentenceConceptNodeList) [too large]	#int(w2Max-w2Min) [not possible as will vary across secondDataIndex]	#contextSizeMax [too large]
	else:
		contextLength = contextSizeMax
	contextConnectionVector = HFNLPpy_ConnectionMatrixAlgorithm.createContextVectorTensor(HFconnectionGraphObject, contextLength)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(bidirectionalContext or (w2 < w1)):
				if(w1-w2 <= getContextSize(contextSizeIndex)):
					conceptNodeContext = sentenceConceptNodeList[w2]
					neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
					if(HFconnectionMatrixAlgorithmContextVectorSparse):
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
	contextConnectionVector = HFNLPpy_ConnectionMatrixAlgorithm.createContextVectorTensor(HFconnectionGraphObject, contextLength)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(w2 >= w2Min and w2 < w2Max):
				conceptNodeContext = sentenceConceptNodeList[w2]
				neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
				if(HFconnectionMatrixAlgorithmContextVectorSparse):
					contextConnectionVectorIndex = w2	#w2-w2Min
					contextConnectionVector.values[contextConnectionVectorIndex] = calculateContextVectorValue(weightStore, w1, w2)
					contextConnectionVector.indices[contextConnectionVectorIndex] = neuronIDcontext
				else:
					contextConnectionVector[neuronIDcontext] = calculateContextVectorValue(weightStore, w1, w2)
	return contextConnectionVector

def createContextVectorSANIempty(w1, sentenceConceptNodeList, HFconnectionGraphObject, w2Min, w2Max, weightStore):
	contextLength = w1	#len(sentenceConceptNodeList) [too large]	#int(w2Max-w2Min) [not possible as will vary across secondDataIndex]	#contextSizeMax [too large]
	contextConnectionVector = HFNLPpy_ConnectionMatrixAlgorithm.createContextVectorTensor(HFconnectionGraphObject, contextLength)
	return contextConnectionVector
	
def calculateContextVectorValue(weightStore, w1, w2):
	if(weightStore):
		weight = 1.0/(abs(w1 - w2))
		contextVectorValue = weight
	else:
		if(useHFconnectionMatrixAlgorithmBool):
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

def updateDendriticBranchClosestValue(foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, closestDendriticBranchIndex, targetSet, connectionStrength, dendriticBranchIndex, threshold=False, connectionStrengthNormalised=None):
	if(connectionStrength > closestConnectionStrength):
		if((not threshold) or (connectionStrengthNormalised > simulatedDendriticBranchesMinMatchStrength)):
			foundClosestBranchIndex = True
			if(dendriticBranchClosestTargetSet is not None):
				dendriticBranchClosestTargetSet = targetSet
			if(closestConnectionStrength is not None):
				closestConnectionStrength = connectionStrength
			if(closestDendriticBranchIndex is not None):
				closestDendriticBranchIndex = dendriticBranchIndex
	return foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, closestDendriticBranchIndex
	
