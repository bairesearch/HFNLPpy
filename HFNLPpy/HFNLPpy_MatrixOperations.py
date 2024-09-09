"""HFNLPpy_MatrixOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

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

import random

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

def connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4, k=matrixPropagateTopKconceptNodes, useReversePredictions=False, connectionTargetNeuronSet=None):
	if(algorithmMatrixPropagationOrder == "propagateForward"):
		conceptNeuronContextVector = createContextVectorWrapperForward(w1, sentenceConceptNodeList, HFconnectionGraphObject, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4)	#len(HFconnectionGraphObject.neuronNamelist)
		validContextVector = True
	elif(algorithmMatrixPropagationOrder == "propagateReverseLookup"):
		conceptNeuronContextVector = createContextVectorWrapperReverseLookup(w1, sentenceConceptNodeList, HFconnectionGraphObject, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4, useReversePredictions, connectionTargetNeuronSet)	#len(HFconnectionGraphObject.neuronNamelist)
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

if(algorithmMatrixSANImethod=="completeSANI"):
	def SANImethodUpdateActivationsIntegrate(HFconnectionGraphObject, activationsNew):
		activationsLevelPrevious = HFconnectionGraphObject.HFconnectionGraphActivationsLevel.unsqueeze(-1)
		activationsLevel = activationsLevelPrevious
		#activationTime = HFconnectionGraphObject.activationTime
		#numberOfSegments = activationsNew.shape[1]
		for i in range(numberOfBranchSequentialSegments):
			if(i > 0):
				priorActivation = activationsLevelPrevious[:, 0:i]
				if(activationDecayType=="linear"):
					space = createLinearSpace(0, 1, i)
				elif(activationDecayType=="exponential"):
					space = createExponentialSpace(0, 1, i)
				space = space.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
				priorActivationProximityBiased = priorActivation * space
				priorActivationProximityBiased = pt.sum(priorActivationProximityBiased, dim=1)
				activationsLevelUpdates = activationsNew[:, i] + priorActivationProximityBiased
			else:
				activationsLevelUpdates = activationsNew[:, i]
			activationsLevel[:, i] = activationsLevelPrevious[:, i] + activationsLevelUpdates
		HFconnectionGraphObject.HFconnectionGraphActivationsLevel = activationsLevel.squeeze(-1)	#overwrite stored activations
		#HFconnectionGraphObject.HFconnectionGraphActivationsTime = activationsTime	#overwrite stored activations
		return activationsLevel
'''
elif(algorithmMatrixSANImethod=="posthocSANI"):
	def SANImethodUpdateActivationsAdd(HFconnectionGraphObject, activationsNew):
		activationsLevelPrevious = HFconnectionGraphObject.HFconnectionGraphActivationsLevel.unsqueeze(-1)
		activationsLevel = activationsLevelPrevious + activationsNew
		activationsLevel = pt.sum(activationsLevel, dim=-1, keepdims=True)	#sum along input dimension
		HFconnectionGraphObject.HFconnectionGraphActivationsLevel = activationsLevel.squeeze(-1)	#overwrite stored activations
		return activationsLevel
'''
		
if(algorithmMatrixSANImethodPosthoc=="addActivationAcrossSegments"):
	def SANImethodAddActivationAcrossSegments(array):
		array = pt.sum(array, dim=1, keepdim=True)
		return array
elif(algorithmMatrixSANImethodPosthoc=="supportSequentialActivationAcrossSegments"):
	def SANImethodSupportSequentialActivationAcrossSegments(HFconnectionGraph, conceptNeuronContextVector):	
		#incomplete
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
elif(algorithmMatrixSANImethodPosthoc=="enforceSequentialActivationAcrossSegments"):
	#incomplete
	pass
elif(algorithmMatrixSANImethodPosthoc == "getLastSequentialSegmentActivation"):
	def SANImethodGetLastSegmentActivation(array):
		array = array[:, -1]	#select last segment
		array = array.unsqueeze(dim=1)
		return array


def connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject, HFconnectionGraph, neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, k, matrixTensorDim4):
	connectionTargetNeuronList = []
	conceptNeuronContextVectorExtended = HFNLPpy_ConnectionMatrixAlgorithm.extendConceptNeuronContextVector(HFconnectionGraphObject, conceptNeuronContextVector, matrixTensorDim4)
	if(matrixTensorDim4):
		if(not algorithmMatrixSANI):
			HFconnectionGraph = HFconnectionGraph[:, 0:conceptNeuronContextVectorExtended.shape[1], :, :]	#only compare selected contextIndices
			
	if(algorithmMatrixPropagationOrder=="propagateForward"):
		activationsNew = HFconnectionGraph * conceptNeuronContextVectorExtended
		if(algorithmMatrixSANImethod=="completeSANI"):
			activations = SANImethodUpdateActivationsIntegrate(HFconnectionGraphObject, activationsNew)
			#print("activations.sum() = ", activations.sum())
		'''
		elif(algorithmMatrixSANImethod=="posthocSANI"):
			activations = SANImethodUpdateActivationsAdd(HFconnectionGraphObject, activationsNew)
		'''
	else:
		activations = HFconnectionGraph * conceptNeuronContextVectorExtended
	#activations shape: [numberOfIndependentDendriticBranches, numberOfBranchSequentialSegments, connectionMatrixMaxConcepts[target], sentencePriorContextSize[source]]
	array = activations
	
	neuronIndexList = []
	if(matrixTensorDim4):
		if(not matrixPropagateTopCommonSegmentPredictions):
			if(algorithmMatrixSANImethodPosthoc=="addActivationAcrossSegments"):
				array = SANImethodAddActivationAcrossSegments(array)
			elif(algorithmMatrixSANImethodPosthoc=="supportSequentialActivationAcrossSegments"):
				array = SANImethodSupportSequentialActivationAcrossSegments(HFconnectionGraph, conceptNeuronContextVectorExtended)
			elif(algorithmMatrixSANImethodPosthoc=="enforceSequentialActivationAcrossSegments"):
				printe("connectionMatrixCalculateConnectionTargetSet error: algorithmMatrixSANImethodPosthoc==enforceSequentialActivationAcrossSegments not coded")
			elif(algorithmMatrixSANImethodPosthoc == "getLastSequentialSegmentActivation"):
				array = SANImethodGetLastSegmentActivation(array)
			
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
	if(algorithmMatrixSANImethodPosthoc=="addActivationAcrossSegments"):
		arraySummedTopKindexSelCommonValues = pt.sum(arraySummedTopKindexSelCommonValues, dim=batchDims+1, keepdims=True)	#along sequential segments
	elif(algorithmMatrixSANImethodPosthoc=="supportSequentialActivationAcrossSegments"):
		printe("matrixPropagateTopCommonSegmentPredictions+supportSequentialActivationAcrossSegments not currently supported")
	elif(algorithmMatrixSANImethodPosthoc=="enforceSequentialActivationAcrossSegments"):
		printe("connectionMatrixCalculateConnectionTargetSet error: algorithmMatrixSANImethodPosthoc==enforceSequentialActivationAcrossSegments not coded")
		
	return arraySummedTopKindexSelCommonValues
	
	
def calculateSequentialSegmentActivation(connectionStrength):
	if(algorithmMatrixSANImethodPosthoc=="addActivationAcrossSegments"):
		activationValue = connectionStrength
	elif(algorithmMatrixSANImethodPosthoc=="supportSequentialActivationAcrossSegments"):
		printe("calculateSequentialSegmentActivation error: algorithmMatrixSANImethodPosthoc==supportSequentialActivationAcrossSegments not coded")
	elif(algorithmMatrixSANImethodPosthoc=="enforceSequentialActivationAcrossSegments"):
		printe("connectionMatrixCalculateConnectionTargetSet error: algorithmMatrixSANImethodPosthoc==enforceSequentialActivationAcrossSegments not coded")
	return activationValue

def performSumTopK(array, k, dim):
	arraySummed = pt.sum(array, dim=dim)
	arraySummedTopK = pt.topk(arraySummed, k, dim=dim-1)
	return arraySummedTopK
		
if(algorithmMatrixPropagationOrder == "propagateForward"):
	def createContextVectorWrapperForward(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, contextSizeMaxSource, weightStore, bidirectionalContext, matrixTensorDim4):
		if(algorithmMatrixSANI):
			if(matrixTensorDim4):
				contextConnectionVector = createContextVectorsSANIForward(w1, sentenceConceptNodeList, HFconnectionGraphObject, weightStore)
			else:
				contextConnectionVector = createContextVectorSANIForward(w1, sentenceConceptNodeList, HFconnectionGraphObject, weightStore)
		else:
			printe("createContextInputWrapper error: propagateForward requires algorithmMatrixSANI")

		if(not HFconnectionMatrixAlgorithmSplit):
			if(HFconnectionMatrixAlgorithmGPU):
				contextConnectionVector = contextConnectionVector.to(HFNLPpy_ConnectionMatrixOperations.device)

		return contextConnectionVector

	def createContextVectorsSANIForward(w1, sentenceConceptNodeList, HFconnectionGraphObject, weightStore):
		contextConnectionVectorList = []	
		for sequentialSegmentIndex in range(numberOfBranchSequentialSegments):
			#propagateForward creates identical contextInput for each target sequential segment
			contextConnectionVector = createContextVectorSANI1Forward(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore)
			contextConnectionVectorList.append(contextConnectionVector)
		if(HFconnectionMatrixAlgorithmSplit):
			contextConnectionVector = contextConnectionVectorList
		else:
			contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
		return contextConnectionVector

	def createContextVectorSANI1Forward(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore):
		contextConnectionVector = createContextVectorSANIForward(w1, sentenceConceptNodeList, HFconnectionGraphObject, weightStore)	#len(HFconnectionGraphObject.neuronNamelist)
		return contextConnectionVector

	def createContextVectorSANIForward(w1, sentenceConceptNodeList, HFconnectionGraphObject, weightStore):
		contextLength = 1
		contextConnectionVector = HFNLPpy_ConnectionMatrixAlgorithm.createContextVectorTensor(HFconnectionGraphObject, contextLength)
		conceptNodeContext = sentenceConceptNodeList[w1]	#CHECKTHIS
		neuronIDcontext = HFconnectionGraphObject.neuronIDdict[conceptNodeContext.nodeName]
		if(HFconnectionMatrixAlgorithmContextVectorSparse):
			contextConnectionVectorIndex = 0
			contextConnectionVector.values[contextConnectionVectorIndex] = calculateContextScalarValue(weightStore, w1)	
			contextConnectionVector.indices[contextConnectionVectorIndex] = neuronIDcontext
		else:
			contextConnectionVector[neuronIDcontext] = calculateContextScalarValue(weightStore, contextConnectionVectorIndex)
		return contextConnectionVector
	
def createContextVectorWrapperReverseLookup(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, contextSizeMaxSource, weightStore, bidirectionalContext, matrixTensorDim4, useReversePredictions=False, connectionTargetNeuronSet=None):
	if(algorithmMatrixSANI):
		if(matrixTensorDim4):
			contextConnectionVector = createContextVectorsSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeMaxSource, weightStore, useReversePredictions, connectionTargetNeuronSet)
		else:
			contextConnectionVector = createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, useReversePredictions, connectionTargetNeuronSet)
	else:
		if(matrixTensorDim4):
			contextConnectionVector = createContextVectors(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeMaxSource, weightStore, bidirectionalContext, useReversePredictions, connectionTargetNeuronSet)
		else:
			contextConnectionVector = createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext, useReversePredictions, connectionTargetNeuronSet)
	if(not HFconnectionMatrixAlgorithmSplit):
		if(HFconnectionMatrixAlgorithmGPU):
			contextConnectionVector = contextConnectionVector.to(HFNLPpy_ConnectionMatrixOperations.device)

	return contextConnectionVector

	
def createContextVectors(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeMaxSource, weightStore, bidirectionalContext, useReversePredictions=False, connectionTargetNeuronSet=None):
	contextConnectionVectorList = []
	for contextSizeIndex in range(contextSizeMaxSource):
		contextConnectionVector = createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext, useReversePredictions, connectionTargetNeuronSet)	#len(HFconnectionGraphObject.neuronNamelist)
		contextConnectionVectorList.append(contextConnectionVector)
	if(HFconnectionMatrixAlgorithmSplit):
		contextConnectionVector = contextConnectionVectorList
	else:
		contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector
	
def createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext, useReversePredictions=False, connectionTargetNeuronSet=None):
	if(HFconnectionMatrixAlgorithmContextVectorSparse):
		contextLength = getContextLength(w1, sentenceConceptNodeList, bidirectionalContext)
	else:
		contextLength = contextSizeMax
	contextConnectionVector = HFNLPpy_ConnectionMatrixAlgorithm.createContextVectorTensor(HFconnectionGraphObject, contextLength)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(bidirectionalContext or (w2 < w1)):
				if(w1-w2 <= getContextSize(contextSizeIndex)):
					addToContextVector(w1, w2, sentenceConceptNodeList, HFconnectionGraphObject, contextConnectionVector, weightStore, bidirectionalContext)
	if(useReversePredictions):
		#add future candidate predictions to context vector
		for futurePredictionTargetConceptNeuron in connectionTargetNeuronSet:
			w2 = futurePredictionTargetConceptNeuron.w
			addToContextVector(w1, w2, sentenceConceptNodeList, HFconnectionGraphObject, contextConnectionVector, weightStore, bidirectionalContext)
	return contextConnectionVector

def addToContextVector(w1, w2, sentenceConceptNodeList, HFconnectionGraphObject, contextConnectionVector, weightStore, bidirectionalContext):
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


def createContextVectorsSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentMax, weightStore, useReversePredictions=False, connectionTargetNeuronSet=None):
	contextConnectionVectorList = []
	for sequentialSegmentIndex in range(sequentialSegmentMax):	#numberOfBranchSequentialSegments
		contextConnectionVector = createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore, useReversePredictions, connectionTargetNeuronSet)
		contextConnectionVectorList.append(contextConnectionVector)
	if(HFconnectionMatrixAlgorithmSplit):
		contextConnectionVector = contextConnectionVectorList
	else:
		contextConnectionVector = pt.stack(contextConnectionVectorList, dim=0)
	return contextConnectionVector

def createContextVectorSANI1(w1, sentenceConceptNodeList, HFconnectionGraphObject, sequentialSegmentIndex, weightStore, useReversePredictions=False, connectionTargetNeuronSet=None):
	contextSequenceLength = getContextLength(w1, sentenceConceptNodeList)	
	validSequentialSegment = True
	if(sequentialSegmentContextEncoding=="linear"):
		#print("contextSequenceLength = ", contextSequenceLength)
		w2Min = contextSequenceLength-(numberOfBranchSequentialSegments*sequentialSegmentContextEncodingSize)+(sequentialSegmentIndex*sequentialSegmentContextEncodingSize)
		w2Max = w2Min+sequentialSegmentContextEncodingSize
		if(w2Min < 0):
			validSequentialSegment = False
	elif(sequentialSegmentContextEncoding=="relativeLinear"):
		contextSequenceSegmentLength = contextSequenceLength/numberOfBranchSequentialSegments
		w2Min = sequentialSegmentIndex*contextSequenceSegmentLength
		w2Max = w2Min + contextSequenceSegmentLength
	elif(sequentialSegmentContextEncoding=="relativeExponential"):
		#createExponentialRange(0, contextSequenceLength, numberOfBranchSequentialSegments)
		#w2Min = sum(expRange[0:sequentialSegmentIndex+1])
		#w2Max = sum(expRange[0:sequentialSegmentIndex+2])
		expRange = createExponentialRange(0, sequentialSegmentContextEncodingMaxLength, numberOfBranchSequentialSegments+1, decay_rate=4)
		#print("expRange = ", expRange)
		expRange = [number-sequentialSegmentContextEncodingMaxLength+contextSequenceLength for number in expRange]
		#print("expRange = ", expRange)
		w2Min = expRange[sequentialSegmentIndex]
		w2Max = expRange[sequentialSegmentIndex+1]
		if(w2Min < 0):
			validSequentialSegment = False
	w2Min = int(w2Min)
	w2Max = int(w2Max)
	#print("w2Min = ", w2Min)
	#print("w2Max = ", w2Max)
	#print("sequentialSegmentIndex = ", sequentialSegmentIndex)
	#print("w1 = ", w1)
	if(validSequentialSegment):
		contextConnectionVector = createContextVectorSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, w2Min, w2Max, weightStore, useReversePredictions, connectionTargetNeuronSet)	#len(HFconnectionGraphObject.neuronNamelist)
	else:
		contextConnectionVector = createContextVectorSANIempty(w1, sentenceConceptNodeList, HFconnectionGraphObject, useReversePredictions)	#len(HFconnectionGraphObject.neuronNamelist)
		#contextConnectionVector = pt.zeros(HFconnectionGraphObject.connectionMatrixMaxConcepts)
	return contextConnectionVector

#if decay; go from maxVal to minVal
def createExponentialRange(minVal, maxVal, s, decay_rate=5):
    # Generate s numbers between 0 and 1
    uniform_samples = np.linspace(0, 1, s)

    # Apply the exponential decay function to the uniform samples with the specified decay rate
    exponential_samples = (1 - np.exp(-decay_rate * uniform_samples))

    # Normalize the samples to have a minimum of 0 and a maximum of 1
    normalized_samples = (exponential_samples - np.min(exponential_samples)) / (np.max(exponential_samples) - np.min(exponential_samples))

    # Scale the samples to be between minVal and maxVal
    scaled_samples = normalized_samples * (maxVal - minVal) + minVal

    # Convert the samples to integers
    expRange = np.round(scaled_samples).astype(int)

    return expRange

def createLinearSpace(minVal, maxVal, size):
	space = pt.linspace(minVal, maxVal, size)
	if(HFconnectionMatrixAlgorithmGPU):
		space = space.to(HFNLPpy_ConnectionMatrixOperations.device)
	return space
	
def createExponentialSpace(minVal, maxVal, size):
	space = pt.logspace(torch.log10(minVal), torch.log10(maxVal), size)
	if(HFconnectionMatrixAlgorithmGPU):
		space = space.to(HFNLPpy_ConnectionMatrixOperations.device)
	return space
	
def createContextVectorSANI(w1, sentenceConceptNodeList, HFconnectionGraphObject, w2Min, w2Max, weightStore, useReversePredictions=False, connectionTargetNeuronSet=None):
	contextLength = getContextLength(w1, sentenceConceptNodeList)
	contextConnectionVector = HFNLPpy_ConnectionMatrixAlgorithm.createContextVectorTensor(HFconnectionGraphObject, contextLength)
	for w2, conceptNeuron2 in enumerate(sentenceConceptNodeList):
		if(w1 != w2):
			if(w2 >= w2Min and w2 < w2Max):
				addToContextVector(w1, w2, sentenceConceptNodeList, HFconnectionGraphObject, contextConnectionVector, weightStore, False)
	if(useReversePredictions):
		#add future candidate predictions to context vector
		for futurePredictionTargetConceptNeuron in connectionTargetNeuronSet:
			w2 = futurePredictionTargetConceptNeuron.w
			if(w2 >= w2Min and w2 < w2Max):
				addToContextVector(w1, w2, sentenceConceptNodeList, HFconnectionGraphObject, contextConnectionVector, weightStore, bidirectionalContext)
	return contextConnectionVector

def createContextVectorSANIempty(w1, sentenceConceptNodeList, HFconnectionGraphObject, useReversePredictions=False):
	contextLength = getContextLength(w1, sentenceConceptNodeList)
	contextConnectionVector = HFNLPpy_ConnectionMatrixAlgorithm.createContextVectorTensor(HFconnectionGraphObject, contextLength)
	return contextConnectionVector

def getContextLength(w1, sentenceConceptNodeList, bidirectionalContext=False):
	if(bidirectionalContext or reversePredictions):
		contextLength = len(sentenceConceptNodeList)
	else:
		contextLength = w1	#len(sentenceConceptNodeList) [too large]	#int(w2Max-w2Min) [not possible as will vary across secondDataIndex]	#contextSizeMax [too large]
	return contextLength
	
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

def calculateContextScalarValue(weightStore, w1):
	if(weightStore):
		printe("calculateContextScalarValue error: propagateForward currently requires !weightStore")
	else:
		if(useHFconnectionMatrixAlgorithmBool):
			contextScalarValue = True
		else:
			contextScalarValue = 1.0
	return contextScalarValue
	

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
	
