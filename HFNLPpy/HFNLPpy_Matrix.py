"""HFNLPpy_Matrix.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Matrix - simulate training/inference of matrix hopfield graph/network based on textual input

training;
	activate concept neurons in order of sentence word order
inference;
	calculate neuron firing exclusively from prior/contextual subsequence detections

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_MatrixGlobalDefs import *
import HFNLPpy_MatrixGenerate
import HFNLPpy_MatrixPropagate
import HFNLPpy_hopfieldOperations
import HFNLPpy_MatrixOperations
if(useHFconnectionMatrixBasic):
	import HFNLPpy_ConnectionMatrixBasic as HFNLPpy_ConnectionMatrix
import random

printVerbose = False

def seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences, HFconnectionGraphObject):
	
	targetSentenceConceptNodeList = seedSentenceConceptNodeList
	
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	if(not seedHFnetworkSubsequenceBasic):
		conceptNeuronSourceList = []

	for wSource in range(len(targetSentenceConceptNodeList)-1):
		wTarget = wSource+1
		conceptNeuronSource = targetSentenceConceptNodeList[wSource]
		conceptNeuronTarget = targetSentenceConceptNodeList[wTarget]
		print("\nseedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
		
		if(seedHFnetworkSubsequenceBasic):
			activationTime = calculateActivationTimeSequence(wSource)
			somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet, HFconnectionGraphObject)
		else:
			connectionTargetNeuronSetLocal = set()
			activationTime = None
			if(wSource < seedHFnetworkSubsequenceLength):
				somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSetLocal, HFconnectionGraphObject)
			else:
				somaActivationFound = simulateBiologicalHFnetworkSequenceNodesPropagateForward(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSourceList, connectionTargetNeuronSetLocal, HFconnectionGraphObject)
			
			connectionTargetNeuronSetLocalFiltered = selectActivatedNeurons(networkConceptNodeDict, connectionTargetNeuronSetLocal, HFconnectionGraphObject)
				
			conceptNeuronSourceList.clear()
			for connectionTargetNeuron in connectionTargetNeuronSetLocalFiltered:
				print("\tconceptNeuronSourceList.append connectionTargetNeuron = ", connectionTargetNeuron.nodeName)
				conceptNeuronSourceList.append(connectionTargetNeuron)
			connectionTargetNeuronSet = connectionTargetNeuronSet.union(connectionTargetNeuronSetLocal)

		expectPredictiveSequenceToBeFound = False
		if(enforceMinimumEncodedSequenceLength):
			if(wSource >= minimumEncodedSequenceLength-1):
				expectPredictiveSequenceToBeFound = True
		else:
			expectPredictiveSequenceToBeFound = True
		if(expectPredictiveSequenceToBeFound):
			if(somaActivationFound):
				#if(printVerbose):
				print("somaActivationFound")
			else:
				#if(printVerbose):
				print("!somaActivationFound: HFNLP algorithm error detected")
		else:
			print("!expectPredictiveSequenceToBeFound: wSource < minimumEncodedSequenceLength-1")
			


def simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet, HFconnectionGraphObject):
	print("simulateBiologicalHFnetworkSequenceNodePropagateForward: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)	
	somaActivationFound = HFNLPpy_MatrixPropagate.simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject)
	return somaActivationFound

def simulateBiologicalHFnetworkSequenceNodesPropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSourceList, connectionTargetNeuronSet, HFconnectionGraphObject):
	somaActivationFound = HFNLPpy_MatrixPropagate.simulateBiologicalHFnetworkSequenceNodesPropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject)
	return somaActivationFound

'''
#independent method (does not need to be executed in order of wSource)
def simulateBiologicalHFnetworkSequenceNodePropagateForwardFull(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
	somaActivationFound = False
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	
	for wSource, conceptNeuronSource in enumerate(sentenceConceptNodeList):	#support for simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain:!biologicalSimulationEncodeSyntaxInDendriticBranchStructureFormat
	#orig for wSource in range(0, wTarget):
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		activationTime = calculateActivationTimeSequence(wSource)
		
		connectionTargetNeuronSetLocal = set()
		somaActivationFoundTemp = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSetLocal)
		
		if(wSource == len(sentenceConceptNodeList)-1):
			if(somaActivationFoundTemp):
				somaActivationFound = True
		
		connectionTargetNeuronSet = connectionTargetNeuronSet.union(connectionTargetNeuronSetLocal)
		
	return somaActivationFound
'''

def selectActivatedNeurons(networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject=None):
	connectionTargetNeuronSetLocalFiltered = connectionTargetNeuronSet
	if(linkSimilarConceptNodes):
		connectionTargetNeuronSetLocalFiltered = retrieveSimilarConcepts(networkConceptNodeDict, connectionTargetNeuronSetLocalFiltered, HFconnectionGraphObject)
	return connectionTargetNeuronSetLocalFiltered
	
	
#connection graph generation;

def addContextWordsToConnectionGraphLinkConcepts(tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject):
	for w1, token1 in enumerate(tokenisedSentence):
		conceptNode = sentenceConceptNodeList[w1]
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		HFconnectionGraphObject.HFconnectionGraphBasic, HFconnectionGraphObject.HFconnectionGraphBasicNormalised = addContextWordsToConnectionGraph(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphBasic, linkSimilarConceptNodesBagOfWordsDistanceMax, linkSimilarConceptNodesBagOfWordsWeightStore, linkSimilarConceptNodesBagOfWordsBidirectional)

def addContextWordsToConnectionGraphMatrix(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject):
	for w1, token1 in enumerate(tokenisedSentence):
		#print("w1 = ", w1)
		conceptNode = sentenceConceptNodeList[w1]
		#print("addContextWordsToConnectionGraphMatrix: conceptNode.nodeName = ", conceptNode.nodeName) 
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax(getContextSizeSource=True, wSource=w1)
		dendriticBranchClosestIndex = calculateDendriticBranchClosestIndex(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1)
		if(algorithmMatrixSingleTensorEfficientAdd):
			HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchClosestIndex], HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchClosestIndex] = addContextWordsToConnectionGraph(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchClosestIndex], None, secondDataIndexMax, contextMatrixWeightStore, False)
		else:
			for secondDataIndex in range(secondDataIndexMax):
				#print("contextSizeIndex = ", contextSizeIndex)
				HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchClosestIndex][secondDataIndex], HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchClosestIndex][secondDataIndex] = addContextWordsToConnectionGraph(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchClosestIndex][secondDataIndex], secondDataIndex, secondDataIndexMax, contextMatrixWeightStore, False)

def calculateDendriticBranchClosestIndex(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1):
	dendriticBranchClosestIndex = 0
	if(numberOfIndependentDendriticBranches == 1):
		dendriticBranchClosestIndex = 0
	else:
		foundDendriticBranchClosest = False
		dendriticBranchClosestIndex = 0
		dendriticBranchClosestValue = 0
		if(algorithmMatrixSingleTensor):
			if(algorithmMatrixSANI or secondDataIndexMax > 0):
				connectionStrength, dendriticBranchIndex = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionStrengthIndex(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised, None, secondDataIndexMax, contextMatrixWeightStore, False)
				if(connectionStrength > simulatedDendriticBranchesMinMatchStrength):
					foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex = updateDendriticBranchClosestValue(foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex, connectionStrength, dendriticBranchIndex)
		else:
			for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
				#print("dendriticBranchIndex = ", dendriticBranchIndex)
				if(algorithmMatrixSANI):
					connectionStrengthSegments = 0
				for secondDataIndex in range(secondDataIndexMax):
					#print("contextSizeIndex = ", contextSizeIndex)
					connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionStrengthIndex(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][secondDataIndex], secondDataIndex, None, contextMatrixWeightStore, False)
					if(algorithmMatrixSANI):
						connectionStrengthSegments += calculateSequentialSegmentActivation(connectionStrength)
					else:
						if(connectionStrength/secondDataIndex > simulatedDendriticBranchesMinMatchStrength):
							foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex = updateDendriticBranchClosestValue(foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex, connectionStrength, dendriticBranchIndex)
				if(algorithmMatrixSANI):
					foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex = updateDendriticBranchClosestValue(foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex, connectionStrengthSegments, dendriticBranchIndex)
					
		#if(debugAlgorithmMatrix):
		print("dendriticBranchClosestIndex = ", dendriticBranchClosestIndex)
		if(not foundDendriticBranchClosest):
			dendriticBranchClosestIndex = random.randint(0, numberOfIndependentDendriticBranches-1)
	return dendriticBranchClosestIndex

def updateDendriticBranchClosestValue(foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex, connectionStrength, dendriticBranchIndex):
	if(connectionStrength > dendriticBranchClosestValue):
		foundDendriticBranchClosest = True
		dendriticBranchClosestValue = connectionStrength
		dendriticBranchClosestIndex = dendriticBranchIndex
	return foundDendriticBranchClosest, dendriticBranchClosestValue, dendriticBranchClosestIndex
			
def addContextWordsToConnectionGraph(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraph, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext):
	contextConnectionVector = HFNLPpy_MatrixOperations.createContextVectorWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, len(HFconnectionGraphObject.neuronNamelist), secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, algorithmMatrixSingleTensorEfficientAdd)
	if(algorithmMatrixSingleTensorEfficientAdd):
		#contextConnectionVector = HFNLPpy_ConnectionMatrixBasic.extendConceptNeuronContextVector(contextConnectionVector, secondDataIndexMax)
		printe("algorithmMatrixSingleTensorEfficientAdd is incomplete")
	HFNLPpy_ConnectionMatrix.addContextConnectionsToGraph(HFconnectionGraph, neuronID, contextConnectionVector)
	HFconnectionGraphNormalised = HFNLPpy_MatrixOperations.normaliseBatchedTensor(HFconnectionGraph)
	#if(debugAlgorithmMatrix):
	#	print("contextConnectionVector = ", contextConnectionVector)
	#	#print("HFconnectionGraph[neuronID] = ", HFconnectionGraph[neuronID])
	#	print("HFconnectionGraphNormalised[neuronID] = ", HFconnectionGraphNormalised[neuronID])
	return HFconnectionGraph, HFconnectionGraphNormalised

def retrieveSimilarConcepts(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject=None):
	if(linkSimilarConceptNodesWordnet):
		for conceptNeuron in connectionTargetNeuronSet:
			connectionTargetNeuronSetExtended.append(conceptNeuron)
			for synonym in conceptNeuron.synonymsList:
				synonymConcept, conceptInDict = convertLemmaToConcept(networkConceptNodeDict, synonym)
				if(conceptInDict):
					#print("conceptInDict: ", synonymConcept.nodeName)
					connectionTargetNeuronSetExtended.append(synonymConcept)
	elif(linkSimilarConceptNodesBagOfWords):
		connectionTargetNeuronSetExtended = HFNLPpy_MatrixOperations.retrieveSimilarConceptsBagOfWords(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject)

	return connectionTargetNeuronSetExtended
