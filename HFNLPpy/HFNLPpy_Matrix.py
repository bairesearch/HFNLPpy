"""HFNLPpy_Matrix.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

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
from HFNLPpy_globalDefs import *	#required for linkSimilarConceptNodesBagOfWords
from HFNLPpy_MatrixGlobalDefs import *
import HFNLPpy_MatrixGenerate
import HFNLPpy_MatrixPropagate
import HFNLPpy_hopfieldOperations
import HFNLPpy_MatrixOperations
import HFNLPpy_ConnectionMatrixAlgorithm
import HFNLPpy_ConnectionMatrixBasic
if(linkSimilarConceptNodesBagOfWords):
	import HFNLPpy_ConceptsMatrixOperations
import HFNLPpy_ConnectionMatrixOperations
if(HFconnectionMatrixAlgorithmSplitDatabase):
	import HFNLPpy_MatrixDatabase
	
import random

printVerbose = False

if(seedHFnetworkSubsequenceType=="all"):
	feedPredictionErrors = 0
	feedPredictionSuccesses = 0

def addSentenceConceptNodeToHFconnectionGraphObject(HFconnectionGraphObject, conceptNode):
	if(useHFconnectionMatrix):
		neuronIDdictNewlyAdded = {}	#CHECKTHIS
		neuronIDdictPrevious = HFconnectionGraphObject.neuronIDdict.copy()	#CHECKTHIS
		HFNLPpy_ConnectionMatrixOperations.addSentenceConceptNodeToHFconnectionGraphObject(HFconnectionGraphObject, conceptNode)
		if(HFconnectionMatrixAlgorithmSplit):
			loadSentenceMatrixAlgorithmSplitMatrix(HFconnectionGraphObject, conceptNode, neuronIDdictPrevious, neuronIDdictNewlyAdded)

#preconditions: assumes addSentenceConceptNodesToHFconnectionGraphObject has already been executed
def loadSentenceMatrixAlgorithmSplitMatrices(HFconnectionGraphObject, sentenceConceptNodeList, neuronIDdictPrevious, neuronIDdictNewlyAdded):
	if(HFconnectionMatrixAlgorithmSplit):
		for conceptNodeIndex, conceptNode in enumerate(sentenceConceptNodeList):
			loadSentenceMatrixAlgorithmSplitMatrix(HFconnectionGraphObject, conceptNode, neuronIDdictPrevious, neuronIDdictNewlyAdded)

def loadSentenceMatrixAlgorithmSplitMatrix(HFconnectionGraphObject, conceptNode, neuronIDdictPrevious, neuronIDdictNewlyAdded):
	if(HFconnectionMatrixAlgorithmSplit):
		if(conceptNode.nodeName not in neuronIDdictNewlyAdded):	#ignore repeated concepts in sentence
			neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
			if(not conceptNode.nodeName in neuronIDdictPrevious):
				#dynamically initialise HFconnectionGraphMatrix when new concepts are declared
				HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID], _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID], HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
				neuronIDdictNewlyAdded[conceptNode.nodeName] = True
			else:
				#print("conceptNode.nodeName = ", conceptNode.nodeName)
				if(HFconnectionMatrixAlgorithmSplitDatabase):
					#load HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID] into RAM
					HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID] = HFNLPpy_MatrixDatabase.loadMatrixDatabaseFile(HFconnectionGraphObject, neuronID)
			'''
			if(HFconnectionMatrixTime):
				HFconnectionGraphObject.HFconnectionGraphMatrixTime[neuronID] = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrixTime[neuronID], False, HFconnectionMatrixAlgorithmSplit, False, HFconnectionsMatrixTimeType)
			if(HFconnectionMatrixInContextAssociations):
				HFconnectionGraphObject.HFconnectionGraphMatrixInContextAssociations[neuronID] = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrixInContextAssociations[neuronID], False, HFconnectionMatrixAlgorithmSplit, False, HFconnectionsMatrixInContextAssociationsType)
			'''
									
def HFconnectionGraphMatrixHolderInitialisation(self):
	if(HFconnectionMatrixAlgorithmSplit):
		self.HFconnectionGraphMatrix = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolderSplit()
		self.HFconnectionGraphMatrixMin = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()	#for dynamic normalisation (on demand)
		self.HFconnectionGraphMatrixMax = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()	#for dynamic normalisation (on demand)
		if(HFconnectionMatrixAlgorithmSplitDatabase):
			print("HFconnectionGraphMatrixHolderInitialisation warning: !HFconnectionMatrixAlgorithmSplitDatabase requires check")
		'''
		if(HFconnectionMatrixTime):
			self.HFconnectionGraphMatrixTime = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolderSplit()
		if(HFconnectionMatrixInContextAssociations):
			self.HFconnectionGraphMatrixInContextAssociations = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolderSplit()
		'''
	else:
		self.HFconnectionGraphMatrix = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
		if(HFconnectionMatrixAlgorithmNormaliseStore):
			self.HFconnectionGraphMatrixNormalised = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
		'''
		if(HFconnectionMatrixTime):
			self.HFconnectionGraphMatrixTime = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
		if(HFconnectionMatrixInContextAssociations):
			self.HFconnectionGraphMatrixInContextAssociations = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
		'''
	if(simulatedDendriticBranchesIndependentProximalContext):
		self.simulatedDendriticBranchesProximalContext = None	#HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()

def initialiseHFconnectionMatrixAlgorithmWrapper(HFconnectionGraphObject):
	HFNLPpy_ConnectionMatrixOperations.initialiseNeuronNameList(HFconnectionGraphObject)
	if(HFconnectionMatrixAlgorithmSplit):
		HFconnectionGraphObject.HFconnectionGraphMatrixMin, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrixMin, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
		HFconnectionGraphObject.HFconnectionGraphMatrixMax, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrixMax, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
		''' 
		#uses too much RAM (instead dynamically initialise HFconnectionGraphMatrix when new concepts are declared) - see HFNLPpy_hopfieldGraph:addSentenceConceptNodesToHFconnectionGraphObject:loadSentenceMatrixAlgorithmSplitMatrices
		if(HFconnectionMatrixAlgorithmSplitRAM):
			for sourceNeuronID in range(HFconnectionMatrixBasicMaxConcepts):
				HFconnectionGraphObject.HFconnectionGraphMatrix[sourceNeuronID], _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore)
				if(HFconnectionMatrixTime):
					HFconnectionGraphObject.HFconnectionGraphMatrixTime[sourceNeuronID], _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore)
				if(HFconnectionMatrixInContextAssociations):
					HFconnectionGraphObject.HFconnectionGraphMatrixInContextAssociations[sourceNeuronID], _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore)
		'''
	else:
		HFconnectionGraphObject.HFconnectionGraphMatrix, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
		'''
		if(HFconnectionMatrixTime):
			HFconnectionGraphObject.HFconnectionGraphMatrixTime, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
		if(HFconnectionMatrixInContextAssociations):
			HFconnectionGraphObject.HFconnectionGraphMatrixInContextAssociations, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
		'''
		
	initialiseHFconnectionMatrixWrapperAlgorithmMatrixActivationRecord(HFconnectionGraphObject)
	if(simulatedDendriticBranchesIndependentProximalContext):
		tensorShape = (numberOfIndependentDendriticBranches, HFconnectionGraphObject.connectionMatrixMaxConcepts)	#shape = numberOfIndependentProximalContexts * numberOfTargets
		HFconnectionGraphObject.simulatedDendriticBranchesProximalContext = pt.ones(tensorShape, dtype=HFconnectionsMatrixAlgorithmType)*simulatedDendriticBranchesIndependentProximalContextNeuronIDmissing	#initialise to -1
		if(HFconnectionMatrixAlgorithmGPU):
			HFconnectionGraphObject.simulatedDendriticBranchesProximalContext = HFconnectionGraphObject.simulatedDendriticBranchesProximalContext.to(HFNLPpy_ConnectionMatrixOperations.device)

def initialiseHFconnectionMatrixWrapperAlgorithmMatrixActivationRecord(HFconnectionGraphObject):
	if(algorithmMatrixPropagationOrder=="propagateForward"):
		HFconnectionGraphObject.HFconnectionGraphActivationsLevel, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, None, False, True, False, pt.float)
		HFconnectionGraphObject.HFconnectionGraphActivationsTime, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, None, False, True, False, pt.long)

def initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphMatrix, readSavedConnectionsMatrixAlgorithm, connectionMatrixAlgorithmSplit, connectionMatrixAlgorithmNormaliseStore, dtype):
	if(not connectionMatrixAlgorithmNormaliseStore):
		HFconnectionGraphMatrixNormalised=None
	if(algorithmMatrixTensorDim==4):
		HFconnectionGraphMatrix = HFNLPpy_ConnectionMatrixAlgorithm.initialiseHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, readSavedConnectionsMatrixAlgorithm, connectionMatrixAlgorithmSplit, dtype)
		if(connectionMatrixAlgorithmNormaliseStore):
			HFconnectionGraphMatrixNormalised = HFNLPpy_ConnectionMatrixAlgorithm.normaliseBatchedTensor(HFconnectionGraphMatrix)
	else:
		secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			if(algorithmMatrixTensorDim==3):
				HFconnectionGraphMatrix[dendriticBranchIndex] = HFNLPpy_ConnectionMatrixAlgorithm.initialiseHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, readSavedConnectionsMatrixAlgorithm, connectionMatrixAlgorithmSplit, dtype, HFNLPpy_ConnectionMatrixAlgorithm.createIndexStringDendriticBranch(dendriticBranchIndex))
				if(connectionMatrixAlgorithmNormaliseStore):
					HFconnectionGraphMatrixNormalised[dendriticBranchIndex] = HFNLPpy_ConnectionMatrixAlgorithm.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex])
			else:
				for secondDataIndex in range(secondDataIndexMax):
					HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex] = HFNLPpy_ConnectionMatrixAlgorithm.initialiseHFconnectionMatrixAlgorithmMatrix(HFconnectionGraphObject, readSavedConnectionsMatrixAlgorithm, connectionMatrixAlgorithmSplit, dtype, HFNLPpy_ConnectionMatrixAlgorithm.createIndexStringDendriticBranch(dendriticBranchIndex), HFNLPpy_ConnectionMatrixAlgorithm.createIndexStringSecondDataIndex(secondDataIndex))
					if(connectionMatrixAlgorithmNormaliseStore):
						HFconnectionGraphMatrixNormalised[dendriticBranchIndex][secondDataIndex] = HFNLPpy_ConnectionMatrixAlgorithm.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex])
	return HFconnectionGraphMatrix, HFconnectionGraphMatrixNormalised
	
def seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences, HFconnectionGraphObject):
	
	targetSentenceConceptNodeList = seedSentenceConceptNodeList

	initialiseHFconnectionMatrixWrapperAlgorithmMatrixActivationRecord(HFconnectionGraphObject)

	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	if(not seedHFnetworkSubsequenceBasic):
		conceptNeuronSourceList = []
	if(printPredictionsSentence):
		predictedSentenceConceptNodeList = []
		predictedSentenceConceptNodeList.append(seedSentenceConceptNodeList[0])

	numberSourceNeurons = len(targetSentenceConceptNodeList)-1
	for wSource in range(numberSourceNeurons):
		
		#forward predictions (if reversePredictions: predict future candidates);
		if(reversePredictions):	#predictFutureCandidates
			numberTargetNeurons = len(targetSentenceConceptNodeList)-wSource-1
		else:
			numberTargetNeurons = 1
		connectionTargetNeuronSetLocalForward = set()
		for wTarget in range(wSource+1, wSource+1+numberTargetNeurons):
			somaActivationFound, connectionTargetNeuronSetLocalForwardTarget = performPredictions(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, conceptNeuronSourceList, wTarget, wSource, connectionTargetNeuronSet, HFconnectionGraphObject, False)	#connectionTargetNeuronSet parameter is not used during forward predictions only
			connectionTargetNeuronSetLocalForward = connectionTargetNeuronSetLocalForward.union(connectionTargetNeuronSetLocalForwardTarget)
		
		#forward+reverse predictions;
		wTarget = wSource+1
		conceptNeuronTarget = targetSentenceConceptNodeList[wTarget]
		if(reversePredictions):
			somaActivationFound, connectionTargetNeuronSetLocal = performPredictions(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, conceptNeuronSourceList, wTarget, wSource, connectionTargetNeuronSetLocalForward, HFconnectionGraphObject, True)
		else:
			connectionTargetNeuronSetLocal = connectionTargetNeuronSetLocalForward	#or connectionTargetNeuronSetLocalForwardTarget (as numberTargetNeurons = 1)
			#print("connectionTargetNeuronSetLocal = ", list(connectionTargetNeuronSetLocal)[0].nodeName)
				
		if(seedHFnetworkSubsequenceBasic):
			connectionTargetNeuronSet = connectionTargetNeuronSetLocal
			if(printPredictionsSentence):
				predictedSentenceConceptNodeList.append(conceptNeuronTarget)
		else:
			connectionTargetNeuronSetLocalFiltered = selectActivatedNeurons(wSource, targetSentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSetLocal, HFconnectionGraphObject)
			#print("connectionTargetNeuronSetLocalForward = ", list(connectionTargetNeuronSetLocalForward))
			#print("connectionTargetNeuronSetLocal = ", list(connectionTargetNeuronSetLocal))
			conceptNeuronSourceList.clear()
			for connectionTargetNeuron in connectionTargetNeuronSetLocalFiltered:
				if(printPredictions):
					print("\tconceptNeuronSourceList.append connectionTargetNeuron = ", connectionTargetNeuron.nodeName)
				conceptNeuronSourceList.append(connectionTargetNeuron)
			connectionTargetNeuronSet = connectionTargetNeuronSet.union(connectionTargetNeuronSetLocal)
			if(printPredictionsSentence):
				if(wSource < seedHFnetworkSubsequenceLength):
					predictedSentenceConceptNodeList.append(conceptNeuronTarget)
				else:
					predictedSentenceConceptNodeList.append(list(connectionTargetNeuronSetLocalFiltered)[0])	#only print first prediction per word index

		expectPredictiveSequenceToBeFound = False
		if(enforceMinimumEncodedSequenceLength):
			if(wSource >= minimumEncodedSequenceLength-1):
				expectPredictiveSequenceToBeFound = True
		else:
			expectPredictiveSequenceToBeFound = True
		if(expectPredictiveSequenceToBeFound):
			if(somaActivationFound):
				if(printPredictions):
					print("somaActivationFound")
				if(seedHFnetworkSubsequenceType=="all"):
					global feedPredictionSuccesses
					feedPredictionSuccesses += 1
			else:
				if(printPredictions):
					print("!somaActivationFound: HFNLP algorithm error detected")
				if(seedHFnetworkSubsequenceType=="all"):
					global feedPredictionErrors
					feedPredictionErrors += 1
		else:
			if(printPredictions):
				print("!expectPredictiveSequenceToBeFound: wSource < minimumEncodedSequenceLength-1")

		if(temporarilyWeightInContextAssociations):	
			#CHECKTHIS currently use predictionBidirectionalContext as only seeded words have (limited) known future context
			addContextWordsToConnectionGraphMatrixW(wSource, networkConceptNodeDict, predictedSentenceConceptNodeList, HFconnectionGraphObject, predictionBidirectionalContext, weightEnduranceType=weightEnduranceTypeInContextAdd)

	if(temporarilyWeightInContextAssociations):	
		for wSource in range(numberSourceNeurons):
			addContextWordsToConnectionGraphMatrixW(wSource, networkConceptNodeDict, predictedSentenceConceptNodeList, HFconnectionGraphObject, predictionBidirectionalContext, weightEnduranceType=weightEnduranceTypeInContextSubtract)
	
	if(printPredictionsSentence):
		targetSentenceText = ""
		predictedSentenceText = ""
		for wSource, conceptNeuronTarget in enumerate(targetSentenceConceptNodeList):
			targetSentenceText = targetSentenceText + " " + conceptNeuronTarget.nodeName
			predictedSentenceText = predictedSentenceText + " " + predictedSentenceConceptNodeList[wSource].nodeName
		print("targetSentenceText    = ", targetSentenceText)
		print("predictedSentenceText = ", predictedSentenceText)
			
	if(HFconnectionMatrixAlgorithmSplitDatabase):
		neuronIDalreadySaved = {}
		connectionTargetNeuronList = list(connectionTargetNeuronSet)
		HFNLPpy_MatrixDatabase.finaliseMatrixDatabaseSentence(HFconnectionGraphObject, connectionTargetNeuronList, neuronIDalreadySaved)

def performPredictions(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, conceptNeuronSourceList, wTarget, wSource, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions):
	#print("wSource = ", wSource)
	#print("wTarget = ", wTarget)
	
	conceptNeuronSource = targetSentenceConceptNodeList[wSource]
	conceptNeuronTarget = targetSentenceConceptNodeList[wTarget]

	if(algorithmMatrixSANImethod=="completeSANI"):
		HFconnectionGraphObject.activationTime = wSource
					
	connectionTargetNeuronSetLocal = set()
	if(seedHFnetworkSubsequenceBasic):
		if(printPredictions):
			print("\nseedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName, ", seedSource = ", conceptNeuronSource.nodeName)
		activationTime = calculateActivationTimeSequence(wSource)
		somaActivationFound, connectionTargetNeuronSetLocal = simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, activationTime, wTarget, conceptNeuronTarget, wSource, conceptNeuronSource, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions)
	else:
		activationTime = None
		if(wSource < seedHFnetworkSubsequenceLength):
			if(printPredictions):
				print("\nseedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName, ", seedSource = ", conceptNeuronSource.nodeName)
			somaActivationFound, connectionTargetNeuronSetLocal = simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, activationTime, wTarget, conceptNeuronTarget, wSource, conceptNeuronSource, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions)
		else:
			if(printPredictions):
				print("")
				for feedSource in conceptNeuronSourceList:
					print("feedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName, ", feedSource = ", feedSource.nodeName)
			somaActivationFound, connectionTargetNeuronSetLocal = simulateBiologicalHFnetworkSequenceNodesPropagate(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, activationTime, wTarget, conceptNeuronTarget, wSource, conceptNeuronSourceList, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions)
	return somaActivationFound, connectionTargetNeuronSetLocal
			
def simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget, conceptNeuronTarget, wSource, conceptNeuronSource, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions):
	#print("simulateBiologicalHFnetworkSequenceNodePropagateForward: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)	
	somaActivationFound, connectionTargetNeuronSetLocal = HFNLPpy_MatrixPropagate.simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions)
	return somaActivationFound, connectionTargetNeuronSetLocal

def simulateBiologicalHFnetworkSequenceNodesPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget, conceptNeuronTarget, wSource, conceptNeuronSourceList, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions):
	somaActivationFound, connectionTargetNeuronSetLocal = HFNLPpy_MatrixPropagate.simulateBiologicalHFnetworkSequenceNodesPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions)
	return somaActivationFound, connectionTargetNeuronSetLocal

'''
#independent method (does not need to be executed in order of wSource)
def simulateBiologicalHFnetworkSequenceNodePropagateFull(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
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

def selectActivatedNeurons(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject=None):
	connectionTargetNeuronSetLocalFiltered = connectionTargetNeuronSet
	if(linkSimilarConceptNodes):
		connectionTargetNeuronSetLocalFiltered = retrieveSimilarConcepts(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSetLocalFiltered, HFconnectionGraphObject)
	return connectionTargetNeuronSetLocalFiltered
	
	
#connection graph generation;

def addContextWordsToConnectionGraphMatrix(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject):
	for w1, token1 in enumerate(tokenisedSentence):
		#print("w1 = ", w1)
		addContextWordsToConnectionGraphMatrixW(w1, networkConceptNodeDict, sentenceConceptNodeList, HFconnectionGraphObject, formationBidirectionalContext, weightEnduranceTypeDefaultAdd)

def addContextWordsToConnectionGraphMatrixW(w1, networkConceptNodeDict, sentenceConceptNodeList, HFconnectionGraphObject, bidirectionalContext, weightEnduranceType=weightEnduranceTypeDefaultAdd):
	#print("w1 = ", w1)
	conceptNode = sentenceConceptNodeList[w1]
	#print("addContextWordsToConnectionGraphMatrixW: conceptNode.nodeName = ", conceptNode.nodeName) 
	neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
	secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax(getContextSizeSource=True, wSource=w1)
	closestDendriticBranchIndex = calculateDendriticBranchClosestIndex(networkConceptNodeDict, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1, bidirectionalContext)
	if(not closestDendriticBranchIndex==-1):
		for secondDataIndex in range(secondDataIndexMax):
			#print("contextSizeIndex = ", contextSizeIndex)
			addContextWordsToConnectionGraphNeuronID(w1, neuronID, sentenceConceptNodeList, HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex, None, contextMatrixWeightStore, bidirectionalContext, weightEnduranceType)

def calculateDendriticBranchClosestIndex(networkConceptNodeDict, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1, bidirectionalContext):
	if(numberOfIndependentDendriticBranches == 1):
		closestDendriticBranchIndex = 0
	else:
		if(simulatedDendriticBranchesIndependentProximalContext):
			foundClosestBranchIndex = False
			closestDendriticBranchIndex = 0
			if(w1 > 0):
				sourceNode = sentenceConceptNodeList[w1-1]	#proximalContextIndex
				sourceNeuronID = HFconnectionGraphObject.neuronIDdict[sourceNode.nodeName]	#proximalContextNeuronID
				targetNode = sentenceConceptNodeList[w1]
				targetNeuronID = HFconnectionGraphObject.neuronIDdict[targetNode.nodeName]
				for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
					proximalContextNeuronID = HFconnectionGraphObject.simulatedDendriticBranchesProximalContext[dendriticBranchIndex][targetNeuronID]
					if(proximalContextNeuronID == sourceNeuronID):
						#proximalContext has already been assigned to an independentDendriticBranch of the targetNeuronID neuron
						closestDendriticBranchIndex = dendriticBranchIndex
						foundClosestBranchIndex = True
					if(not foundClosestBranchIndex):
						if(proximalContextNeuronID == simulatedDendriticBranchesIndependentProximalContextNeuronIDmissing):
							#select first available independentDendriticBranch to assign proximalContext
							closestDendriticBranchIndex = dendriticBranchIndex
							foundClosestBranchIndex = True
							HFconnectionGraphObject.simulatedDendriticBranchesProximalContext[dendriticBranchIndex][targetNeuronID] = sourceNeuronID
				if(not foundClosestBranchIndex):
					print("sourceNode.nodeName = ", sourceNode.nodeName)
					print("targetNode.nodeName = ", targetNode.nodeName)
					printe("simulatedDendriticBranchesIndependentProximalContext:calculateDendriticBranchClosestIndex error: not enough available simulatedDendriticBranches to store proximalContext; numberOfIndependentDendriticBranches must be increased")
			else:
				pass #no context to predict target neuron ID
		else:
			foundClosestBranchIndex = False
			#dendriticBranchClosestTargetSet = set()
			closestConnectionStrength = 0
			closestDendriticBranchIndex = -1
			if(algorithmMatrixTensorDim==4):
				if(algorithmMatrixSANI or secondDataIndexMax > 0):					
					_, connectionStrength, dendriticBranchIndex = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, None, None, secondDataIndexMax, contextMatrixWeightStore, bidirectionalContext, True)
					foundClosestBranchIndex, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrength)
			else:
				for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
					#print("dendriticBranchIndex = ", dendriticBranchIndex)
					if(algorithmMatrixTensorDim==3):
						_, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, None, secondDataIndexMax, contextMatrixWeightStore, bidirectionalContext, True)
						foundClosestBranchIndex, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrength)
					else:
						for secondDataIndex in range(secondDataIndexMax):
							_, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, None, contextMatrixWeightStore, bidirectionalContext, False)
							if(normaliseConnectionStrengthWrtContextLength):	
								connectionStrengthNormalised = connectionStrength/secondDataIndex
							else:
								connectionStrengthNormalised = connectionStrength
							foundClosestBranchIndex, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrengthNormalised)

			#if(debugAlgorithmMatrix):
			print("closestDendriticBranchIndex = ", closestDendriticBranchIndex)
			if(not foundClosestBranchIndex):
				closestDendriticBranchIndex = random.randint(0, numberOfIndependentDendriticBranches-1)
	return closestDendriticBranchIndex

def addContextWordsToConnectionGraphNeuronID(w1, neuronID, sentenceConceptNodeList, HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, weightEnduranceType=weightEnduranceTypeDefaultAdd):
	contextConnectionVector = HFNLPpy_MatrixOperations.createContextVectorWrapperReverseLookup(w1, sentenceConceptNodeList, HFconnectionGraphObject, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, False, weightEnduranceType=weightEnduranceType)
	HFNLPpy_ConnectionMatrixAlgorithm.addContextConnectionsToGraphNeuronIDWrapper(HFconnectionGraphObject, contextConnectionVector, closestDendriticBranchIndex, secondDataIndex, neuronID)
	if(HFconnectionMatrixAlgorithmNormaliseStore):	#if(not HFconnectionMatrixAlgorithmSplit):
		HFNLPpy_ConnectionMatrixAlgorithm.normaliseBatchedTensorWrapper(HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex)
	
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
		connectionTargetNeuronSetExtended = HFNLPpy_ConceptsMatrixOperations.retrieveSimilarConceptsBagOfWords(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject)

	return connectionTargetNeuronSetExtended

def updateSynapsesMemoryForget(HFconnectionGraphObject, sentenceConceptNodeList):
	if(forgetSynapses):
		if(algorithmMatrixTensorDim==4):
			HFNLPpy_ConnectionMatrixAlgorithm.updateSynapsesMemoryForget(HFconnectionGraphObject.HFconnectionGraphMatrix, sentenceConceptNodeList, HFconnectionMatrixAlgorithmSplit)
		else:
			secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
			for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
				if(algorithmMatrixTensorDim==3):
					HFNLPpy_ConnectionMatrixAlgorithm.updateSynapsesMemoryForget(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex], sentenceConceptNodeList, HFconnectionMatrixAlgorithmSplit)
				else:
					for secondDataIndex in range(secondDataIndexMax):
						 HFNLPpy_ConnectionMatrixAlgorithm.updateSynapsesMemoryForget(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex], sentenceConceptNodeList, HFconnectionMatrixAlgorithmSplit)
