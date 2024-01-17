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
												
def HFconnectionGraphMatrixHolderInitialisation(self):
	if(HFconnectionMatrixAlgorithmSplit):
		self.HFconnectionGraphMatrix = [None]*HFconnectionMatrixBasicMaxConcepts	#store a separate matrix column for each source neuronID context index in python list
		for sourceNeuronID in range(HFconnectionMatrixBasicMaxConcepts):
			self.HFconnectionGraphMatrix[sourceNeuronID] = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
		self.HFconnectionGraphMatrixMin = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()	#for dynamic normalisation (on demand)
		self.HFconnectionGraphMatrixMax = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()	#for dynamic normalisation (on demand)
		if(HFconnectionMatrixAlgorithmSplitDatabase):
			print("HFconnectionGraphMatrixHolderInitialisation warning: !HFconnectionMatrixAlgorithmSplitDatabase requires check")
	else:
		self.HFconnectionGraphMatrix = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
		if(HFconnectionMatrixAlgorithmNormaliseStore):
			self.HFconnectionGraphMatrixNormalised = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()

def initialiseHFconnectionMatrixAlgorithmWrapper(HFconnectionGraphObject):
	HFNLPpy_ConnectionMatrixOperations.initialiseNeuronNameList(HFconnectionGraphObject)
	if(HFconnectionMatrixAlgorithmSplit):
		HFconnectionGraphObject.HFconnectionGraphMatrixMin, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrixMin, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
		HFconnectionGraphObject.HFconnectionGraphMatrixMax, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrixMax, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
		''' 
		#uses too much RAM (instead dynamically initialise HFconnectionGraphMatrix when new concepts are declared) - see addSentenceConceptNodesToHFconnectionGraphObject
		if(HFconnectionMatrixAlgorithmSplitRAM):
			for sourceNeuronID in range(HFconnectionMatrixBasicMaxConcepts):
				HFconnectionGraphObject.HFconnectionGraphMatrix[sourceNeuronID], _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore)
		'''
	else:
		HFconnectionGraphObject.HFconnectionGraphMatrix, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix, HFreadSavedConnectionsMatrixAlgorithm, HFconnectionMatrixAlgorithmSplit, HFconnectionMatrixAlgorithmNormaliseStore, HFconnectionsMatrixAlgorithmType)
	initialiseHFconnectionMatrixWrapperAlgorithmMatrixActivationRecord(HFconnectionGraphObject)

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

	for wSource in range(len(targetSentenceConceptNodeList)-1):
		wTarget = wSource+1
		conceptNeuronSource = targetSentenceConceptNodeList[wSource]
		conceptNeuronTarget = targetSentenceConceptNodeList[wTarget]
		
		if(algorithmMatrixSANImethod=="completeSANI"):
			HFconnectionGraphObject.activationTime = wSource
		
		if(seedHFnetworkSubsequenceBasic):
			if(printPredictions):
				print("\nseedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName, ", seedSource = ", conceptNeuronSource.nodeName)
			activationTime = calculateActivationTimeSequence(wSource)
			somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet, HFconnectionGraphObject)
			if(printPredictionsSentence):
				predictedSentenceConceptNodeList.append(conceptNeuronTarget)
		else:
			connectionTargetNeuronSetLocal = set()
			activationTime = None
			if(wSource < seedHFnetworkSubsequenceLength):
				if(printPredictions):
					print("\nseedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName, ", seedSource = ", conceptNeuronSource.nodeName)
				somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSetLocal, HFconnectionGraphObject)
			else:
				if(printPredictions):
					print("")
					for feedSource in conceptNeuronSourceList:
						print("feedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName, ", feedSource = ", feedSource.nodeName)
				somaActivationFound = simulateBiologicalHFnetworkSequenceNodesPropagate(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSourceList, connectionTargetNeuronSetLocal, HFconnectionGraphObject)
			
			connectionTargetNeuronSetLocalFiltered = selectActivatedNeurons(wSource, targetSentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSetLocal, HFconnectionGraphObject)
				
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

def simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet, HFconnectionGraphObject):
	#print("simulateBiologicalHFnetworkSequenceNodePropagateForward: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)	
	somaActivationFound = HFNLPpy_MatrixPropagate.simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject)
	return somaActivationFound

def simulateBiologicalHFnetworkSequenceNodesPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSourceList, connectionTargetNeuronSet, HFconnectionGraphObject):
	somaActivationFound = HFNLPpy_MatrixPropagate.simulateBiologicalHFnetworkSequenceNodesPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject)
	return somaActivationFound

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
		conceptNode = sentenceConceptNodeList[w1]
		#print("addContextWordsToConnectionGraphMatrix: conceptNode.nodeName = ", conceptNode.nodeName) 
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax(getContextSizeSource=True, wSource=w1)
		closestDendriticBranchIndex = calculateDendriticBranchClosestIndex(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1)
		if(not closestDendriticBranchIndex==-1):
			for secondDataIndex in range(secondDataIndexMax):
				#print("contextSizeIndex = ", contextSizeIndex)
				addContextWordsToConnectionGraphNeuronID(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex, None, contextMatrixWeightStore, False)

def calculateDendriticBranchClosestIndex(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1):
	if(numberOfIndependentDendriticBranches == 1):
		closestDendriticBranchIndex = 0
	else:
		foundClosestBranchIndex = False
		#dendriticBranchClosestTargetSet = set()
		closestConnectionStrength = 0
		closestDendriticBranchIndex = -1
		if(algorithmMatrixTensorDim==4):
			if(algorithmMatrixSANI or secondDataIndexMax > 0):					
				_, connectionStrength, dendriticBranchIndex = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, None, None, secondDataIndexMax, contextMatrixWeightStore, False, True)
				foundClosestBranchIndex, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrength)
		else:
			for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
				#print("dendriticBranchIndex = ", dendriticBranchIndex)
				if(algorithmMatrixTensorDim==3):
					_, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, None, secondDataIndexMax, contextMatrixWeightStore, False, True)
					foundClosestBranchIndex, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrength)
				else:
					for secondDataIndex in range(secondDataIndexMax):
						_, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, None, contextMatrixWeightStore, False, False)
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

def addContextWordsToConnectionGraphNeuronID(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext):
	contextConnectionVector = HFNLPpy_MatrixOperations.createContextVectorWrapperReverseLookup(w1, sentenceConceptNodeList, HFconnectionGraphObject, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, False)
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
