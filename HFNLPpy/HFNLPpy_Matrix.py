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

def HFconnectionGraphMatrixHolderInitialisation(self):
	if(useHFconnectionMatrixBasicSplit):
		if(HFconnectionMatrixBasicSplitRAM):
			self.HFconnectionGraphMatrix = [None]*HFconnectionMatrixBasicMaxConcepts	#store a separate matrix column for each source neuronID context index in python list
			for sourceNeuronID in range(HFconnectionMatrixBasicMaxConcepts):
				self.HFconnectionGraphMatrix[sourceNeuronID] = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
			self.HFconnectionGraphMatrixMin = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()	#for dynamic normalisation (on demand)
			self.HFconnectionGraphMatrixMax = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()	#for dynamic normalisation (on demand)
		else:
			printe("HFconnectionGraphMatrixHolderInitialisation error: !HFconnectionMatrixBasicSplitRAM has not been coded")
	else:
		self.HFconnectionGraphMatrix = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()
		if(HFconnectionMatrixNormaliseRAM):
			self.HFconnectionGraphMatrixNormalised = HFNLPpy_MatrixOperations.createConnectionGraphMatrixHolder()

def initialiseHFconnectionMatrixWrapper(HFconnectionGraphObject):
	if(useAlgorithmMatrix):
		if(useHFconnectionMatrixBasicSplit):
			HFconnectionGraphObject.HFconnectionGraphMatrixMin, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject.HFconnectionGraphMatrixMin)
			HFconnectionGraphObject.HFconnectionGraphMatrixMax, _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject.HFconnectionGraphMatrixMax)
			''' 
			#uses too much RAM (instead dynamically initialise HFconnectionGraphMatrix when new concepts are declared) - see addSentenceConceptNodesToHFconnectionGraphObject
			if(HFconnectionMatrixBasicSplitRAM):
				for sourceNeuronID in range(HFconnectionMatrixBasicMaxConcepts):
					HFconnectionGraphObject.HFconnectionGraphMatrix[sourceNeuronID], _ = initialiseHFconnectionMatrixWrapperAlgorithmMatrix()
			'''
		else:
			HFconnectionGraphObject.HFconnectionGraphMatrix, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised = initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix)
	else:
		HFconnectionGraphObject.HFconnectionGraphBasic = HFNLPpy_ConnectionMatrix.initialiseHFconnectionMatrix()
	HFconnectionGraphObject.neuronNamelist = HFNLPpy_ConnectionMatrix.initialiseNeuronNameList()

def initialiseHFconnectionMatrixWrapperAlgorithmMatrix(HFconnectionGraphMatrix):
	if(not HFconnectionMatrixNormaliseRAM):
		HFconnectionGraphMatrixNormalised=None
	if(algorithmMatrixTensorDim==4):
		HFconnectionGraphMatrix = HFNLPpy_ConnectionMatrix.initialiseHFconnectionMatrix()
		if(HFconnectionMatrixNormaliseRAM):
			HFconnectionGraphMatrixNormalised = HFNLPpy_ConnectionMatrix.normaliseBatchedTensor(HFconnectionGraphMatrix)
	else:
		secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			if(algorithmMatrixTensorDim==3):
				HFconnectionGraphMatrix[dendriticBranchIndex] = HFNLPpy_ConnectionMatrix.initialiseHFconnectionMatrix(HFNLPpy_ConnectionMatrix.createIndexStringDendriticBranch(dendriticBranchIndex))
				if(HFconnectionMatrixNormaliseRAM):
					HFconnectionGraphMatrixNormalised[dendriticBranchIndex] = HFNLPpy_ConnectionMatrix.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex])
			else:
				for secondDataIndex in range(secondDataIndexMax):
					HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex] = HFNLPpy_ConnectionMatrix.initialiseHFconnectionMatrix(HFNLPpy_ConnectionMatrix.createIndexStringDendriticBranch(dendriticBranchIndex), HFNLPpy_ConnectionMatrix.createIndexStringSecondDataIndex(secondDataIndex))
					if(HFconnectionMatrixNormaliseRAM):
						HFconnectionGraphMatrixNormalised[dendriticBranchIndex][secondDataIndex] = HFNLPpy_ConnectionMatrix.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][secondDataIndex])
	return HFconnectionGraphMatrix, HFconnectionGraphMatrixNormalised
	
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
		closestDendriticBranchIndex = calculateDendriticBranchClosestIndex(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1)
		for secondDataIndex in range(secondDataIndexMax):
			#print("contextSizeIndex = ", contextSizeIndex)
			addContextWordsToConnectionGraphNeuronID(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex, None, contextMatrixWeightStore, False, algorithmMatrixTensorDim==4)

def calculateDendriticBranchClosestIndex(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, neuronID, secondDataIndexMax, w1):
	closestDendriticBranchIndex = 0
	if(numberOfIndependentDendriticBranches == 1):
		closestDendriticBranchIndex = 0
	else:
		foundClosest = False
		#dendriticBranchClosestTargetSet = set()
		closestConnectionStrength = 0
		closestDendriticBranchIndex = 0
		if(algorithmMatrixTensorDim==4):
			if(algorithmMatrixSANI or secondDataIndexMax > 0):					
				_, connectionStrength, dendriticBranchIndex = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, None, None, secondDataIndexMax, contextMatrixWeightStore, False, True)
				foundClosest, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosest, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrength)
		else:
			for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
				#print("dendriticBranchIndex = ", dendriticBranchIndex)
				if(algorithmMatrixTensorDim==3):
					_, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, None, secondDataIndexMax, contextMatrixWeightStore, False, True)
					foundClosest, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosest, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrength)
				else:
					for secondDataIndex in range(secondDataIndexMax):
						_, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, None, contextMatrixWeightStore, False, False)
						if(normaliseConnectionStrengthWrtContextLength):	
							connectionStrengthNormalised = connectionStrength/secondDataIndex
						else:
							connectionStrengthNormalised = connectionStrength
						foundClosest, _, closestConnectionStrength, closestDendriticBranchIndex = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosest, None, closestConnectionStrength, closestDendriticBranchIndex, None, connectionStrength, dendriticBranchIndex, True, connectionStrengthNormalised)

		#if(debugAlgorithmMatrix):
		print("closestDendriticBranchIndex = ", closestDendriticBranchIndex)
		if(not foundClosest):
			closestDendriticBranchIndex = random.randint(0, numberOfIndependentDendriticBranches-1)
	return closestDendriticBranchIndex

def addContextWordsToConnectionGraphNeuronID(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4):
	contextConnectionVector = HFNLPpy_MatrixOperations.createContextVectorWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, False)
	if(HFconnectionMatrixGPU):
		contextConnectionVector = contextConnectionVector.to(HFNLPpy_ConnectionMatrix.device)
	HFNLPpy_ConnectionMatrix.addContextConnectionsToGraphNeuronIDWrapper(HFconnectionGraphObject, contextConnectionVector, closestDendriticBranchIndex, secondDataIndex, neuronID, matrixTensorDim4)
	if(not useHFconnectionMatrixBasicSplit):
		HFNLPpy_ConnectionMatrix.normaliseBatchedTensorWrapper(HFconnectionGraphObject, closestDendriticBranchIndex, secondDataIndex, matrixTensorDim4)

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
