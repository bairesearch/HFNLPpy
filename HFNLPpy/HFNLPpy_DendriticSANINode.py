"""HFNLPpy_DendriticSANINode.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI Node Classes

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import random

from HFNLPpy_DendriticSANIGlobalDefs import *

#currently used for HFNLPpy_DendriticSANIDraw:getActivationColor only;
objectTypeConceptNeuron = 1
objectTypeDendriticBranch = 2
objectTypeSequentialSegment = 3
objectTypeSequentialSegmentInput = 4
		
def dendriticSANINodePropertiesInitialisation(conceptNode):

	conceptNode.objectType = objectTypeConceptNeuron
	
	conceptNode.activationLevel = objectAreaActivationLevelOff	#currently only used by drawBiologicalSimulationDynamic

	conceptNode.activationTimeWord = None

	#if(biologicalSimulationDraw):
	#required to assign independent names to each;
	conceptNode.currentBranchIndexNeuron = 0
	conceptNode.currentSequentialSegmentIndexNeuron = 0
	conceptNode.currentSequentialSegmentInputIndexNeuron = 0 

	if(vectoriseComputationCurrentDendriticInput):
		if(recordVectorisedBranchObjectList):
			conceptNode.vectorisedBranchActivationLevelList, conceptNode.vectorisedBranchActivationTimeList, conceptNode.vectorisedBranchActivationFlagList, conceptNode.vectorisedBranchObjectList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=recordVectorisedBranchObjectList, storeSequentialSegmentInputActivationLevels=False)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
			#!performSummationOfSequentialSegmentInputsAcrossBranch: vectorisedBranchActivationLevelList stores effective boolean 1/0 values (since their activations are calculated across all wSource of same activationTime simultaneously); could be converted to dtype=tf.bool
			#weightedSequentialSegmentInputs:vectorisedBranchActivationLevelListBuffer will store numeric values of the synaptic input activation levels being accumulated prior to batch processing			
		else:
			conceptNode.vectorisedBranchActivationLevelList, conceptNode.vectorisedBranchActivationTimeList, conceptNode.vectorisedBranchActivationFlagList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=recordVectorisedBranchObjectList, storeSequentialSegmentInputActivationLevels=False)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
		#vectorisedBranchObjectList required for drawBiologicalSimulationDynamic only

	conceptNode.dendriticTree = createDendriticTree(conceptNode, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)

class DendriticBranch:
	def __init__(self, conceptNode, parentBranch, numberOfBranchSequentialSegments, branchIndex1, branchIndex2, horizontalBranchIndex):
		self.objectType = objectTypeDendriticBranch
		self.parentBranch = parentBranch
		self.subbranches = []
		
		#if(biologicalSimulationDraw):
		self.nodeName = generateDendriticBranchName(conceptNode)
		self.branchIndex1 = branchIndex1
		self.branchIndex2 = branchIndex2	#local horizontalBranchIndex (wrt horizontalBranchWidth)
		#print("horizontalBranchIndex = ", horizontalBranchIndex)
		self.horizontalBranchIndex = horizontalBranchIndex	#absolute horizontalBranchIndex	#required by vectoriseComputationCurrentDendriticInput only
		self.conceptNode = conceptNode

		self.sequentialSegments = [SequentialSegment(conceptNode, self, i) for i in range(numberOfBranchSequentialSegments)]	#[SequentialSegment(conceptNode, self, i)]*numberOfBranchSequentialSegments	#indexed from most proximal to most distal
		self.activationLevel = objectAreaActivationLevelOff
		self.activationTime = None	#within sequence/sentence activation time
		if(drawBiologicalSimulationDynamicHighlightNewActivations):
			self.activationStateNew = False
						
class SequentialSegment:
	def __init__(self, conceptNode, branch, sequentialSegmentIndex):
		#self.inputs = []
		self.objectType = objectTypeSequentialSegment
		self.inputs = {}
		self.activationLevel = objectLocalActivationLevelOff	#only consider depolarised if activationLevel passes threshold
		self.activationTime = None	#within sequence/sentence activation time
		if(drawBiologicalSimulationDynamicHighlightNewActivations):
			self.activationStateNew = False
		self.branch = branch
		self.sequentialSegmentIndex = sequentialSegmentIndex 

		#if(biologicalSimulationDraw):
		self.nodeName = generateSequentialSegmentName(conceptNode)
		self.conceptNode = conceptNode	#not required (as can lookup SequentialSegment.branch.conceptNode) 
			
		if(vectoriseComputation):
			if(recordVectorisedBranchObjectList):
				#print("recordVectorisedBranchObjectList: branch.branchIndex1 = ", branch.branchIndex1, "branch.branchIndex2 = ", branch.branchIndex2, "branch.horizontalBranchIndex = ", branch.horizontalBranchIndex, "sequentialSegmentIndex = ", sequentialSegmentIndex)
				conceptNode.vectorisedBranchObjectList[branch.branchIndex1][branch.horizontalBranchIndex, branch.branchIndex2, sequentialSegmentIndex] = self
				#print("conceptNode.vectorisedBranchObjectList[branch.branchIndex1][branch.horizontalBranchIndex, branch.branchIndex2, sequentialSegmentIndex].nodeName = ", conceptNode.vectorisedBranchObjectList[branch.branchIndex1][branch.horizontalBranchIndex, branch.branchIndex2, sequentialSegmentIndex].nodeName)
		if(overwriteSequentialSegments):
			self.frozen = False
				
class SequentialSegmentInput:
	def __init__(self, conceptNode, SequentialSegment, sequentialSegmentInputIndex, nodeSource):
		self.objectType = objectTypeSequentialSegmentInput
		self.input = None
		self.sequentialSegment = SequentialSegment
		self.firstInputInSequence = False	#within sequence/sentence activation time
		
		if(recordSequentialSegmentInputActivationLevels):
			self.activationLevel = objectLocalActivationLevelOff	#input has been temporarily triggered for activation (only affects dendritic signal if sequentiality requirements met)
			self.activationTime = None	#numeric	#input has been temporarily triggered for activation (only affects dendritic signal if sequentiality requirements met)
			self.sequentialSegmentInputIndex = sequentialSegmentInputIndex
			if(drawBiologicalSimulationDynamicHighlightNewActivations):
				self.activationStateNew = False
			
		#if(preventGenerationOfDuplicateConnections):
		self.nodeSource = nodeSource
			
		#if(biologicalSimulationDraw):
		self.nodeName = generateSequentialSegmentInputName(conceptNode)
		if(storeSequentialSegmentInputIndexValues):
			self.sequentialSegmentInputIndex = sequentialSegmentInputIndex	#not required	#index record value not robust if inputs are removed (synaptic atrophy)
		else:
			self.sequentialSegmentInputIndex = None
		self.conceptNode = conceptNode	#not required (as can lookup SequentialSegment.branch.conceptNode)
		




#if(vectoriseComputationCurrentDendriticInput):
								
def createDendriticTreeVectorised(batched, createVectorisedBranchObjectList, storeSequentialSegmentInputActivationLevels):
	vectorisedBranchActivationLevelList = []	#list of tensors for every branchIndex1	- each element is of shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
	vectorisedBranchActivationTimeList = []
	vectorisedBranchActivationFlagList = []
	if(createVectorisedBranchObjectList):
		vectorisedBranchObjectList = []
	for currentBranchIndex1 in range(calculateNumberOfVerticalBranches(numberOfBranches1)):
		numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(currentBranchIndex1, numberOfBranches2)
		#tf.Variable designation is required for assign() operations
		if(batched):
			if(storeSequentialSegmentInputActivationLevels):
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationFlag = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs), dtype=object)			
			else:
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationFlag = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments), dtype=object)
		else:
			if(storeSequentialSegmentInputActivationLevels):
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationFlag = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs), dtype=object)			
			else:
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))		
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationFlag = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments), dtype=object)
		vectorisedBranchActivationLevelList.append(vectorisedBranchActivationLevel)
		vectorisedBranchActivationTimeList.append(vectorisedBranchActivationTime)
		vectorisedBranchActivationFlagList.append(vectorisedBranchActivationFlag)
		if(createVectorisedBranchObjectList):
			vectorisedBranchObjectList.append(vectorisedBranchObject)
				
	if(createVectorisedBranchObjectList):
		return vectorisedBranchActivationLevelList, vectorisedBranchActivationTimeList, vectorisedBranchActivationFlagList, vectorisedBranchObjectList	
	else:
		return vectorisedBranchActivationLevelList, vectorisedBranchActivationTimeList, vectorisedBranchActivationFlagList

def printVectorisedBranchObjectList(conceptNode):
	print("printVectorisedBranchObjectList: conceptNode = ", conceptNode.nodeName)
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
	for branchIndex1 in reversed(range(numberOfVerticalBranches)):
		vectorisedBranchObjectSequentialSegment = conceptNode.vectorisedBranchObjectList[branchIndex1]
		print("vectorisedBranchObjectSequentialSegment = ", vectorisedBranchObjectSequentialSegment)
		for horizontalBranchIndex in range(vectorisedBranchObjectSequentialSegment.shape[0]):
			for branchIndex2 in range(vectorisedBranchObjectSequentialSegment.shape[1]):
				for sequentialSegmentIndex in range(vectorisedBranchObjectSequentialSegment.shape[2]):
					print("branchIndex1 = ", branchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex, ", branchIndex2 = ", branchIndex2, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
					sequentialSegment = vectorisedBranchObjectSequentialSegment[horizontalBranchIndex, branchIndex2, sequentialSegmentIndex]
					print("sequentialSegment.nodeName = ", sequentialSegment.nodeName)
					


def calculateNumberOfHorizontalBranches(currentBranchIndex1, numberOfBranches2):
	if(currentBranchIndex1 <= 1):
		numberOfHorizontalBranches = 1
	else:
		numberOfHorizontalBranches = pow(numberOfBranches2, currentBranchIndex1-1)
	
	horizontalBranchWidth = numberOfBranches2
	if(currentBranchIndex1 == 0):
		horizontalBranchWidth = 1	#first branch has single width
	
	return numberOfHorizontalBranches, horizontalBranchWidth

def calculateNumberOfVerticalBranches(numberOfBranches1):
	if(expectFirstBranchSequentialSegmentConnectionStrictNumBranches1):
		numberOfVerticalBranches = numberOfBranches1
	else:
		numberOfVerticalBranches = numberOfBranches1+1
	return numberOfVerticalBranches
		
def createDendriticTree(conceptNode, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments):	
	currentBranchIndex1, currentBranchIndex2, horizontalBranchIndex = (0, 0, 0)
	dendriticTreeHeadBranch = DendriticBranch(conceptNode, None, numberOfBranchSequentialSegments, currentBranchIndex1, currentBranchIndex2, horizontalBranchIndex)
	createDendriticTreeBranch(conceptNode, dendriticTreeHeadBranch, currentBranchIndex1+1, horizontalBranchIndex, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)	#execution update: set currentBranchIndex1+1
	return dendriticTreeHeadBranch
			
def createDendriticTreeBranch(conceptNode, currentBranch, currentBranchIndex1, horizontalBranchIndex, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments):
	#printIndentation(currentBranchIndex1)
	#print("currentBranchIndex1 = ", currentBranchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex)
	if(not expectFirstBranchSequentialSegmentConnectionStrictNumBranches1):
		currentBranch.subbranches = [DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, i, horizontalBranchIndex) for i in range(numberOfBranches2)]	#[DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, currentBranchIndex2)]*numberOfBranches2
	if(currentBranchIndex1 < numberOfBranches1):
		if(expectFirstBranchSequentialSegmentConnectionStrictNumBranches1):
			currentBranch.subbranches = [DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, i, horizontalBranchIndex) for i in range(numberOfBranches2)]	#[DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, currentBranchIndex2)]*numberOfBranches2
		for subBranchIndex2, subbranch in enumerate(currentBranch.subbranches):
			createDendriticTreeBranch(conceptNode, subbranch, currentBranchIndex1+1, horizontalBranchIndex*numberOfBranches2+subBranchIndex2, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)
	
	
#if(biologicalSimulationDraw):
def generateDendriticBranchName(conceptNode):
	nodeName = conceptNode.nodeName + "branch" + str(conceptNode.currentBranchIndexNeuron)
	conceptNode.currentBranchIndexNeuron += 1
	return nodeName
	
def generateSequentialSegmentName(conceptNode):
	nodeName = conceptNode.nodeName + "segment" + str(conceptNode.currentSequentialSegmentIndexNeuron)
	conceptNode.currentSequentialSegmentIndexNeuron += 1
	return nodeName
	
def generateSequentialSegmentInputName(conceptNode):
	nodeName = conceptNode.nodeName + "segmentInput" + str(conceptNode.currentSequentialSegmentInputIndexNeuron)
	conceptNode.currentSequentialSegmentInputIndexNeuron += 1
	return nodeName


def resetConnectionTargetNeurons(connectionTargetNeuronSet, duringSourcePropagation, conceptNeuronTarget=None):
	if(biologicalSimulationForward):		
		if(duringSourcePropagation):
			if(resetConnectionTargetNeuronDendriteDuringActivation):
				resetConnectionTargetNeuronsBasic(connectionTargetNeuronSet, duringSourcePropagation)
			if(resetConnectionTargetNeuronDendriteAfterActivation):
				resetConnectionTargetNeuronsBasic(connectionTargetNeuronSet, duringSourcePropagation)
			if(resetTargetNeuronDendriteAfterActivation):
				resetDendriticTreeActivation(conceptNeuronTarget)
		else:
			#if(not resetConnectionTargetNeuronDendriteAfterActivation):
			resetConnectionTargetNeuronsBasic(connectionTargetNeuronSet, duringSourcePropagation)	#reset all encountered connectionTargets irrespective of connectionTargetNeuron.activationLevel after sentence sourcePropagation complete
			
def resetConnectionTargetNeuronsBasic(connectionTargetNeuronSet, duringSourcePropagation):
	for connectionTargetNeuron in connectionTargetNeuronSet:
		if(duringSourcePropagation):
			if(connectionTargetNeuron.activationLevel):
				#print("resetConnectionTargetNeuronsBasic: connectionTargetNeuron.activationLevel = ", connectionTargetNeuron.nodeName)
				if(resetConnectionTargetNeuronDendriteDuringActivation):
					resetDendriticTreeLastSequentialSegmentActivation(connectionTargetNeuron)
					if(resetConnectionTargetNeuronDendriteDuringActivationFreezeUntilRoundCompletion):
						unfreezeDendriticTreeActivation(connectionTargetNeuron.dendriticTree)
				else:
					resetDendriticTreeActivation(connectionTargetNeuron)
		else:
			resetDendriticTreeActivation(connectionTargetNeuron)
			if(overwriteSequentialSegmentsAfterPropagatingSignal):
				unfreezeDendriticTreeActivation(connectionTargetNeuron.dendriticTree)
	connectionTargetNeuronSet.clear()
		
def resetSourceNeuronAfterActivation(conceptNeuronSource):
	if(resetSourceNeuronAxonAfterActivation):
		resetAxonsActivation(conceptNeuronSource)
	if(resetSourceNeuronDendriteAfterActivation):
		resetDendriticTreeActivation(conceptNeuronSource, updateDendriticTreeObjects=updateNeuronObjectActivationLevels)

def resetDendriticTreeActivation(conceptNeuron, updateDendriticTreeObjects=True):
	conceptNeuron.activationLevel = objectAreaActivationLevelOff
	if(updateDendriticTreeObjects):
		resetBranchActivationRecurse(conceptNeuron.dendriticTree)
	if(vectoriseComputationCurrentDendriticInput):
		resetDendriticTreeActivationVectorised(conceptNeuron)
	
def resetDendriticTreeActivationVectorised(conceptNeuron):
	conceptNeuron.activationLevel = objectAreaActivationLevelOff
	conceptNeuron.vectorisedBranchActivationLevelList, conceptNeuron.vectorisedBranchActivationTimeList,  conceptNeuron.vectorisedBranchActivationFlagList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=False, storeSequentialSegmentInputActivationLevels=False)	#rezero tensors by regenerating them 	#do not overwrite conceptNeuron.vectorisedBranchObjectList

def resetAxonsActivation(conceptNeuron):
	conceptNeuron.activationLevel = objectAreaActivationLevelOff
	for targetConnectionConceptName, connectionList in conceptNeuron.HFcontextTargetConnectionMultiDict.items():
		resetAxonsActivationConnectionList(connectionList)

def resetAxonsActivationConnectionList(connectionList):
	for connection in connectionList:
		connection.activationLevel = objectAreaActivationLevelOff

def resetBranchActivationRecurse(currentBranch):
	resetBranchActivation(currentBranch)
	for subbranch in currentBranch.subbranches:	
		resetBranchActivationRecurse(subbranch)

def resetBranchActivation(currentBranch):
	currentBranch.activationLevel = objectAreaActivationLevelOff
	for sequentialSegment in currentBranch.sequentialSegments:
		resetSequentialSegmentActivation(sequentialSegment)

def resetSequentialSegmentActivation(sequentialSegment):
	sequentialSegment.activationLevel = objectLocalActivationLevelOff
	if(recordSequentialSegmentInputActivationLevels):
		for sequentialSegmentInput in sequentialSegment.inputs.values():
			resetSequentialSegmentInputActivation(sequentialSegmentInput)

def resetSequentialSegmentInputActivation(sequentialSegmentInput):	
	sequentialSegmentInput.activationLevel = objectLocalActivationLevelOff
			
						
								
def printIndentation(level):
	for indentation in range(level):
		print('\t', end='')

def verifyReactivationTime(currentSequentialSegment, activationTime):
	repolarised = False
	if(calculateSequentialSegmentActivationState(currentSequentialSegment.activationLevel)):
		if(verifyRepolarisationTime):
			if(activationTime >= currentSequentialSegment.activationTime+activationRepolarisationTime):
				#do not reactivate sequential segment if already activated by same source neuron
				repolarised = True
		else:
			repolarised = True	
	else:
		repolarised = True
	activationTimeVerified = repolarised
	return activationTimeVerified
	
	
def verifySequentialActivationTime(activationTime, previousSequentialSegmentActivationTime):
	sequentiality = False
	if(algorithmTimingWorkaround1):
		if(activationTime >= previousSequentialSegmentActivationTime):
			sequentiality = True	
	else:
		if(activationTime > previousSequentialSegmentActivationTime):
			sequentiality = True
	
	propagate = False
	if(verifyPropagationTime):
		#print("activationTime = ", activationTime)
		#print("previousSequentialSegmentActivationTime = ", previousSequentialSegmentActivationTime)
		#print("activationPropagationTimeMax = ", activationPropagationTimeMax)
		if(activationTime <= previousSequentialSegmentActivationTime+activationPropagationTimeMax):
			propagate = True
	else:
		propagate = True
		
	activationTimeVerified = sequentiality and propagate
		
	return activationTimeVerified

#last access time	
def calculateActivationTimeSequence(wordIndex):
	activationTime = wordIndex
	return activationTime

def calculateInputActivationLevel(connection):
	inputActivationLevel = objectLocalActivationLevelOff
	if(weightedSequentialSegmentInputs):
		inputActivationLevel = connection.weight
	else:
		inputActivationLevel = objectLocalActivationLevelOn
	return inputActivationLevel

def calculateSequentialSegmentActivationState(activationLevel, vectorised=False):
	if(weightedSequentialSegmentInputs):
		if(performSummationOfSequentialSegmentInputs):
			if(activationLevel >= sequentialSegmentMinActivationLevel):
				activationState = objectAreaActivationLevelOn
			else:
				activationState = objectAreaActivationLevelOff
		else:
			if(activationLevel > 0):
				activationState = objectAreaActivationLevelOn
			else:
				activationState = objectAreaActivationLevelOff			
	else:
		if(vectorised):
			if(activationLevel == vectorisedActivationLevelOn):
				activationState = objectAreaActivationLevelOn
			else:
				activationState = objectAreaActivationLevelOff		
		else:
			if(activationLevel):
				activationState = objectAreaActivationLevelOn
			else:
				activationState = objectAreaActivationLevelOff
	return activationState

def sequentialSegmentActivationLevelAboveZero(activationLevel):
	result = False
	if(weightedSequentialSegmentInputs):
		if(activationLevel > 0):
			result = True
		else:
			result = False
	else:
		if(activationLevel):
			result = True
		else:
			result = False
	return result
	
	
def generateDendriticSANIFileName(sentenceOrNetwork, sentenceIndex=None, write=False):
	fileName = "useAlgorithmDendriticSANIbiologicalSimulation"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	else:
		fileName = fileName + "Network"
		if(not outputBiologicalSimulationNetworkLastSentenceOnly):
			fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	if(outputFileNameComputationType):
		if(vectoriseComputation):
			fileName = fileName + "VectorisedComputation"
		else:
			fileName = fileName + "StandardComputation"
	return fileName
	
def generateDendriticSANIDynamicNeuronFileName(sentenceOrNetwork, wSource, sentenceIndex=None):
	fileName = "biologicalSimulationDynamic"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	fileName = fileName + "Wsource" + convertIntToString(wSource)
	if(outputFileNameComputationType):
		if(vectoriseComputation):
			fileName = fileName + "VectorisedComputation"
		else:
			fileName = fileName + "StandardComputation"
	return fileName

def generateDendriticSANIDynamicSequentialSegmentFileName(sentenceOrNetwork, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex=None):
	fileName = "biologicalSimulationDynamic"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	fileName = fileName + "Wsource" + convertIntToString(wSource)
	fileName = fileName + "branchIndex1" + convertIntToString(branchIndex1)
	fileName = fileName + "sequentialSegmentIndex" + convertIntToString(sequentialSegmentIndex)
	if(outputFileNameComputationType):
		if(vectoriseComputation):
			fileName = fileName + "VectorisedComputation"
		else:
			fileName = fileName + "StandardComputation"
	return fileName

def convertIntToString(integer, zeroPadLength=3):
	string = str(integer).zfill(zeroPadLength)
	return string
		
def findSequentialSegmentInputBySourceNode(sequentialSegment, sourceConceptNode):
	foundSequentialSegmentInput = False
	sequentialSegmentInput = None
	if(preventGenerationOfDuplicateConnections):
		if(sourceConceptNode.nodeName in sequentialSegment.inputs):		
			foundSequentialSegmentInput = True	
			sequentialSegmentInput = sequentialSegment.inputs[sourceConceptNode.nodeName]
	else:
		print("findSequentialSegmentInputBySourceNode error: currently requires preventGenerationOfDuplicateConnections")
		exit()
	return foundSequentialSegmentInput, sequentialSegmentInput

def applySomaActivation(conceptNeuronConnectionTarget, conceptNeuronTarget, somaActivationFoundCurrent, deactivateConnectionTarget, connectionTargetActivationFoundSet=None):
	if(deactivateConnectionTarget):
		conceptNeuronConnectionTarget.activationLevel = somaActivationFoundCurrent	
	if(somaActivationFoundCurrent):
		if(not deactivateConnectionTarget):
			conceptNeuronConnectionTarget.activationLevel = somaActivationFoundCurrent
			if(biologicalSimulationTestHarness):
				if(emulateVectorisedComputationOrder):
					if(conceptNeuronConnectionTarget not in(connectionTargetActivationFoundSet)):
						connectionTargetActivationFoundSet.add(conceptNeuronConnectionTarget)	#current implementation only works for !deactivateConnectionTargetIfSomaActivationNotFound
						print("biologicalSimulationTestHarness: conceptNeuronConnectionTarget somaActivationFoundCurrent = ", conceptNeuronConnectionTarget.nodeName)
				else:
					print("biologicalSimulationTestHarness: conceptNeuronConnectionTarget somaActivationFoundCurrent = ", conceptNeuronConnectionTarget.nodeName)
	somaActivationFound = calculateSomaActivationFound(conceptNeuronConnectionTarget, conceptNeuronTarget, somaActivationFoundCurrent)
	return somaActivationFound
	
def calculateSomaActivationFound(conceptNeuronConnectionTarget, conceptNeuronTarget, somaActivationFoundCurrent):
	somaActivationFound = False
	if(somaActivationFoundCurrent):
		if(conceptNeuronConnectionTarget == conceptNeuronTarget):
			somaActivationFound = True
	return somaActivationFound
							
def isMostDistalSequentialSegmentInBranch(sequentialSegmentIndex):
	result = False
	if(sequentialSegmentIndex == numberOfBranchSequentialSegments-1):
		result = True
	return result
	
def deactivatePreviousSequentialSegmentOrSubbranch(currentSequentialSegment):
	dendriticBranch = currentSequentialSegment.branch
	sequentialSegmentIndex = currentSequentialSegment.sequentialSegmentIndex
	if(isMostDistalSequentialSegmentInBranch(sequentialSegmentIndex)):
		for subbranchIndex, subbranch in enumerate(dendriticBranch.subbranches):
			subbranch.activationLevel = objectAreaActivationLevelOff
			previousSequentialSegment = subbranch.sequentialSegments[sequentialSegmentIndexMostProximal]
			resetSequentialSegmentActivation(previousSequentialSegment)
			if(resetConnectionTargetNeuronDendriteDuringActivationFreezeUntilRoundCompletion):
				previousSequentialSegment.frozen = True
				#print("freeze")
	else:
		previousSequentialSegmentIndex = sequentialSegmentIndex+1
		previousSequentialSegment = dendriticBranch.sequentialSegments[previousSequentialSegmentIndex]
		resetSequentialSegmentActivation(previousSequentialSegment)
		if(resetConnectionTargetNeuronDendriteDuringActivationFreezeUntilRoundCompletion):
			previousSequentialSegment.frozen = True
			#print("freeze")

def resetDendriticTreeLastSequentialSegmentActivation(conceptNeuron):
	conceptNeuron.activationLevel = objectAreaActivationLevelOff
	
	dendriticTreeLastBranch = conceptNeuron.dendriticTree
	dendriticTreeLastBranch.activationLevel = objectAreaActivationLevelOff
	dendriticTreeLastSequentialSegment = dendriticTreeLastBranch.sequentialSegments[sequentialSegmentIndexMostProximal]
	resetSequentialSegmentActivation(dendriticTreeLastSequentialSegment)

	if(vectoriseComputationCurrentDendriticInput):
		resetDendriticTreeLastSequentialSegmentActivationVectorised(conceptNeuron)

def resetDendriticTreeLastSequentialSegmentActivationVectorised(conceptNeuron):
	#print(conceptNeuron.vectorisedBranchActivationLevelList[branchIndex1MostProximal][0, 0, sequentialSegmentIndexMostProximal])
	conceptNeuron.vectorisedBranchActivationLevelList[branchIndex1MostProximal][0, 0, sequentialSegmentIndexMostProximal].assign(vectorisedActivationLevelOff)
	
def unfreezeDendriticTreeActivation(currentBranch):
	for sequentialSegment in currentBranch.sequentialSegments:
		sequentialSegment.frozen = False
		#print("unfreezeDendriticTreeActivation")
	for subbranch in currentBranch.subbranches:	
		unfreezeDendriticTreeActivation(subbranch)

def measureBranchActivationRecurse(currentBranch):
	branchActivation = 0
	branchActivation+= measureBranchActivation(currentBranch)	
	for subbranch in currentBranch.subbranches:	
		branchActivation+= measureBranchActivationRecurse(subbranch)
	return branchActivation

def measureBranchActivation(currentBranch):
	branchActivation = 0
	#branchActivation+= int(currentBranch.activationLevel)
	for sequentialSegment in currentBranch.sequentialSegments:
		branchActivation+= measureSequentialSegmentActivation(sequentialSegment)
	return branchActivation

def measureSequentialSegmentActivation(sequentialSegment):
	branchActivation = 0
	if(recordSequentialSegmentInputActivationLevels):
		for sequentialSegmentInput in sequentialSegment.inputs.values():
			branchActivation+= measureSequentialSegmentInputActivation(sequentialSegmentInput)
	else:
		branchActivation+= int(sequentialSegment.activationLevel)
	return branchActivation

def measureSequentialSegmentInputActivation(sequentialSegmentInput):	
	return int(sequentialSegmentInput.activationLevel)
	
def measureDendriticTreeActivationVectorised(conceptNode):
	branchActivation = 0
	for vectorisedBranchActivationLevel in conceptNode.vectorisedBranchActivationLevelList:
		branchIndex1Activation = tf.math.reduce_sum(vectorisedBranchActivationLevel)
		branchActivation+= branchIndex1Activation
	return branchActivation
