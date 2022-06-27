"""HFNLPpy_biologicalSimulationNode.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
ATNLP Biological Simulation Node Classes

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np


	
storeConceptNodesByLemma = True	#else store by word (morphology included)

graphNodeTypeConcept = 1	#base/input neuron (network neuron)

#if(biologicalImplementationReuseSynapticSubstrateForIdenticalSubsequences):
graphNodeTypeStart = 5	#start of sequence - used by biologicalImplementationReuseSynapticSubstrateForIdenticalSubsequences only
nodeNameStart = "SEQUENCESTARTNODE"

preventReactivationOfSequentialSegments = True	#prevent reactivation of sequential segments (equates to a long repolarisation time of ~= sentenceLength)	#algorithmTimingWorkaround2
algorithmTimingWorkaround1 = False	#insufficient workaround

#FUTURE: performSummationOfSequentialSegmentInputs = False #allows sequential segment activation to be dependent on summation of individual local inputs #support multiple source neurons fired simultaneously

vectoriseComputation = True	#parallel processing for optimisation
if(vectoriseComputation):
	import tensorflow as tf
	vectoriseComputationCurrentDendriticInput = True	#mandatory - default behaviour
	if(vectoriseComputationCurrentDendriticInput):
		vectoriseComputationIndependentBranches = True	#mandatory - default behaviour
	batchSize = 100	#high batch size allowed since parallel processing simple/small scalar operations (on effective boolean synaptic inputs), lowered proportional to max (most distal) numberOfHorizontalBranches
	
	updateNeuronObjectActivationLevels = True	#activation levels are required to be stored in denditicTree object structure (HopfieldNode/DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic
	if(updateNeuronObjectActivationLevels):
		recordVectorisedBranchObjectList = True	#vectorisedBranchObjectList is required to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels (as HFNLPpy_biologicalSimulationDraw currently only supports drawing of denditicTree object structure activations)  
	else:
		recordVectorisedBranchObjectList = False	#vectorisedBranchObjectList is not required as it is not necessary to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput); activation levels are not required to be stored in denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput)
else:
	vectoriseComputationCurrentDendriticInput = False

biologicalSimulationForward = True	#default mode	#required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic
if(not vectoriseComputation):
	biologicalSimulationForward = True	#False; orig implementation; simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup
if(biologicalSimulationForward):
	resetWsourceNeuronDendriteAfterActivation = True

if(vectoriseComputation):
	recordSequentialSegmentInputActivationLevels = True	#optional
	if(updateNeuronObjectActivationLevels):
		recordSequentialSegmentInputActivationLevels = True	#required for draw of active simulation - required by drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels	
else:
	recordSequentialSegmentInputActivationLevels = True	#optional (not required by HFNLPpy_biologicalSimulationStandard processing, and dynamic draw is not supported)
if(vectoriseComputation):
	if(recordSequentialSegmentInputActivationLevels):
		vectoriseComputionUseSequentialSegmentInputActivationLevels	= False	#not yet implemented	#not required as local segment inputs must fire simultaneously; so they can be stored as a segment scalar value	#only ever used in buffer processing
		if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
			numberOfSequentialSegmentInputs = 100	#max number available

			
numberOfBranches1 = 3	#number of vertical branches -1
numberOfBranches2 = 2	#number of new horizontal branches created at each vertical branch
	#[1,2,4,8]	#number of new horizontal branches created at each vertical branch
numberOfBranchSequentialSegments = 1	#1+	#sequential inputs (FUTURE: if > 1: each branch segment may require sequential inputs)
#numberOfBranchSequentialSegmentInputs = 1	#1+	#nonSequentialInputs	#in current implementation (non-parallel generative network) number of inputs at sequential segment is dynamically increased on demand #not used; currently encode infinite number of

#probabilityOfSubsequenceThreshold = 0.01	#FUTURE: calibrate depending on number of branches/sequentialSegments etc

subsequenceLengthCalibration = 1.0

numberOfHorizontalSubBranchesRequiredForActivation = 2	#calibrate
activationRepolarisationTime = 1	#calibrate

resetSequentialSegments = False


			
def biologicalSimulationNodePropertiesInitialisation(conceptNode):

	conceptNode.activationLevel = False	#currently only used by drawBiologicalSimulationDynamic

	conceptNode.activationTimeWord = None

	#if(biologicalSimulationDraw):
	#required to assign independent names to each;
	conceptNode.currentBranchIndexNeuron = 0
	conceptNode.currentSequentialSegmentIndexNeuron = 0
	conceptNode.currentSequentialSegmentInputIndexNeuron = 0 

	if(vectoriseComputationCurrentDendriticInput):
		if(recordVectorisedBranchObjectList):
			conceptNode.vectorisedBranchActivationLevelList, conceptNode.vectorisedBranchActivationTimeList, conceptNode.vectorisedBranchObjectList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=recordVectorisedBranchObjectList, storeSequentialSegmentInputActivationLevels=False)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
		else:
			conceptNode.vectorisedBranchActivationLevelList, conceptNode.vectorisedBranchActivationTimeList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=recordVectorisedBranchObjectList, storeSequentialSegmentInputActivationLevels=False)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
		#vectorisedBranchObjectList required for drawBiologicalSimulationDynamic only

	conceptNode.dendriticTree = createDendriticTree(conceptNode, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)

class DendriticBranch:
	def __init__(self, conceptNode, parentBranch, numberOfBranchSequentialSegments, branchIndex1, branchIndex2, horizontalBranchIndex):
		self.parentBranch = parentBranch
		self.subbranches = []
		
		#if(biologicalSimulationDraw):
		self.nodeName = generateDendriticBranchName(conceptNode)
		self.branchIndex1 = branchIndex1
		self.branchIndex2 = branchIndex2	#local horizontalBranchIndex (wrt horizontalBranchWidth)
		#print("horizontalBranchIndex = ", horizontalBranchIndex)
		self.horizontalBranchIndex = horizontalBranchIndex	#absolute horizontalBranchIndex	#required by vectoriseComputationCurrentDendriticInput only
		self.conceptNode = conceptNode

		self.sequentialSegments = [SequentialSegment(conceptNode, self, i) for i in range(numberOfBranchSequentialSegments)]	#[SequentialSegment(conceptNode, self, i)]*numberOfBranchSequentialSegments
		self.activationLevel = False
		self.activationTime = None	#within sequence/sentence activation time
		
		#if(vectoriseComputationCurrentDendriticInput):
		#	self.sequentialSegmentInputIndex = None
						
class SequentialSegment:
	def __init__(self, conceptNode, branch, sequentialSegmentIndex):
		self.inputs = []
		self.activationLevel = False
		self.activationTime = None	#within sequence/sentence activation time
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
					
class SequentialSegmentInput:
	def __init__(self, conceptNode, SequentialSegment, sequentialSegmentInputIndex):
		self.input = None
		self.sequentialSegment = SequentialSegment
		self.firstInputInSequence = False	#within sequence/sentence activation time
		
		if(recordSequentialSegmentInputActivationLevels):
			self.activationLevel = False	#input has been temporarily triggered for activation (only affects dendritic signal if sequentiality requirements met)
			self.activationTime = None	#input has been temporarily triggered for activation (only affects dendritic signal if sequentiality requirements met)
			self.sequentialSegmentInputIndex = sequentialSegmentInputIndex
			
		#if(biologicalSimulationDraw):
		self.nodeName = generateSequentialSegmentInputName(conceptNode)
		self.sequentialSegmentInputIndex = None	#not required	#index record value not robust if inputs are removed (synaptic atrophy)
		self.conceptNode = conceptNode	#not required (as can lookup SequentialSegment.branch.conceptNode)
		




#if(vectoriseComputationCurrentDendriticInput):
								
def createDendriticTreeVectorised(batched, createVectorisedBranchObjectList, storeSequentialSegmentInputActivationLevels):
	vectorisedBranchActivationLevelList = []	#list of tensors for every branchIndex1	- each element is of shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
	vectorisedBranchActivationTimeList = []
	if(createVectorisedBranchObjectList):
		vectorisedBranchObjectList = []
	for currentBranchIndex1 in range(calculateNumberOfVerticalBranches(numberOfBranches1)):
		numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(currentBranchIndex1, numberOfBranches2)
		#tf.Variable designation is required for assign() operations
		if(batched):
			if(storeSequentialSegmentInputActivationLevels):
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs), dtype=object)			
			else:
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments), dtype=object)
		else:
			if(storeSequentialSegmentInputActivationLevels):
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs), dtype=object)			
			else:
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments), dtype=object)
		vectorisedBranchActivationLevelList.append(vectorisedBranchActivationLevel)
		vectorisedBranchActivationTimeList.append(vectorisedBranchActivationTime)
		if(createVectorisedBranchObjectList):
			vectorisedBranchObjectList.append(vectorisedBranchObject)
	
	if(createVectorisedBranchObjectList):
		return vectorisedBranchActivationLevelList, vectorisedBranchActivationTimeList, vectorisedBranchObjectList	
	else:
		return vectorisedBranchActivationLevelList, vectorisedBranchActivationTimeList

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
		horizontalBranchWidth = 1	#CHECKTHIS: first branch has single width
	
	return numberOfHorizontalBranches, horizontalBranchWidth

def calculateNumberOfVerticalBranches(numberOfBranches1):
	numberOfVerticalBranches = numberOfBranches1+1	#or +2
	return numberOfVerticalBranches
		
def createDendriticTree(conceptNode, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments):	
	currentBranchIndex1, currentBranchIndex2, horizontalBranchIndex = (0, 0, 0)
	dendriticTreeHeadBranch = DendriticBranch(conceptNode, None, numberOfBranchSequentialSegments, currentBranchIndex1, currentBranchIndex2, horizontalBranchIndex)
	createDendriticTreeBranch(conceptNode, dendriticTreeHeadBranch, currentBranchIndex1+1, horizontalBranchIndex, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)	#execution update: set currentBranchIndex1+1
	return dendriticTreeHeadBranch
			
def createDendriticTreeBranch(conceptNode, currentBranch, currentBranchIndex1, horizontalBranchIndex, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments):
	#printIndentation(currentBranchIndex1)
	#print("currentBranchIndex1 = ", currentBranchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex)
	currentBranch.subbranches = [DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, i, horizontalBranchIndex) for i in range(numberOfBranches2)]	#[DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, currentBranchIndex2)]*numberOfBranches2
	if(currentBranchIndex1 < numberOfBranches1):
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



def resetDendriticTreeActivation(conceptNeuron):
	conceptNeuron.activationLevel = False
	resetBranchActivation(conceptNeuron.dendriticTree)
	if(vectoriseComputationCurrentDendriticInput):
		conceptNeuron.vectorisedBranchActivationLevelList, conceptNeuron.vectorisedBranchActivationTimeList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=False, storeSequentialSegmentInputActivationLevels=False)	#rezero tensors by regenerating them 	#do not overwrite conceptNeuron.vectorisedBranchObjectList

def resetAxonsActivation(conceptNeuron):
	conceptNeuron.activationLevel = False
	for targetConnectionConceptName, connectionList in conceptNeuron.targetConnectionDict.items():
		for connection in connectionList:
			connection.activationLevel = False

def resetBranchActivation(currentBranch):
	currentBranch.activationLevel = False
	
	for sequentialSegment in currentBranch.sequentialSegments:
		sequentialSegment.activationLevel = False
		if(recordSequentialSegmentInputActivationLevels):
			for sequentialSegmentInput in sequentialSegment.inputs:
				sequentialSegmentInput.activationLevel = False
										
	for subbranch in currentBranch.subbranches:	
		resetBranchActivation(subbranch)
	
def printIndentation(level):
	for indentation in range(level):
		print('\t', end='')

def verifyRepolarised(currentSequentialSegment, activationTime):
	if(currentSequentialSegment.activationLevel):
		repolarised = False
		if(activationTime >= currentSequentialSegment.activationTime+activationRepolarisationTime):
			#do not reactivate sequential segment if already activated by same source neuron
			repolarised = True
	else:
		repolarised = True
	return repolarised
	
def verifySequentialActivationTime(currentSequentialSegmentActivationTime, previousSequentialSegmentActivationTime):
	sequentiality = False
	if(algorithmTimingWorkaround1):
		if(currentSequentialSegmentActivationTime >= previousSequentialSegmentActivationTime):
			sequentiality = True	
	else:
		if(currentSequentialSegmentActivationTime > previousSequentialSegmentActivationTime):
			sequentiality = True
	return sequentiality

#last access time	
def calculateActivationTimeSequence(wordIndex):
	activationTime = wordIndex
	return activationTime
