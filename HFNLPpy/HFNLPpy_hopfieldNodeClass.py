"""HFNLPpy_hopfieldNodeClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
ATNLP Hopfield Node Class

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

#if(biologicalSimulation):
vectoriseComputation = True	#parallel processing for optimisation
if(vectoriseComputation):
	vectoriseComputationCurrentDendriticInput = True	#mandatory - default behaviour
	if(vectoriseComputationCurrentDendriticInput):
		vectoriseComputationIndependentBranches = True	#mandatory - default behaviour
	batchSize = 100	#high batch size allowed since parallel processing simple/small scalar operations (on effective boolean synaptic inputs), lowered proportional to max (most distal) numberOfHorizontalBranches
	import tensorflow as tf
	recordVectorisedBranchObjectList = True	#vectorisedBranchObjectList required for drawBiologicalSimulationDynamic only
	if(recordVectorisedBranchObjectList):
		debugVectorisedBranchObjectList = False
else:
	vectoriseComputationCurrentDendriticInput = False
	debugVectorisedBranchObjectList = False

biologicalSimulationForward = True	#default mode	#required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic
if(not vectoriseComputation):
	biologicalSimulationForward = True	#False; orig implementation; simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup
if(biologicalSimulationForward):
	resetWsourceNeuronDendriteAfterActivation = True
	
recordSequentialSegmentInputActivationLevels = True	#required for draw of active simulation
if(vectoriseComputation):
	if(recordSequentialSegmentInputActivationLevels):
		vectoriseComputionUseSequentialSegmentInputActivationLevels	= False	#optional - not yet implemented (allows sequential segment activation to be dependent on summation of individual local inputs)
		if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
			vectoriseComputionUseSequentialSegmentBuffer = False
			numberOfSequentialSegmentInputs = 100	#max number available
		else:
			if(preventReactivationOfSequentialSegments):
				vectoriseComputionUseSequentialSegmentBuffer = True	#mandatory
			else:
				vectoriseComputionUseSequentialSegmentBuffer = False	#optional
			
numberOfBranches1 = 3	#number of vertical branches
numberOfBranches2 = 2	#number of new horizontal branches created at each vertical branch
#[3,3,3]	#number of new horizontal branches created at each vertical branch
numberOfBranchSequentialSegments = 1	#1+	#sequential inputs (FUTURE: if > 1: each branch segment may require sequential inputs)
#numberOfBranchSequentialSegmentInputs = 1	#1+	#nonSequentialInputs	#in current implementation (non-parallel generative network) number of inputs at sequential segment is dynamically increased on demand #not used; currently encode infinite number of

#probabilityOfSubsequenceThreshold = 0.01	#FUTURE: calibrate depending on number of branches/sequentialSegments etc

subsequenceLengthCalibration = 1.0

numberOfHorizontalSubBranchesRequiredForActivation = 2	#calibrate
activationRepolarisationTime = 1	#calibrate

resetSequentialSegments = False



class HopfieldNode:
	def __init__(self, networkIndex, nodeName, wordVector, nodeGraphType, activationTime, biologicalSimulation, w, sentenceIndex):
		#primary vars;
		self.networkIndex = networkIndex
		self.nodeName = str(nodeName)
		self.wordVector = wordVector	#numpy array
		#self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
		self.activationLevel = False	#currently only used by drawBiologicalSimulationDynamic
		self.activationTime = activationTime	#last activation time (used to calculate recency)	#not currently used
		
		#sentence artificial vars (for sentence graph only, do not generalise to network graph);	
		self.w = w
		self.sentenceIndex = sentenceIndex
		
		#connection vars;
		self.sourceConnectionDict = {}
		self.targetConnectionDict = {}
		#self.sourceConnectionList = []
		#self.targetConnectionList = []

		if(biologicalSimulation):
			self.activationTimeWord = None

			#if(biologicalSimulationDraw):
			#required to assign independent names to each;
			self.currentBranchIndexNeuron = 0
			self.currentSequentialSegmentIndexNeuron = 0
			self.currentSequentialSegmentInputIndexNeuron = 0 

			if(vectoriseComputationCurrentDendriticInput):
				self.vectorisedBranchActivationLevelList, self.vectorisedBranchActivationTimeList, self.vectorisedBranchObjectList = createBatchDendriticTreeVectorised(batched=False)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
				#vectorisedBranchObjectList required for drawBiologicalSimulationDynamic only
				
			self.dendriticTree = createDendriticTree(self, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)

			if(debugVectorisedBranchObjectList):
				numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
				for branchIndex1 in reversed(range(numberOfVerticalBranches)):
					vectorisedBranchObjectSequentialSegment = self.vectorisedBranchObjectList[branchIndex1]
					print("vectorisedBranchObjectSequentialSegment = ", vectorisedBranchObjectSequentialSegment)
					for horizontalBranchIndex in range(vectorisedBranchObjectSequentialSegment.shape[0]):
						for branchIndex2 in range(vectorisedBranchObjectSequentialSegment.shape[1]):
							for sequentialSegmentIndex in range(vectorisedBranchObjectSequentialSegment.shape[2]):
								#print("branchIndex1 = ", branchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex, ", branchIndex2 = ", branchIndex2, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
								sequentialSegment = vectorisedBranchObjectSequentialSegment[horizontalBranchIndex, branchIndex2, sequentialSegmentIndex]
								#print("sequentialSegment.nodeName = ", sequentialSegment.nodeName)



def createVectorisedBranches(batched=False):
	vectorisedBranchActivationLevelList = []	#list of tensors for every branchIndex1	- each element is of shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
	vectorisedBranchActivationTimeList = []
	vectorisedBranchObjectList = []
	for currentBranchIndex1 in range(calculateNumberOfVerticalBranches(numberOfBranches1)):
		numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(currentBranchIndex1, numberOfBranches2)
		#tf.Variable designation is required for assign() operations
		if(batched):
			if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				if(recordVectorisedBranchObjectList):
					recordVectorisedBranchObject = np.empty(shape=(batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs), dtype=object)			
			else:
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				if(recordVectorisedBranchObjectList):
					recordVectorisedBranchObject = np.empty(shape=(batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments), dtype=object)
		else:
			if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				if(recordVectorisedBranchObjectList):
					recordVectorisedBranchObject = np.empty(shape=(numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs), dtype=object)			
			else:
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				if(recordVectorisedBranchObjectList):
					recordVectorisedBranchObject = np.empty(shape=(numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments), dtype=object)
		vectorisedBranchActivationLevelList.append(vectorisedBranchActivationLevel)
		vectorisedBranchActivationTimeList.append(vectorisedBranchActivationTime)
		if(recordVectorisedBranchObjectList):
			vectorisedBranchObjectList.append(recordVectorisedBranchObject)
	return vectorisedBranchActivationLevelList, vectorisedBranchActivationTimeList, vectorisedBranchObjectList

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
								
#last access time	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime
	
def calculateActivationTimeSequence(wordIndex):
	activationTime = wordIndex
	return activationTime
	
#creation time
def calculateSpatioTemporalIndex(sentenceIndex):
	#for biologicalPrototype: e.g. 1) interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex/clause); or 2) store spatiotemporal index synapses on separate dendritic branch
	spatioTemporalIndex = sentenceIndex
	return spatioTemporalIndex

def createConnectionKeyIfNonExistant(dic, key):
	if key not in dic:
		dic[key] = []	#create new empty list
		
def generateHopfieldGraphNodeName(word, lemma):
	if(storeConceptNodesByLemma):
		nodeName = lemma
	else:
		nodeName = word
	return nodeName

#if(biologicalSimulation):
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
				if(debugVectorisedBranchObjectList):
					print("recordVectorisedBranchObjectList: branch.branchIndex1 = ", branch.branchIndex1, "branch.branchIndex2 = ", branch.branchIndex2, "branch.horizontalBranchIndex = ", branch.horizontalBranchIndex, "sequentialSegmentIndex = ", sequentialSegmentIndex)
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
		
			
def createBatchDendriticTreeVectorised(batched=False):
	vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchObjectBatchList = createVectorisedBranches(batched)
	return vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchObjectBatchList
		
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
		conceptNeuron.vectorisedBranchActivationLevelList, conceptNeuron.vectorisedBranchActivationTimeList, _ = createBatchDendriticTreeVectorised(batched=False)	#rezero tensors by regenerating them 	#do not overwrite conceptNeuron.vectorisedBranchObjectList

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

