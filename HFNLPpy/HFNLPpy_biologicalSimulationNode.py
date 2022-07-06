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


vectoriseComputation = False	#parallel processing for optimisation

deactivateConnectionTargetIfSomaActivationNotFound = True	#default:True #True: orig simulateBiologicalHFnetworkSequenceNodesPropagateParallel:calculateNeuronActivationParallel method, False: orig simulateBiologicalHFnetworkSequenceNodePropagateStandard method

biologicalSimulationTestHarness = False
HFNLPnonrandomSeed = False
emulateVectorisedComputationOrder = False
if(biologicalSimulationTestHarness):
	if(not vectoriseComputation):
		#emulateVectorisedComputationOrder requires biologicalSimulationForward, !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
		emulateVectorisedComputationOrder = True	#change standard computation to execute in order of vectorised computation (for comparison)
	HFNLPnonrandomSeed = True	#always generate the same set of random numbers upon execution
		
enforceMinimumEncodedSequenceLength = False	#do not execute addPredictiveSequenceToNeuron if predictive sequence is short (ie does not use up the majority of numberOfBranches1)
if(enforceMinimumEncodedSequenceLength):
	minimumEncodedSequenceLength = 4	#should be high enough to fill a significant proportion of dendrite vertical branch length (numberOfBranches1)	#~seedHFnetworkSubsequenceLength
	
seedHFnetworkSubsequence = False #seed/prime HFNLP network with initial few words of a trained sentence and verify that full sentence is sequentially activated (interpret last sentence as target sequence, interpret first seedHFnetworkSubsequenceLength words of target sequence as seed subsequence)
if(seedHFnetworkSubsequence):
	#seedHFnetworkSubsequence currently requires !biologicalSimulationEncodeSyntaxInDendriticBranchStructure, resetWsourceNeuronDendriteAfterActivation
	seedHFnetworkSubsequenceLength = 2	#must be < len(targetSentenceConceptNodeList)
	seedHFnetworkSubsequenceBasic = False	#emulate simulateBiologicalHFnetworkSequenceTrain:simulateBiologicalHFnetworkSequenceNodePropagateWrapper method (only propagate those activate neurons that exist in the target sequence); else propagate all active neurons
	seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant = True
	
supportForNonBinarySubbranchSize = False
performSummationOfSequentialSegmentInputsAcrossBranch = False
weightedSequentialSegmentInputs = False
allowNegativeActivationTimes = False	#calculateNeuronActivationSyntacticalBranchDPlinear current implementation does not require allowNegativeActivationTimes
expectFirstBranchSequentialSegmentConnection = True	#True:default	#False: orig implementation

biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False	#determines HFNLPpy_hopfieldGraph:useDependencyParseTree	#speculative: use precalculated syntactical structure to generate dendritic branch connections (rather than deriving syntax from commonly used dendritic subsequence encodings)
if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
	biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect = True	#speculative: directly encode precalculated syntactical structure into dendritic branches
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect):
		expectFirstBranchSequentialSegmentConnection = False
		supportForNonBinarySubbranchSize = True	#required by useDependencyParseTree:biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect with non-binary dependency/constituency parse trees
		allowNegativeActivationTimes = True
	else:
		#implied biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinear = True	#speculative: convert precalculated syntactical structure to linear subsequences before encoding into dendritic branches
		biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearHierarchical = False	#incomplete #adds the most distant nodes to the start of a linear contextConceptNodesList - will still perform propagate/predict in reverse order of tree crawl
		#if(not biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearHierarchical):
		#	implied biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearCrawl = True: adds the nodes in reverse order of tree crawl to a linear contextConceptNodesList) - will also perform propagate/predict in reverse order of tree crawl
			
if(supportForNonBinarySubbranchSize):
	performSummationOfSequentialSegmentInputsAcrossBranch = True
	debugBiologicalSimulationEncodeSyntaxInDendriticBranchStructure = False	#reduce number of subbranches to support and draw simpler dependency tree (use with drawBiologicalSimulationDynamic)
	if(performSummationOfSequentialSegmentInputsAcrossBranch):
		weightedSequentialSegmentInputs = True
		#performSummationOfSequentialSegmentInputsAcrossBranch does not support numberOfBranchSequentialSegments>1 as branch summation occurs using the final sequentialSegment activationLevel of each subbranch
		#performSummationOfSequentialSegmentInputsAcrossBranch does not support performSummationOfSequentialSegmentInputs (or multiple simultaneous inputs at a sequential segment) as this will arbitrarily overwrite the precise sequentialSegment activationLevel of a subbranch


if(allowNegativeActivationTimes):
	minimumActivationTime = -1000
else:
	minimumActivationTime = 0	#alternatively set -1;	#initial activation time of dendritic sequence set artificially low such that passSegmentActivationTimeTests automatically pass (not required (as passSegmentActivationTimeTests are ignored for currentSequentialSegmentInput.firstInputInSequence)
		
preventGenerationOfDuplicateConnections = True	#note sequentialSegment inputs will be stored as a dictionary indexed by source node name (else indexed by sequentialSegmentInputIndex)

storeSequentialSegmentInputIndexValues = False	#not required	#index record value not robust if inputs are removed (synaptic atrophy)	#HFNLPpy_biologicalSimulationDraw can use currentSequentialSegmentInputIndexDynamic instead

preventReactivationOfSequentialSegments = True	#prevent reactivation of sequential segments (equates to a long repolarisation time of ~= sentenceLength)	#algorithmTimingWorkaround2
algorithmTimingWorkaround1 = False	#insufficient workaround

performSummationOfSequentialSegmentInputs = False #allows sequential segment activation to be dependent on summation of individual local inputs #support multiple source neurons fired simultaneously	#consider renaming to performSummationOfSequentialSegmentInputsLocal
if(performSummationOfSequentialSegmentInputs):
	weightedSequentialSegmentInputs = True
	#summationOfSequentialSegmentInputsFirstInputInSequenceOverride = True	#mandatory (only implementation coded) #True: orig HFNLPpy_biologicalSimulationPropagateStandard method	 #False: orig HFNLPpy_biologicalSimulationPropagateVectorised method
if(weightedSequentialSegmentInputs):
	sequentialSegmentMinActivationLevel = 1.0	#requirement: greater or equal to sequentialSegmentMinActivationLevel
else:
	sequentialSegmentMinActivationLevel = 1	#always 1 (not used)

if(vectoriseComputation):
	import tensorflow as tf
	vectoriseComputationCurrentDendriticInput = True	#mandatory - default behaviour
	if(vectoriseComputationCurrentDendriticInput):
		vectoriseComputationIndependentBranches = True	#mandatory - default behaviour
	batchSizeDefault = 100	#high batch size allowed since parallel processing simple/small scalar operations (on effective boolean synaptic inputs), lowered proportional to max (most distal) numberOfHorizontalBranches	#not used (createDendriticTreeVectorised is never called with batched=True)
	
	updateNeuronObjectActivationLevels = True	#only required for drawBiologicalSimulationDynamic (slows down processing)	#activation levels are required to be stored in denditicTree object structure (HopfieldNode/DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic
	if(updateNeuronObjectActivationLevels):
		recordVectorisedBranchObjectList = True	#vectorisedBranchObjectList is required to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels (as HFNLPpy_biologicalSimulationDraw currently only supports drawing of denditicTree object structure activations)  
	else:
		recordVectorisedBranchObjectList = False	#vectorisedBranchObjectList is not required as it is not necessary to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput); activation levels are not required to be stored in denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput)
else:
	vectoriseComputationCurrentDendriticInput = False

#default activation levels;
#key:
#"object" = neuron/dendritic tree class structure
#"local" = activation level for synaptic inputs/sequential segments
#"area" = activation level for dendritic branches/somas/axons
objectAreaActivationLevelOff = False
objectAreaActivationLevelOn = True
if(weightedSequentialSegmentInputs):
	#numeric (sequential segment only consider depolarised if sequential requirements met and summed activationLevel of sequential inputs passes threshold)
	objectLocalActivationLevelOff = 0.0
	objectLocalActivationLevelOn = sequentialSegmentMinActivationLevel		
else:
	#bool (sequential segment only consider depolarised if sequential requirements met)
	objectLocalActivationLevelOff = False
	objectLocalActivationLevelOn = True	
vectorisedActivationLevelOff = 0.0
vectorisedActivationLevelOn = 1.0
vectorisedActivationTimeFlagDefault = 0	#boolean flag (not a numeric activation time)
vectorisedActivationTimeFlagFirstInputInSequence = 1	#boolean flag (not a numeric activation time)
	

if(vectoriseComputation):
	biologicalSimulationForward = True	#mandatory (only implementation) #required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic
else:
	biologicalSimulationForward = True	#optional	#orig implementation; False (simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup)
if(biologicalSimulationForward):
	resetWsourceNeuronDendriteAfterActivation = True

if(vectoriseComputation):
	recordSequentialSegmentInputActivationLevels = True	#optional
	if(updateNeuronObjectActivationLevels):
		recordSequentialSegmentInputActivationLevels = True	#required for draw of active simulation - required by drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels	
else:
	recordSequentialSegmentInputActivationLevels = True	#optional (not required by HFNLPpy_biologicalSimulationPropagateStandard processing, and dynamic draw is not supported)
if(vectoriseComputation):
	if(recordSequentialSegmentInputActivationLevels):
		vectoriseComputionUseSequentialSegmentInputActivationLevels	= False	#not yet implemented	#not required as local segment inputs must fire simultaneously; so they can be stored as a segment scalar value	#only ever used in buffer processing
		if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
			numberOfSequentialSegmentInputs = 100	#max number available


numberOfBranches1 = 3	#number of vertical branches -1
if(supportForNonBinarySubbranchSize):
	if(debugBiologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		numberOfBranches2 = 4
	else:
		numberOfBranches2 = 20	#8	#number of new horizontal branches created at each vertical branch	#must sync with max number subbrances of constituency/dependency parser	#if dependencyParser: maximum number of dependents per governor
else:
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

	conceptNode.activationLevel = objectAreaActivationLevelOff	#currently only used by drawBiologicalSimulationDynamic

	conceptNode.activationTimeWord = None

	#if(biologicalSimulationDraw):
	#required to assign independent names to each;
	conceptNode.currentBranchIndexNeuron = 0
	conceptNode.currentSequentialSegmentIndexNeuron = 0
	conceptNode.currentSequentialSegmentInputIndexNeuron = 0 

	if(vectoriseComputationCurrentDendriticInput):
		if(recordVectorisedBranchObjectList):
			conceptNode.vectorisedBranchActivationLevelList, conceptNode.vectorisedBranchActivationTimeList, conceptNode.vectorisedBranchObjectList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=recordVectorisedBranchObjectList, storeSequentialSegmentInputActivationLevels=False)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]
			#vectorisedBranchActivationLevelList always stores effective boolean 1/0 values (since their activations are calculated across all wSource of same activationTime simultaneously); could be converted to dtype=tf.bool
			#vectorisedBranchActivationLevelListBuffer will store variable numeric values, biased for first inputs in sequences (likewise weightedSequentialSegmentInputs:vectorisedBranchActivationLevelListBuffer will store numeric values of the synaptic input activation levels being accumulated prior to batch processing)
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
		self.activationLevel = objectAreaActivationLevelOff
		self.activationTime = None	#within sequence/sentence activation time
						
class SequentialSegment:
	def __init__(self, conceptNode, branch, sequentialSegmentIndex):
		#self.inputs = []
		self.inputs = {}
		self.activationLevel = objectLocalActivationLevelOff	#only consider depolarised if activationLevel passes threshold
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
	def __init__(self, conceptNode, SequentialSegment, sequentialSegmentInputIndex, nodeSource):
		self.input = None
		self.sequentialSegment = SequentialSegment
		self.firstInputInSequence = False	#within sequence/sentence activation time
		
		if(recordSequentialSegmentInputActivationLevels):
			self.activationLevel = objectLocalActivationLevelOff	#input has been temporarily triggered for activation (only affects dendritic signal if sequentiality requirements met)
			self.activationTime = None	#numeric	#input has been temporarily triggered for activation (only affects dendritic signal if sequentiality requirements met)
			self.sequentialSegmentInputIndex = sequentialSegmentInputIndex
		
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
	if(createVectorisedBranchObjectList):
		vectorisedBranchObjectList = []
	for currentBranchIndex1 in range(calculateNumberOfVerticalBranches(numberOfBranches1)):
		numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(currentBranchIndex1, numberOfBranches2)
		#tf.Variable designation is required for assign() operations
		if(batched):
			if(storeSequentialSegmentInputActivationLevels):
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, numberOfSequentialSegmentInputs), dtype=object)			
			else:
				vectorisedBranchActivationLevel = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				vectorisedBranchActivationTime = tf.Variable(tf.zeros([batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]))
				if(createVectorisedBranchObjectList):
					vectorisedBranchObject = np.empty(shape=(batchSizeDefault, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments), dtype=object)
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
	conceptNeuron.activationLevel = objectAreaActivationLevelOff
	resetBranchActivation(conceptNeuron.dendriticTree)
	if(vectoriseComputationCurrentDendriticInput):
		resetDendriticTreeActivationVectorised(conceptNeuron)
		
def resetDendriticTreeActivationVectorised(conceptNeuron):
	conceptNeuron.activationLevel = objectAreaActivationLevelOff
	conceptNeuron.vectorisedBranchActivationLevelList, conceptNeuron.vectorisedBranchActivationTimeList = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=False, storeSequentialSegmentInputActivationLevels=False)	#rezero tensors by regenerating them 	#do not overwrite conceptNeuron.vectorisedBranchObjectList


def resetAxonsActivation(conceptNeuron):
	conceptNeuron.activationLevel = objectAreaActivationLevelOff
	for targetConnectionConceptName, connectionList in conceptNeuron.targetConnectionDict.items():
		resetAxonsActivationConnectionList(connectionList)

def resetAxonsActivationConnectionList(connectionList):
	for connection in connectionList:
		connection.activationLevel = objectAreaActivationLevelOff

def resetBranchActivation(currentBranch):
	currentBranch.activationLevel = objectAreaActivationLevelOff
	
	for sequentialSegment in currentBranch.sequentialSegments:
		sequentialSegment.activationLevel = objectLocalActivationLevelOff
		if(recordSequentialSegmentInputActivationLevels):
			#for sequentialSegmentInput in sequentialSegment.inputs:
			for sequentialSegmentInput in sequentialSegment.inputs.values():
				sequentialSegmentInput.activationLevel = objectLocalActivationLevelOff
										
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
	

def generateBiologicalSimulationFileName(sentenceOrNetwork, wSource, sentenceIndex=None):
	fileName = "biologicalSimulationDynamic"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + str(sentenceIndex)
	fileName = fileName + "Wsource" + str(wSource)
	return fileName

def generateBiologicalSimulationDynamicFileName(sentenceOrNetwork, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex=None):
	fileName = "biologicalSimulationDynamic"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + str(sentenceIndex)
	fileName = fileName + "Wsource" + str(wSource)
	fileName = fileName + "branchIndex1" + str(branchIndex1)
	fileName = fileName + "sequentialSegmentIndex" + str(sequentialSegmentIndex)
	return fileName
	
def findSequentialSegmentInputBySourceNode(sequentialSegment, sourceConceptNode):
	foundSequentialSegmentInput = False
	sequentialSegmentInput = None
	#for sequentialSegmentInput in sequentialSegment.inputs:
		#if(sequentialSegmentInput.nodeSource == sourceConceptNode):
	if(preventGenerationOfDuplicateConnections):
		if(sourceConceptNode.nodeName in sequentialSegment.inputs):		
			foundSequentialSegmentInput = True	
			sequentialSegmentInput = sequentialSegment.inputs[sourceConceptNode.nodeName]
	else:
		print("findSequentialSegmentInputBySourceNode error: currently requires preventGenerationOfDuplicateConnections")
		exit()
	return foundSequentialSegmentInput, sequentialSegmentInput

def applySomaActivation(conceptNeuronConnectionTarget, conceptNeuronTarget, somaActivationLevel, connectionTargetActivationFoundSet=None):
	somaActivationFound = False
	if(deactivateConnectionTargetIfSomaActivationNotFound):
		conceptNeuronConnectionTarget.activationLevel = somaActivationLevel	
	if(somaActivationLevel):
		if(not deactivateConnectionTargetIfSomaActivationNotFound):
			conceptNeuronConnectionTarget.activationLevel = somaActivationLevel
			if(biologicalSimulationTestHarness):
				if(emulateVectorisedComputationOrder):
					if(conceptNeuronConnectionTarget not in(connectionTargetActivationFoundSet)):
						connectionTargetActivationFoundSet.add(conceptNeuronConnectionTarget)	#current implementation only words for !deactivateConnectionTargetIfSomaActivationNotFound
						print("biologicalSimulationTestHarness: conceptNeuronConnectionTarget somaActivationLevel = ", conceptNeuronConnectionTarget.nodeName)
				else:
					print("biologicalSimulationTestHarness: conceptNeuronConnectionTarget somaActivationLevel = ", conceptNeuronConnectionTarget.nodeName)
		if(conceptNeuronConnectionTarget == conceptNeuronTarget):
			somaActivationFound = True
	return somaActivationFound
	
						
