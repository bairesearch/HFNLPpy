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

import numpy as np

storeConceptNodesByLemma = True	#else store by word (morphology included)


graphNodeTypeConcept = 1	#base/input neuron (network neuron)


#if(biologicalImplementationReuseSynapticSubstrateForIdenticalSubsequences):
graphNodeTypeStart = 5	#start of sequence - used by biologicalImplementationReuseSynapticSubstrateForIdenticalSubsequences only
nodeNameStart = "SEQUENCESTARTNODE"

#if(biologicalSimulation):
numberOfBranches1 = 3	#number of vertical branches
numberOfBranches2 = 2	#number of new horizontal branches created at each vertical branch
#[3,3,3]	#number of new horizontal branches created at each vertical branch
numberOfBranchSequentialSegments = 1	#1+	#sequential inputs (FUTURE: if > 1: each branch segment may require sequential inputs)
#numberOfBranchSequentialSegmentInputs = 1	#1+	#nonSequentialInputs	#in current implementation (non-parallel generative network) number of inputs at sequential segment is dynamically increased on demand #not used; currently encode infinite number of


class HopfieldNode:
	def __init__(self, networkIndex, nodeName, wordVector, nodeGraphType, activationTime, biologicalSimulation, w, sentenceIndex):
		#primary vars;
		self.networkIndex = networkIndex
		self.nodeName = str(nodeName)
		self.wordVector = wordVector	#numpy array
		#self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
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
			#if(biologicalSimulationDraw):
			#required to assign independent names to each;
			self.currentBranchIndexNeuron = 0
			self.currentSequentialSegmentIndexNeuron = 0
			self.currentSequentialSegmentInputIndexNeuron = 0 
			
			currentBranchIndex1, currentBranchIndex2 = (0, 0)
			dendriticTreeHeadBranch = DendriticBranch(self, None, numberOfBranchSequentialSegments, currentBranchIndex1, currentBranchIndex2)
			createDendriticTree(self, dendriticTreeHeadBranch, currentBranchIndex1, currentBranchIndex2, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)
			self.dendriticTree = dendriticTreeHeadBranch
		
			
#last access time	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
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
	def __init__(self, conceptNode, parentBranch, numberOfBranchSequentialSegments, branchIndex1, branchIndex2):
		self.parentBranch = parentBranch
		self.subbranches = []
		self.sequentialSegments = [SequentialSegment(conceptNode, self, i) for i in range(numberOfBranchSequentialSegments)]	#[SequentialSegment(conceptNode, self, i)]*numberOfBranchSequentialSegments
		self.activationLevel = None
		self.activationTime = None
		
		#if(biologicalSimulationDraw):
		self.nodeName = generateDendriticBranchName(conceptNode)
		self.branchIndex1 = branchIndex1	#not required
		self.branchIndex2 = branchIndex2	#not required
		self.conceptNode = conceptNode

class SequentialSegment:
	def __init__(self, conceptNode, branch, sequentialSegmentIndex):
		self.inputs = []
		self.activationLevel = None
		self.activationTime = None
		self.branch = branch
		
		#if(biologicalSimulationDraw):
		self.nodeName = generateSequentialSegmentName(conceptNode)
		self.sequentialSegmentIndex = None	#not required
		self.conceptNode = conceptNode	#not required (as can lookup SequentialSegment.branch.conceptNode) 

class SequentialSegmentInput:
	def __init__(self, conceptNode, SequentialSegment, sequentialSegmentInputIndex):
		self.input = None
		self.sequentialSegment = SequentialSegment
		self.firstInputInSequence = False
		
		#if(biologicalSimulationDraw):
		self.nodeName = generateSequentialSegmentInputName(conceptNode)
		self.sequentialSegmentInputIndex = None	#not required	#index record value not robust if inputs are removed (synaptic atrophy)
		self.conceptNode = conceptNode	#not required (as can lookup SequentialSegment.branch.conceptNode) 

def createDendriticTree(conceptNode, currentBranch, currentBranchIndex1, currentBranchIndex2, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments):
	currentBranch.subbranches = [DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, i) for i in range(numberOfBranches2)]	#[DendriticBranch(conceptNode, currentBranch, numberOfBranchSequentialSegments, currentBranchIndex1, currentBranchIndex2)]*numberOfBranches2
	if(currentBranchIndex1 < numberOfBranches1):
		for currentBranchIndex2, subbranch in enumerate(currentBranch.subbranches):	
			createDendriticTree(conceptNode, subbranch, currentBranchIndex1+1, currentBranchIndex2, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)
		
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
	

