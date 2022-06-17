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
	def __init__(self, networkIndex, nodeName, wordVector, nodeGraphType, activationTime, biologicalSimulation):
		#primary vars;
		self.networkIndex = networkIndex
		self.nodeName = nodeName
		self.wordVector = wordVector	#numpy array
		#self.posTag = posTag	#nlp in context prediction only (not certain)
		self.graphNodeType = nodeGraphType
		self.activationTime = activationTime	#last activation time (used to calculate recency)	#not currently used
		
		#connection vars;
		self.sourceConnectionDict = {}
		self.targetConnectionDict = {}
		#self.sourceConnectionList = []
		#self.targetConnectionList = []

		if(biologicalSimulation):
			dendriticTreeHeadBranch = DendriticBranch(None, numberOfBranchSequentialSegments)
			createDendriticTree(dendriticTreeHeadBranch, 0, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)
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
	def __init__(self, parentBranch, numberOfBranchSequentialSegments):
		self.parentBranch = parentBranch
		self.subbranches = []
		self.sequentialSegments = [SequentialSegment(self)]*numberOfBranchSequentialSegments
		self.activationLevel = None
		self.activationTime = None

		#def __init__(self, numberOfBranches2, numberOfBranchSequentialSegments):
		#	self.subbranches = [DendriticBranch]*numberOfBranches2
		#	self.sequentialSegments = [SequentialSegment]*numberOfBranchSequentialSegments

class SequentialSegment:
	def __init__(self, branch):
		self.inputs = []
		self.activationLevel = None
		self.activationTime = None
		self.branch = branch

class SequentialSegmentInput:
	def __init__(self, SequentialSegment):
		self.input = None
		self.sequentialSegment = SequentialSegment
		self.firstInputInSequence = False
		
def createDendriticTree(currentBranch, currentBranchIndex1, numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments):
	currentBranch.subbranches = [DendriticBranch(currentBranch, numberOfBranchSequentialSegments)]*numberOfBranches2
	if(currentBranchIndex1 < numberOfBranches1):
		for subbranch in currentBranch.subbranches:	
			createDendriticTree(subbranch, currentBranchIndex1+1,  numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments)
		
