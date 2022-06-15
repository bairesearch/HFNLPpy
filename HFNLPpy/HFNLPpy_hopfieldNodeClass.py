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
numberOfBranches2 = 3	#number of new horizontal branches created at each vertical branch
#[3,3,3]	#number of new horizontal branches created at each vertical branch
numberOfBranchSequentialSegments = 1	#1+	#sequential inputs (FUTURE: if > 1: each branch segment may require sequential inputs)
#numberOfBranchSequentialSegmentInputs = 1	#1+	#nonSequentialInputs	#in current implementation (non-parallel generative network) number of inputs at sequential segment is dynamically increased on demand #not used; currently encode infinite number of


class HopfieldNode:
	def __init__(self, networkIndex, nodeName, wordVector, nodeGraphType, activationTime):
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

		#if(biologicalSimulation):
		#self.branch1ActivationLevel = createMultilist(0, [numberOfBranches1])
		self.branch2ActivationLevel = createMultilist(0, [numberOfBranches1, numberOfBranches2])
		self.branchSequentialSegmentActivationLevel = createMultilist(0, [numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments])
		#self.branchSequentialSegmentInputActivationLevel = [[[[]]]]
		#self.branch1ActivationTime = createMultilist(0, [numberOfBranches1])
		self.branch2ActivationTime = createMultilist(0, [numberOfBranches1, numberOfBranches2])
		self.branchSequentialSegmentActivationTime = createMultilist(0, [numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments])
		#self.branchSequentialSegmentInputActivationTime = [[[[]]]]
		self.branchSequentialSegmentInputSize = createMultilist(0, [numberOfBranches1, numberOfBranches2, numberOfBranchSequentialSegments])	#[[[]]]	#number of inputs at branch sequential segment
		

#last access time	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime
	
#creation time
def calculateSpatioTemporalIndex(sentenceIndex):
	#for biologicalImplementation: e.g. 1) interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex/clause); or 2) store spatiotemporal index synapses on separate dendritic branch
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
		
def createMultilist(level, listSizeList):
	lst = []
	if(level < len(listSizeList)):
		listSize = listSizeList[level]
		for i in range(listSize):
			sublist = createMultilist(level+1, listSizeList)
			lst.append(sublist)
	return lst
