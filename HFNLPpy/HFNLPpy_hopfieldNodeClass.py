"""HFNLPpy_hopfieldNodeClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Hopfield Node Class

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

from HFNLPpy_globalDefs import *
from HFNLPpy_SANINode import biologicalSimulationNodePropertiesInitialisation

	
storeConceptNodesByLemma = True	#else store by word (morphology included)
convertLemmasToLowercase = True #required to ensure that capitalised start of sentence words are always converted to the same lemmas (irrespective of inconsistent propernoun detection)

graphNodeTypeConcept = 1	#base/input neuron (network neuron)

#if(biologicalImplementationReuseSynapticSubstrateForIdenticalSubsequences):
graphNodeTypeStart = 5	#start of sequence - used by biologicalImplementationReuseSynapticSubstrateForIdenticalSubsequences only
nodeNameStart = "SEQUENCESTARTNODE"

class HopfieldNode:
	def __init__(self, networkIndex, nodeName, wordVector, nodeGraphType, activationTime, w, sentenceIndex):
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

		if(SANIbiologicalSimulation):
			biologicalSimulationNodePropertiesInitialisation(self)
		if(ScanBiologicalSimulation):
			#self.activationLevel = 0
			self.activationState = False
			self.activationStateFiltered = False

								
#last access time	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime
	
#creation time
def calculateSpatioTemporalIndex(sentenceIndex):
	#for SANIbiologicalPrototype: e.g. 1) interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex/clause); or 2) store spatiotemporal index synapses on separate dendritic branch
	spatioTemporalIndex = sentenceIndex
	return spatioTemporalIndex

def createConnectionKeyIfNonExistant(dic, key):
	if key not in dic:
		dic[key] = []	#create new empty list
		
def generateHopfieldGraphNodeName(word, lemma):
	if(convertLemmasToLowercase):
		lemma = lemma.lower()
		
	if(storeConceptNodesByLemma):
		nodeName = lemma
	else:
		nodeName = word
	return nodeName


