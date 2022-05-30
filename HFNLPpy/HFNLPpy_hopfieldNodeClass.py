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

storeLeafNodesByLemma = True	#else store by word (morphology included)

graphNodeTypeLeaf = 1	#base/input neuron (network neuron)

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
		self.sourceConnectionList = []
		self.targetConnectionList = []

def generateHopfieldGraphNodeName(word, lemma):
	if(storeLeafNodesByLemma):
		nodeName = lemma
	else:
		nodeName = word
	return nodeName
		
