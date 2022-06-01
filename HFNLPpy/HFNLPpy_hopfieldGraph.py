"""HFNLPpy_hopfieldGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Hopfield Graph - generate hopfield graph/network based on textual input

- different instances (sentence clauses) are stored via a spatiotemporal connection index

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *


printVerbose = False

biologicalImplementation = True

drawHopfieldGraphSentence = False
if(drawHopfieldGraphSentence):
	import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawSentence
drawHopfieldGraphNetwork = True	#draw graph for entire network (not just sentence)
if(drawHopfieldGraphNetwork):
	import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawNetwork

networkConceptNodeDict = {}
networkSize = 0

def generateHopfieldGraphNetwork(articles):
	for sentenceIndex, sentence in enumerate(articles):
		generateHopfieldGraphSentenceString(sentenceIndex, sentence)	

def generateHopfieldGraphSentenceString(sentenceIndex, sentence):
	print("\n\ngenerateHopfieldGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	print("sentenceLength = ", sentenceLength)
	
	if(sentenceLength > 1):
		return generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence)

def generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence):
		
	currentTime = calculateActivationTime(sentenceIndex)

	if(drawHopfieldGraphSentence):
		ATNLPtf_hopfieldGraphDrawSentence.clearHopfieldGraph()
	if(drawHopfieldGraphNetwork):
		ATNLPtf_hopfieldGraphDrawNetwork.clearHopfieldGraph()
			
	sentenceConceptNodeList = []
	sentenceLength = len(tokenisedSentence)
				
	#declare graph nodes;	
	for w, token in enumerate(tokenisedSentence):	

		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		nodeName = generateHopfieldGraphNodeName(word, lemma)	
		if(graphNodeExists(nodeName)):
			conceptNode = getGraphNode(nodeName)
		else:
			#primary vars;
			wordVector = getTokenWordVector(token)	#numpy word vector
			#posTag = getTokenPOStag(token)	#not used
			activationTime = calculateActivationTime(sentenceIndex)
			nodeGraphType = graphNodeTypeConcept
			networkIndex = getNetworkIndex()
			conceptNode = HopfieldNode(networkIndex, nodeName, wordVector, nodeGraphType, activationTime)
			addNodeToGraph(conceptNode)
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.lemma)
		sentenceConceptNodeList.append(conceptNode)
			
		#connection vars;
		if(w > 0):
			spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)	#for biologicalImplementation: interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex/clause)
			if(biologicalImplementation):
				totalConceptsInSubsequence = 0
				for w2 in range(w-1):
					totalConceptsInSubsequence += 1
					#multiple connections/synapses are made between current neuron and ealier neurons in sequence, and synapse weights are adjusted such that the particular combination (or permutation if SANI synapses) will fire the neuron
					previousConceptNode = sentenceConceptNodeList[w2]
					weight = 1.0/totalConceptsInSubsequence	#for biologicalImplementation: interpret connection as unique synapse
					#print("weight = ", weight)
					addConnectionToNode(previousConceptNode, conceptNode, activationTime, spatioTemporalIndex, weight=weight)					
			else:
				previousConceptNode = sentenceConceptNodeList[w-1]
				addConnectionToNode(previousConceptNode, conceptNode, activationTime, spatioTemporalIndex)
		
		#if(drawHopfieldGraphSentence):
		#	ATNLPtf_hopfieldGraphDrawSentence.drawHopfieldGraphNode(instanceNode, w, treeLevel)	#done later

	
	if(drawHopfieldGraphSentence):
		for conceptNode in sentenceConceptNodeList:
			ATNLPtf_hopfieldGraphDrawSentence.drawHopfieldGraphNodeAndConnections(conceptNode, networkSize, drawGraph=False)
		print("ATNLPtf_hopfieldGraphDrawSentence.displayHopfieldGraph()")
		ATNLPtf_hopfieldGraphDrawSentence.displayHopfieldGraph()
		
	if(drawHopfieldGraphNetwork):
		ATNLPtf_hopfieldGraphDrawNetwork.drawHopfieldGraphNetwork(networkConceptNodeDict)
		print("ATNLPtf_hopfieldGraphDrawNetwork.displayHopfieldGraph()")
		ATNLPtf_hopfieldGraphDrawNetwork.displayHopfieldGraph()
	
	result = True			
	return result

def getGraphNode(nodeName):
	return networkConceptNodeDict[nodeName]
	
def graphNodeExists(nodeName):
	result = False
	if(nodeName in networkConceptNodeDict):
		result = True
	return result
	
def addNodeToGraph(conceptNode):
	global networkSize
	if(conceptNode.nodeName not in networkConceptNodeDict):
		networkConceptNodeDict[conceptNode.nodeName] = conceptNode
		networkSize = networkSize + 1
	else:
		print("addNodeToGraph error: conceptNode.nodeName already in networkConceptNodeDict")
		exit()
		
def connectionExists(nodeSource, nodeTarget):
	result = False
	if(nodeTarget.nodeName in nodeSource.targetConnectionDict):
		result = True
		#connectionList = nodeSource.targetConnectionDict[nodeTarget.nodeName]
		#for connection in connectionList:
		#	if(connection
	return result

def addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, weight=1.0, subsequenceConnection=False):
	connection = HopfieldConnection(nodeSource, nodeTarget, spatioTemporalIndex, activationTime, weight)
	#nodeSource.targetConnectionList.append(connection)
	#nodeTarget.sourceConnectionList.append(connection)
	createConnectionKeyIfNonExistant(nodeSource.targetConnectionDict, nodeTarget.nodeName)
	createConnectionKeyIfNonExistant(nodeTarget.sourceConnectionDict, nodeSource.nodeName)
	nodeSource.targetConnectionDict[nodeTarget.nodeName].append(connection)
	nodeTarget.sourceConnectionDict[nodeSource.nodeName].append(connection)
	#connection.subsequenceConnection = subsequenceConnection

def createConnectionKeyIfNonExistant(dic, key):
	if key not in dic:
		dic[key] = []	#create new empty list
		

			
#tokenisation:

def tokeniseSentence(sentence):
	tokenList = spacyWordVectorGenerator(sentence)
	return tokenList

def getTokenWord(token):
	word = token.text
	return word
	
def getTokenLemma(token):
	lemma = token.lemma_
	if(token.lemma_ == '-PRON-'):
		lemma = token.text	#https://stackoverflow.com/questions/56966754/how-can-i-make-spacy-not-produce-the-pron-lemma
	return lemma
		
def getTokenWordVector(token):
	wordVector = token.vector	#cpu: type numpy
	return wordVector

def getTokenPOStag(token):
	#nlp in context prediction only (not certain)
	posTag = token.pos_
	return posTag


#creation/access time:

def getNetworkIndex():
	networkIndex = len(networkConceptNodeDict)
	return networkIndex
		
#creation time
def calculateSpatioTemporalIndex(sentenceIndex):
	spatioTemporalIndex = sentenceIndex
	return spatioTemporalIndex

#last access time	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime
	
