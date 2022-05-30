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

drawHopfieldGraphSentence = False
if(drawHopfieldGraphSentence):
	import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawSentence
drawHopfieldGraphNetwork = True	#draw graph for entire network (not just sentence)
if(drawHopfieldGraphNetwork):
	import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawNetwork

networkLeafNodeDict = {}
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
	
	global networkSize
	
	currentTime = calculateActivationTime(sentenceIndex)

	if(drawHopfieldGraphSentence):
		ATNLPtf_hopfieldGraphDrawSentence.clearHopfieldGraph()
	if(drawHopfieldGraphNetwork):
		ATNLPtf_hopfieldGraphDrawNetwork.clearHopfieldGraph()
			
	sentenceLeafNodeList = []
	
	sentenceLength = len(tokenisedSentence)
	
	#declare graph nodes;
	for w, token in enumerate(tokenisedSentence):	

		#primary vars;
		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		nodeName = generateHopfieldGraphNodeName(word, lemma)
		wordVector = getTokenWordVector(token)	#numpy word vector
		#posTag = getTokenPOStag(token)	#not used
		activationTime = calculateActivationTime(sentenceIndex)
		nodeGraphType = graphNodeTypeLeaf
		networkIndex = getNetworkIndex()
		
		#add instance to sentenceLeafNodeList
		leafNode = HopfieldNode(networkIndex, nodeName, wordVector, nodeGraphType, activationTime)
		if(printVerbose):
			print("create new leafNode; ", leafNode.lemma)
		sentenceLeafNodeList.append(leafNode)
		
		#connection vars;
		if(w > 0):
			previousLeafNode = sentenceLeafNodeList[w-1]
			spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
			addConnectionToNode(previousLeafNode, leafNode, spatioTemporalIndex, activationTime)
		
		networkLeafNodeDict[nodeName] = leafNode
		networkSize = networkSize + 1
		
		#if(drawHopfieldGraphSentence):
		#	ATNLPtf_hopfieldGraphDrawSentence.drawHopfieldGraphNode(instanceNode, w, treeLevel)	#done later

	
	if(drawHopfieldGraphSentence):
		for leafNode in sentenceLeafNodeList:
			ATNLPtf_hopfieldGraphDrawSentence.drawHopfieldGraphNodeAndConnections(leafNode, networkSize, drawGraph=False)
		print("ATNLPtf_hopfieldGraphDrawSentence.displayHopfieldGraph()")
		ATNLPtf_hopfieldGraphDrawSentence.displayHopfieldGraph()
		
	if(drawHopfieldGraphNetwork):
		ATNLPtf_hopfieldGraphDrawNetwork.drawHopfieldGraphNetwork(networkLeafNodeDict)
		print("ATNLPtf_hopfieldGraphDrawNetwork.displayHopfieldGraph()")
		ATNLPtf_hopfieldGraphDrawNetwork.displayHopfieldGraph()
	
	result = True			
	return result



def addConnectionToNode(nodeSource, nodeTarget, spatioTemporalIndex, activationTime):
	connection = HopfieldConnection(nodeSource, nodeTarget, spatioTemporalIndex, activationTime)
	nodeSource.targetConnectionList.append(connection)
	nodeTarget.sourceConnectionList.append(connection)
	
		
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
	networkIndex = len(networkLeafNodeDict)
	return networkIndex
		
#creation time
def calculateSpatioTemporalIndex(sentenceIndex):
	spatioTemporalIndex = sentenceIndex
	return spatioTemporalIndex

#last access time	
def calculateActivationTime(sentenceIndex):
	activationTime = sentenceIndex
	return activationTime
	
