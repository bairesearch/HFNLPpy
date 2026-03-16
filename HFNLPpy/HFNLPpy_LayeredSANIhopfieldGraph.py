"""SANIHFNLPpy_hopfieldGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
SANIHFNLP Hopfield Graph - generate hopfield graph/network based on textual input

- different instances (sentence clauses) are stored via a spatiotemporal connection index
- useAlgorithmDendriticPrototype: add contextual connections to emulate spatiotemporal index restriction (visualise theoretical biological connections without simulation)
- useAlgorithmDendriticSANI: simulate sequential activation of concept neurons and their dendritic input/synapses

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations
from HFNLPpy_globalDefs import *	
import random

if(useAlgorithmLayeredSANI):
	import HFNLPpy_LayeredSANIGraph
	from HFNLPpy_LayeredSANIGlobalDefs import LayeredSANIwordVectors
				
if(drawHopfieldGraph):
	if(drawHopfieldGraphSentence):
		import HFNLPpy_hopfieldGraphDraw as hopfieldGraphDrawSentence
	if(drawHopfieldGraphNetwork):
		import HFNLPpy_hopfieldGraphDraw as hopfieldGraphDrawNetwork


class HFconnectionGraphClass:
	def __init__(self):
		self.connectionMatrixMaxConcepts = None
		self.neuronNamelist = None
		self.neuronIDdict = {}

networkConceptNodeDict = {}
networkSize = 0
HFconnectionGraphObject = HFconnectionGraphClass()


def generateHopfieldGraphNetwork(articles, tokenizer):
	numberOfSentences = len(articles)

	trainSentence = True
	for sentenceIndex, sentence in enumerate(articles):
		if(seedHFnetworkSubsequenceType=="lastSentence"):
			if(sentenceIndex == numberOfSentences-1):
				trainSentence = False
		generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences, tokenizer, trainSentence)
		
	if(seedHFnetworkSubsequenceType=="all"):
		trainSentence = False
		for sentenceIndex, sentence in enumerate(articles):
			generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences, tokenizer, trainSentence)
	
def generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences, tokenizer, trainSentence):
	print("\n\ngenerateHopfieldGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence, tokenizer)
	sentenceLength = len(tokenisedSentence)
	#print("sentenceLength = ", sentenceLength)
			
	if(sentenceLength > 1):
		return generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences, trainSentence)
 
def regenerateGraphNodes():
	#regenerates graph nodes from a saved list
	if(linkSimilarConceptNodesWordnet and not tokenWordnetSynonymsFromLemma):
		sentence = ' '.join(HFconnectionGraphObject.neuronNamelist)
		tokenisedSentence = tokeniseSentence(sentence, None)
	for neuronID, nodeName in enumerate(HFconnectionGraphObject.neuronNamelist):	
		#token = tokenisedSentence[neuronID]
		networkIndex = getNetworkIndex()
		nodeGraphType = graphNodeTypeConcept
		wordVector = None	#getTokenWordVector(token)	#numpy word vector	#not used by useHFconnectionMatrix
		#posTag = getTokenPOStag(token)	#not used
		w = 0	#sentence artificial var (not used)
		sentenceIndex = 0	#sentence artificial var (not used)
		
		conceptNode = HopfieldNode(networkIndex, nodeName, nodeGraphType, wordVector, w, sentenceIndex)

		addNodeToGraph(conceptNode)
		if(printVerbose):
			print("create new conceptNode; ", conceptNode.nodeName)

def generateHopfieldGraphSentenceNodes(tokenisedSentence, sentenceIndex, sentenceConceptNodeList):
	#declare Hopfield graph nodes;	
	for w, token in enumerate(tokenisedSentence):	
		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		nodeName = generateHopfieldGraphNodeName(word, lemma)	
		if(graphNodeExists(nodeName)):
			conceptNode = getGraphNode(nodeName)
			#set sentence artificial vars (for sentence graph only, do not generalise to network graph);
			conceptNode.w = w
			conceptNode.sentenceIndex = sentenceIndex
			if(printVerbose):
				print("graphNodeExists; ", conceptNode.nodeName)
		else:
			#primary vars;
			nodeGraphType = graphNodeTypeConcept
			networkIndex = getNetworkIndex()
			#unused vars;
			wordVector = getTokenWordVector(token)	#numpy word vector
			#posTag = getTokenPOStag(token)	#not used
			
			conceptNode = HopfieldNode(networkIndex, nodeName, nodeGraphType, wordVector, w, sentenceIndex)

			addNodeToGraph(conceptNode)
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.nodeName)
		sentenceConceptNodeList.append(conceptNode)
		
def generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences, trainSentence):	
	activationTime = calculateActivationTime(sentenceIndex)
			
	sentenceConceptNodeList = []
	sentenceLength = len(tokenisedSentence)
		
	#declare Hopfield graph nodes;	
	generateHopfieldGraphSentenceNodes(tokenisedSentence, sentenceIndex, sentenceConceptNodeList)

	neuronIDdictNewlyAdded = {}
			
	if(trainSentence):
	
		#create Hopfield graph direct connections (for draw only)
		for w, token in enumerate(tokenisedSentence):
			conceptNode = sentenceConceptNodeList[w]
			if(w > 0):
				previousConceptNode = sentenceConceptNodeList[w-1]
				spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
				previousContextConceptNodesList = []
				if(useAlgorithmDendriticPrototype):
					for w2 in range(w-1):
						previousContextConceptNodesList.append(sentenceConceptNodeList[w2]) 
				createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)

		if(useAlgorithmLayeredSANI):
			#HFNLPpy_LayeredSANIGraph creates Hopfield graph direct connections
			sentenceConceptNodeList = HFNLPpy_LayeredSANIGraph.generateLayeredSANIGraphSentence(HFconnectionGraphObject, sentenceIndex, tokenisedSentence, sentenceConceptNodeList, networkConceptNodeDict)	#sentenceConceptNodeList is replaced with sentenceSANINodeList
			recalculateHopfieldGraphNetworkSize()
			#printConceptNodeList(sentenceConceptNodeList)

	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			hopfieldGraphDrawSentence.drawHopfieldGraphSentenceStatic(sentenceIndex, sentenceConceptNodeList, networkSize, drawHopfieldGraphPlot, drawHopfieldGraphSave)
		if(drawHopfieldGraphNetwork):
			hopfieldGraphDrawNetwork.drawHopfieldGraphNetworkStatic(sentenceIndex, networkConceptNodeDict, drawHopfieldGraphPlot, drawHopfieldGraphSave)

	result = True
	return result

#if(useDependencyParseTree):
							
def createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime):
	HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, contextConnection=False)
	#HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, activationTime, spatioTemporalIndex)

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
		#print("addNodeToGraph: conceptNode.nodeName = ", conceptNode.nodeName)
		networkConceptNodeDict[conceptNode.nodeName] = conceptNode
		networkSize += 1
	else:
		print("addNodeToGraph error: conceptNode.nodeName already in networkConceptNodeDict")
		exit()
		
			
#tokenisation:

def tokeniseSentence(sentence, tokenizer):
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
		


#networkSize;

def recalculateHopfieldGraphNetworkSize():
	global networkSize
	networkSize = len(networkConceptNodeDict)

def printConceptNodeList(sentenceConceptNodeList):
	for conceptNode in sentenceConceptNodeList:
		print("conceptNode = ", conceptNode.nodeName)
		
