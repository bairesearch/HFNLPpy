"""HFNLPpy_hopfieldGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Hopfield Graph - generate hopfield graph/network based on textual input

- different instances (sentence clauses) are stored via a spatiotemporal connection index
- SANIbiologicalPrototype: add contextual connections to emulate spatiotemporal index restriction (visualise theoretical biological connections without simulation)
- SANIbiologicalSimulation: simulate sequential activation of concept neurons and their dendritic input/synapses

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations
from HFNLPpy_globalDefs import *


if(ScanBiologicalSimulation):
	import torch as pt
	from HFNLPpy_ScanGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_ScanGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_ScanGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_Scan
	import HFNLPpy_ScanConnectionMatrix
elif(SANIbiologicalSimulation):
	from HFNLPpy_SANIGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	from HFNLPpy_SANIGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_SANIGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_SANIGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_SANI
	if(useDependencyParseTree):
		import HFNLPpy_SANISyntacticalGraph
	
if(useDependencyParseTree):
	import SPNLPpy_syntacticalGraph

if(drawHopfieldGraph):
	if(drawHopfieldGraphSentence):
		import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawSentence
	if(drawHopfieldGraphNetwork):
		import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawNetwork



HFconnectionMatrix = None
networkConceptNodeDict = {}
networkSize = 0
if(ScanBiologicalSimulation):
	HFconnectionMatrix = None
	neuronNamelist = None
	neuronIDdict = {}
		
def generateHopfieldGraphNetwork(articles):
	numberOfSentences = len(articles)
	
	if(HFNLPnonrandomSeed):
		np.random.seed(0)
		print("np.random.randint(0,9) = ", np.random.randint(0,9))
		#random.seed(0)	#not used
		#print("random.randint(0,9) = ", random.randint(0,9))

	if(ScanBiologicalSimulation):
		global HFconnectionMatrix, neuronNamelist
		neuronNamelist, HFconnectionMatrix = HFNLPpy_ScanConnectionMatrix.readHFconnectionMatrix()
		regenerateGraphNodes(neuronNamelist)

	if(seedHFnetworkSubsequence):
		verifySeedSentenceIsReplicant(articles, numberOfSentences)

	for sentenceIndex, sentence in enumerate(articles):
		generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences)	
		
	if(ScanBiologicalSimulation):
		HFNLPpy_ScanConnectionMatrix.writeHFconnectionMatrix(neuronNamelist, HFconnectionMatrix)

def generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences):
	print("\n\ngenerateHopfieldGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	print("sentenceLength = ", sentenceLength)
	
	if(sentenceLength > 1):
		return generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences)

def regenerateGraphNodes(neuronNamelist):
	#regenerates graph nodes from a saved list
	for neuronID, nodeName in enumerate(neuronNamelist):	
		w = 0	#sentence artificial var (not used)
		sentenceIndex = 0	#sentence artificial var (not used)
		
		wordVector = None	#getTokenWordVector(token)	#numpy word vector	#not used by ScanBiologicalSimulation
		#posTag = getTokenPOStag(token)	#not used
		activationTime = calculateActivationTime(sentenceIndex)	#not used by ScanBiologicalSimulation
		nodeGraphType = graphNodeTypeConcept
		networkIndex = getNetworkIndex()
		conceptNode = HopfieldNode(networkIndex, nodeName, wordVector, nodeGraphType, activationTime, SANIbiologicalSimulation, w, sentenceIndex)
		addNodeToGraph(conceptNode)
		if(ScanBiologicalSimulation):
			neuronIDdict[nodeName] = neuronID
		if(printVerbose):
			print("create new conceptNode; ", conceptNode.nodeName)
		
def generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences):
		
	activationTime = calculateActivationTime(sentenceIndex)

	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			ATNLPtf_hopfieldGraphDrawSentence.clearHopfieldGraph()
		if(drawHopfieldGraphNetwork):
			ATNLPtf_hopfieldGraphDrawNetwork.clearHopfieldGraph()
			
	sentenceConceptNodeList = []
	sentenceLength = len(tokenisedSentence)
		
	SPgraphHeadNode = None
	if(useDependencyParseTree):
		performIntermediarySyntacticalTransformation = False
		generateSyntacticalGraphNetwork = False
		sentenceLeafNodeList, _, SPgraphHeadNode = SPNLPpy_syntacticalGraph.generateSyntacticalGraphSentence(sentenceIndex, tokenisedSentence, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations)

	#declare graph nodes;	
	for w, token in enumerate(tokenisedSentence):	
		word = getTokenWord(token)
		lemma = getTokenLemma(token)
		nodeName = generateHopfieldGraphNodeName(word, lemma)	
		if(graphNodeExists(nodeName)):
			conceptNode = getGraphNode(nodeName)
			#set sentence artificial vars (for sentence graph only, do not generalise to network graph);
			conceptNode.w = w
			conceptNode.sentenceIndex = sentenceIndex
			if(ScanBiologicalSimulation):
				neuronID = neuronIDdict[nodeName]
		else:
			#primary vars;
			wordVector = getTokenWordVector(token)	#numpy word vector
			#posTag = getTokenPOStag(token)	#not used
			activationTime = calculateActivationTime(sentenceIndex)
			nodeGraphType = graphNodeTypeConcept
			networkIndex = getNetworkIndex()
			conceptNode = HopfieldNode(networkIndex, nodeName, wordVector, nodeGraphType, activationTime, SANIbiologicalSimulation, w, sentenceIndex)
			addNodeToGraph(conceptNode)
			if(ScanBiologicalSimulation):
				neuronNamelist.append(nodeName)
				neuronID = networkIndex
				neuronIDdict[nodeName] = neuronID
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.nodeName)
		if(ScanBiologicalSimulation):
			if(w > 0):
				sourceNeuronID = neuronIDprevious
				targetNeuronID = neuronID
				HFNLPpy_ScanConnectionMatrix.updateOrAddConnectionToGraph(neuronNamelist, HFconnectionMatrix, sourceNeuronID, targetNeuronID)
			neuronIDprevious = neuronID
			
		sentenceConceptNodeList.append(conceptNode)

	if(ScanBiologicalSimulation):
		# Set the initial activation state for each neuron at time t
		HFconnectionMatrix.activation_state = pt.zeros(len(neuronNamelist), dtype=pt.float)
	
	trainSentence = True
	if(sentenceIndex == numberOfSentences-1):
		if(seedHFnetworkSubsequence):
			trainSentence = False
			seedSentenceConceptNodeList = sentenceConceptNodeList
			if(ScanBiologicalSimulation):
				HFNLPpy_Scan.seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, seedSentenceConceptNodeList, numberOfSentences)
			else:
				HFNLPpy_SANI.seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences)			
	if(ScanBiologicalSimulation):
		if(trainSentence):
			print("HFNLPpy_SANI.simulateBiologicalHFnetwork")
			HFNLPpy_Scan.trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, sentenceConceptNodeList, numberOfSentences)
	elif(SANIbiologicalSimulation):
		if(trainSentence):		
			if(useDependencyParseTree):
				print("HFNLPpy_SANISyntacticalGraph.simulateBiologicalHFnetworkSP")
				HFNLPpy_SANISyntacticalGraph.trainBiologicalHFnetworkSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations)		
			else:
				print("HFNLPpy_SANI.simulateBiologicalHFnetwork")
				HFNLPpy_SANI.trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences)			
	else:
		#connection vars;
		if(useDependencyParseTree):
			spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
			connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, SPgraphHeadNode, spatioTemporalIndex, activationTime)
		else:
			for w, token in enumerate(tokenisedSentence):
				if(w > 0):
					conceptNode = sentenceConceptNodeList[w]
					previousConceptNode = sentenceConceptNodeList[w-1]
					spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
					previousContextConceptNodesList = []
					if(SANIbiologicalPrototype):
						for w2 in range(w-1):
							previousContextConceptNodesList.append(sentenceConceptNodeList[w2]) 
					createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)
	
	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			fileName = generateHopfieldGraphFileName(True, sentenceIndex)
			ATNLPtf_hopfieldGraphDrawSentence.drawHopfieldGraphSentence(sentenceConceptNodeList, networkSize)
			print("ATNLPtf_hopfieldGraphDrawSentence.displayHopfieldGraph()")
			ATNLPtf_hopfieldGraphDrawSentence.displayHopfieldGraph(drawHopfieldGraphPlot, drawHopfieldGraphSave, fileName)
		if(drawHopfieldGraphNetwork):
			fileName = generateHopfieldGraphFileName(False, sentenceIndex)
			ATNLPtf_hopfieldGraphDrawNetwork.drawHopfieldGraphNetwork(networkConceptNodeDict)
			print("ATNLPtf_hopfieldGraphDrawNetwork.displayHopfieldGraph()")
			ATNLPtf_hopfieldGraphDrawNetwork.displayHopfieldGraph(drawHopfieldGraphPlot, drawHopfieldGraphSave, fileName)

	result = True			
	return result



#if(useDependencyParseTree):
	
def connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, DPgovernorNode, spatioTemporalIndex, activationTime):
	for DPdependentNode in DPgovernorNode.DPdependentList:
		previousContextConceptNodesList = []
		conceptNode, previousConceptNode = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList)
		createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)
		connectHopfieldGraphSentence(sentenceConceptNodeList, DPdependentNode, spatioTemporalIndex, activationTime)

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList):
	conceptNode = sentenceConceptNodeList[DPgovernorNode.w]
	previousConceptNode = sentenceConceptNodeList[DPdependentNode.w]
	if(SANIbiologicalPrototype):
		for DPdependentNode2 in DPdependentNode.DPdependentList:
			previousContextConceptNode = sentenceConceptNodeList[DPdependentNode2.w]
			previousContextConceptNodesList.append(previousContextConceptNode)
			_, _ = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode2, previousContextConceptNodesList)
	return conceptNode, previousConceptNode


def createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime):
	HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, activationTime, spatioTemporalIndex)
	
	if(SANIbiologicalPrototype):
		totalConceptsInSubsequence = 0
		for previousContextIndex, previousContextConceptNode in enumerate(previousContextConceptNodesList):
			totalConceptsInSubsequence += 1
			#multiple connections/synapses are made between current neuron and ealier neurons in sequence, and synapse weights are adjusted such that the particular combination (or permutation if SANI synapses) will fire the neuron
			weight = 1.0/totalConceptsInSubsequence	#for SANIbiologicalPrototype: interpret connection as unique synapse
			#print("weight = ", weight)
			HFNLPpy_hopfieldOperations.addConnectionToNode(previousContextConceptNode, conceptNode, activationTime, spatioTemporalIndex, SANIbiologicalPrototype=SANIbiologicalPrototype, weight=weight, contextConnection=True, contextConnectionSANIindex=previousContextIndex)					


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
		


def generateHopfieldGraphFileName(sentenceOrNetwork, sentenceIndex=None):
	fileName = "hopfieldGraph"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + str(sentenceIndex)
	return fileName
	

#subsequence seed	

def verifySeedSentenceIsReplicant(articles, numberOfSentences):
	result = False
	if(seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant):
		seedSentence = articles[numberOfSentences-1]
		for sentenceIndex in range(numberOfSentences-1):
			sentence = articles[sentenceIndex]
			if(compareSentenceStrings(seedSentence, sentence)):
				result = True
		if(not result):
			print("verifySeedSentenceIsReplicant warning: seedSentence (last sentence in dataset) was not found eariler in dataset (sentences which are being trained)")
	return result
	
def compareSentenceStrings(sentence1, sentence2):
	result = True
	if(len(sentence1) == len(sentence2)):
		for wordIndex in range(len(sentence1)):
			word1 = sentence1[wordIndex]
			word2 = sentence1[wordIndex]
			if(word1 != word2):
				result = False
	else:
		result = False	
	return result
	
	
