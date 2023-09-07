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
- useAlgorithmDendriticSANIbiologicalPrototype: add contextual connections to emulate spatiotemporal index restriction (visualise theoretical biological connections without simulation)
- useAlgorithmDendriticSANIbiologicalSimulation: simulate sequential activation of concept neurons and their dendritic input/synapses

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations
from HFNLPpy_globalDefs import *

if(useAlgorithmLayeredSANIbiologicalSimulation):
	#import torch as pt
	import SANIHFNLPpy_LayeredSANI
	import SANIHFNLPpy_LayeredSANINode
	from SANIHFNLPpy_LayeredSANIGlobalDefs import SANInumberOfLayersMax
if(useAlgorithmScanBiologicalSimulation):
	import torch as pt
	from HFNLPpy_ScanGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_ScanGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_ScanGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_Scan
	import HFNLPpy_ScanConnectionMatrix
elif(useAlgorithmDendriticSANIbiologicalSimulation):
	from HFNLPpy_DendriticSANIGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	from HFNLPpy_DendriticSANIGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_DendriticSANIGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_DendriticSANIGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_DendriticSANI
	if(useDependencyParseTree):
		import HFNLPpy_DendriticSANISyntacticalGraph
	
if(useDependencyParseTree):
	import SPNLPpy_globalDefs
	import SPNLPpy_syntacticalGraph
	if(not SPNLPpy_globalDefs.useSPNLPcustomSyntacticalParser):
		SPNLPpy_syntacticalGraph.SPNLPpy_syntacticalGraphConstituencyParserFormal.initalise(spacyWordVectorGenerator)
		
if(drawHopfieldGraph):
	if(drawHopfieldGraphSentence):
		import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawSentence
	if(drawHopfieldGraphNetwork):
		import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawNetwork



HFconnectionGraph = None
networkConceptNodeDict = {}
networkSize = 0
if(useAlgorithmScanBiologicalSimulation):
	HFconnectionGraph = None
	neuronNamelist = None
	neuronIDdict = {}
if(useAlgorithmLayeredSANIbiologicalSimulation):
	SANIlayerList = []
	for layerIndex in range(SANInumberOfLayersMax):
		SANIlayerList.append(SANIHFNLPpy_LayeredSANINode.SANIlayer(layerIndex))
	
def generateHopfieldGraphNetwork(articles):
	numberOfSentences = len(articles)
	
	if(HFNLPnonrandomSeed):
		np.random.seed(0)
		print("np.random.randint(0,9) = ", np.random.randint(0,9))
		#random.seed(0)	#not used
		#print("random.randint(0,9) = ", random.randint(0,9))

	if(useAlgorithmScanBiologicalSimulation):
		global HFconnectionGraph, neuronNamelist
		neuronNamelist, HFconnectionGraph = HFNLPpy_ScanConnectionMatrix.readHFconnectionMatrix()
		regenerateGraphNodes(neuronNamelist)

	if(seedHFnetworkSubsequence):
		verifySeedSentenceIsReplicant(articles, numberOfSentences)

	for sentenceIndex, sentence in enumerate(articles):
		generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences)	
		
	if(useAlgorithmScanBiologicalSimulation):
		HFNLPpy_ScanConnectionMatrix.writeHFconnectionMatrix(neuronNamelist, HFconnectionGraph)

def generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences):
	print("\n\ngenerateHopfieldGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	#print("sentenceLength = ", sentenceLength)
	
	if(sentenceLength > 1):
		return generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences)

def regenerateGraphNodes(neuronNamelist):
	#regenerates graph nodes from a saved list
	for neuronID, nodeName in enumerate(neuronNamelist):	
		networkIndex = getNetworkIndex()
		nodeGraphType = graphNodeTypeConcept
		wordVector = None	#getTokenWordVector(token)	#numpy word vector	#not used by useAlgorithmScanBiologicalSimulation
		#posTag = getTokenPOStag(token)	#not used
		w = 0	#sentence artificial var (not used)
		sentenceIndex = 0	#sentence artificial var (not used)
		
		conceptNode = HopfieldNode(networkIndex, nodeName, nodeGraphType, wordVector, w, sentenceIndex)
		if(useAlgorithmLayeredSANIbiologicalSimulation):
			conceptNode = networkIndex
		if(useAlgorithmScanBiologicalSimulation):
			neuronIDdict[nodeName] = neuronID
		addNodeToGraph(conceptNode)
		if(printVerbose):
			print("create new conceptNode; ", conceptNode.nodeName)
		
def generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences):	
	activationTime = calculateActivationTime(sentenceIndex)
			
	sentenceConceptNodeList = []
	sentenceLength = len(tokenisedSentence)
		
	SPgraphHeadNode = None
	if(useDependencyParseTree):
		performIntermediarySyntacticalTransformation = False
		generateSyntacticalGraphNetwork = False
		sentenceLeafNodeList, _, SPgraphHeadNode = SPNLPpy_syntacticalGraph.generateSyntacticalGraphSentence(sentenceIndex, tokenisedSentence, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations)

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
		else:
			#primary vars;
			nodeGraphType = graphNodeTypeConcept
			networkIndex = getNetworkIndex()
			#unused vars;
			wordVector = getTokenWordVector(token)	#numpy word vector
			#posTag = getTokenPOStag(token)	#not used
			
			conceptNode = HopfieldNode(networkIndex, nodeName, nodeGraphType, wordVector, w, sentenceIndex)
			if(useAlgorithmLayeredSANIbiologicalSimulation):
				conceptNode.SANIlayerNeuronID = networkIndex
				conceptNode.SANIlayerIndex = 0
			if(useAlgorithmScanBiologicalSimulation):
				neuronNamelist.append(nodeName)
				neuronID = networkIndex
				neuronIDdict[nodeName] = neuronID
			addNodeToGraph(conceptNode)
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.nodeName)
		sentenceConceptNodeList.append(conceptNode)

	if(useAlgorithmLayeredSANIbiologicalSimulation):
		SANIlayerList[0].networkSANINodeList = list(networkConceptNodeDict.values())
		SANIlayerList[0].sentenceSANINodeList = sentenceConceptNodeList
		
	trainSentence = True
	if(sentenceIndex == numberOfSentences-1):
		if(seedHFnetworkSubsequence):
			trainSentence = False
			
	if(trainSentence):
		#create Hopfield graph connections (non-useAlgorithmDendriticSANIbiologicalSimulation);
		if(useAlgorithmLayeredSANIbiologicalSimulation):
			SANIHFNLPpy_LayeredSANI.updateLayeredSANIgraph(SANIlayerList, sentenceIndex)
		elif(useAlgorithmDendriticSANIbiologicalSimulation):
			#useAlgorithmDendriticSANIbiologicalSimulation:HFNLPpy_DendriticSANIGenerate:addPredictiveSequenceToNeuron:addPredictiveSynapseToNeuron:addConnectionToNode creates connections between hopfield objects (with currentSequentialSegmentInput object)
				if(useDependencyParseTree):
					print("HFNLPpy_DendriticSANISyntacticalGraph.simulateBiologicalHFnetworkSP")
					HFNLPpy_DendriticSANISyntacticalGraph.trainBiologicalHFnetworkSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations)		
				else:
					print("HFNLPpy_DendriticSANI.simulateBiologicalHFnetwork")
					HFNLPpy_DendriticSANI.trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences)	
		else:
			if(useDependencyParseTree):
				spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
				connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, SPgraphHeadNode, spatioTemporalIndex, activationTime)
			else:
				for w, token in enumerate(tokenisedSentence):
					conceptNode = sentenceConceptNodeList[w]
					if(w > 0):
						previousConceptNode = sentenceConceptNodeList[w-1]
						spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
						previousContextConceptNodesList = []
						if(useAlgorithmDendriticSANIbiologicalPrototype):
							for w2 in range(w-1):
								previousContextConceptNodesList.append(sentenceConceptNodeList[w2]) 
						createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)
					if(useAlgorithmScanBiologicalSimulation):
						neuronID = neuronIDdict[conceptNode.nodeName]
						if(w > 0):
							sourceNeuronID = neuronIDprevious
							targetNeuronID = neuronID
							HFNLPpy_ScanConnectionMatrix.updateOrAddConnectionToGraph(neuronNamelist, HFconnectionGraph, sourceNeuronID, targetNeuronID)
						neuronIDprevious = neuronID
	else:
		#predict Hopfield graph flow;
		seedSentenceConceptNodeList = sentenceConceptNodeList
		if(useAlgorithmScanBiologicalSimulation):
			HFconnectionGraph.activationLevel = pt.zeros(len(neuronNamelist), dtype=pt.float)	# Set the initial activation level for each neuron at time t
			HFconnectionGraph.activationState = pt.zeros(len(neuronNamelist), dtype=pt.bool)	# Set the initial activation state for each neuron at time t
			HFNLPpy_Scan.seedBiologicalHFnetwork(networkConceptNodeDict, networkSize, sentenceIndex, neuronNamelist, neuronIDdict, HFconnectionGraph, seedSentenceConceptNodeList, numberOfSentences)
		else:
			HFNLPpy_DendriticSANI.seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences)			
	
	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			ATNLPtf_hopfieldGraphDrawSentence.drawHopfieldGraphSentenceStatic(sentenceIndex, sentenceConceptNodeList, networkSize, drawHopfieldGraphPlot, drawHopfieldGraphSave)
		if(drawHopfieldGraphNetwork):
			ATNLPtf_hopfieldGraphDrawNetwork.drawHopfieldGraphNetworkStatic(sentenceIndex, networkConceptNodeDict, drawHopfieldGraphPlot, drawHopfieldGraphSave)

	result = True
	return result



#if(useDependencyParseTree):
	
def connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, DPgovernorNode, spatioTemporalIndex, activationTime):
	for DPdependentNode in DPgovernorNode.DPdependentList:
		previousContextConceptNodesList = []
		conceptNode, previousConceptNode = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList)
		createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)
		connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, DPdependentNode, spatioTemporalIndex, activationTime)

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList):
	conceptNode = sentenceConceptNodeList[DPgovernorNode.w]
	previousConceptNode = sentenceConceptNodeList[DPdependentNode.w]
	if(useAlgorithmDendriticSANIbiologicalPrototype):
		for DPdependentNode2 in DPdependentNode.DPdependentList:
			previousContextConceptNode = sentenceConceptNodeList[DPdependentNode2.w]
			previousContextConceptNodesList.append(previousContextConceptNode)
			_, _ = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode2, previousContextConceptNodesList)
	return conceptNode, previousConceptNode


def createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime):
	HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, contextConnection=False)
	#HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, activationTime, spatioTemporalIndex)
	
	if(useAlgorithmDendriticSANIbiologicalPrototype):
		totalConceptsInSubsequence = 0
		for previousContextIndex, previousContextConceptNode in enumerate(previousContextConceptNodesList):
			totalConceptsInSubsequence += 1
			#multiple connections/synapses are made between current neuron and ealier neurons in sequence, and synapse weights are adjusted such that the particular combination (or permutation if SANI synapses) will fire the neuron
			weight = 1.0/totalConceptsInSubsequence	#for useAlgorithmDendriticSANIbiologicalPrototype: interpret connection as unique synapse
			#print("weight = ", weight)
			HFNLPpy_hopfieldOperations.addConnectionToNode(previousContextConceptNode, conceptNode, activationTime, spatioTemporalIndex, useAlgorithmDendriticSANIbiologicalPrototype=useAlgorithmDendriticSANIbiologicalPrototype, weight=weight, contextConnection=True, contextConnectionSANIindex=previousContextIndex)					

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
	
	
