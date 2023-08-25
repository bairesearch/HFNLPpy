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

printVerbose = True

SCANbiologicalSimulation = True
SANIbiologicalPrototype = False	#add contextual connections to emulate primary connection spatiotemporal index restriction (visualise biological connections without simulation)
SANIbiologicalSimulation = False	#simulate sequential activation of dendritic input 
useDependencyParseTree = False

if(SCANbiologicalSimulation):
	import torch as pt
	from HFNLPpy_SCANbiologicalSimulationGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_SCANbiologicalSimulationGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_SCANbiologicalSimulationGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_SCANbiologicalSimulation
	import HFNLPpy_SCANbiologicalSimulationConnectionMatrix
	useDependencyParseTree = False
elif(SANIbiologicalSimulation):
	from HFNLPpy_SANIbiologicalSimulationGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	from HFNLPpy_SANIbiologicalSimulationGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_SANIbiologicalSimulationGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_SANIbiologicalSimulationGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_SANIbiologicalSimulation
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		useDependencyParseTree = True
	else:
		useDependencyParseTree = False
	if(useDependencyParseTree):
		import HFNLPpy_SANIbiologicalSimulationSyntacticalGraph
else:
	useDependencyParseTree = True
		
if(useDependencyParseTree):
	import SPNLPpy_syntacticalGraph
	if(not SPNLPpy_syntacticalGraph.useSPNLPcustomSyntacticalParser):
		SPNLPpy_syntacticalGraph.SPNLPpy_syntacticalGraphConstituencyParserFormal.initalise(spacyWordVectorGenerator)
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		identifySyntacticalDependencyRelations = True	#optional
		#configuration notes:
		#some constituency parse trees are binary trees eg useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphConstituencyParserWordVectors (or Stanford constituency parser with binarize option etc), other constituency parsers are non-binary trees; eg !useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphConstituencyParserFormal (Berkeley neural parser)
		#most dependency parse trees are non-binary trees eg useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphDependencyParserWordVectors / !useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphDependencyParserWordVectors (spacy dependency parser)
		#if identifySyntacticalDependencyRelations False (use constituency parser), synapses are created in most distal branch segments only - requires dendritic tree propagation algorithm mod	
		#if supportForNonBinarySubbranchSize True, dendriticTree will support 2+ subbranches, with inputs adjusted by weight depending on number of subbranches expected to be activated
		#if supportForNonBinarySubbranchSize False, constituency/dependency parser must produce a binary parse tree (or disable biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect)
		if(not identifySyntacticalDependencyRelations):
			print("SANIbiologicalSimulation constituency parse tree support has not yet been implemented: synapses are created in most distal branch segments only - requires dendritic tree propagation algorithm mod")
			exit()
	else:
		identifySyntacticalDependencyRelations = True	#mandatory 	#standard hopfield NLP graph requires words are connected (no intermediary constituency parse tree syntax nodes) 

drawHopfieldGraph = True
if(drawHopfieldGraph):
	drawHopfieldGraphPlot = True
	drawHopfieldGraphSave = False
	drawHopfieldGraphSentence = False
	if(drawHopfieldGraphSentence):
		import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawSentence
	drawHopfieldGraphNetwork = True	#default: True	#draw graph for entire network (not just sentence)
	if(drawHopfieldGraphNetwork):
		import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawNetwork

HFconnectionMatrix = None
networkConceptNodeDict = {}
networkSize = 0
if(SCANbiologicalSimulation):
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

	if(SCANbiologicalSimulation):
		global HFconnectionMatrix, neuronNamelist
		neuronNamelist, HFconnectionMatrix = HFNLPpy_SCANbiologicalSimulationConnectionMatrix.readHFconnectionMatrix()
		regenerateGraphNodes(neuronNamelist)

	if(seedHFnetworkSubsequence):
		verifySeedSentenceIsReplicant(articles, numberOfSentences)

	for sentenceIndex, sentence in enumerate(articles):
		generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences)	
		
	if(SCANbiologicalSimulation):
		HFNLPpy_SCANbiologicalSimulationConnectionMatrix.writeHFconnectionMatrix(neuronNamelist, HFconnectionMatrix)

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
		
		wordVector = None	#getTokenWordVector(token)	#numpy word vector	#not used by SCANbiologicalSimulation
		#posTag = getTokenPOStag(token)	#not used
		activationTime = calculateActivationTime(sentenceIndex)	#not used by SCANbiologicalSimulation
		nodeGraphType = graphNodeTypeConcept
		networkIndex = getNetworkIndex()
		conceptNode = HopfieldNode(networkIndex, nodeName, wordVector, nodeGraphType, activationTime, SANIbiologicalSimulation, w, sentenceIndex)
		addNodeToGraph(conceptNode)
		if(SCANbiologicalSimulation):
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
			if(SCANbiologicalSimulation):
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
			if(SCANbiologicalSimulation):
				neuronNamelist.append(nodeName)
				neuronID = networkIndex
				neuronIDdict[nodeName] = neuronID
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.nodeName)
		if(SCANbiologicalSimulation):
			if(w > 0):
				sourceNeuronID = neuronIDprevious
				targetNeuronID = neuronID
				HFNLPpy_SCANbiologicalSimulationConnectionMatrix.updateOrAddConnectionToGraph(neuronNamelist, HFconnectionMatrix, sourceNeuronID, targetNeuronID)
			neuronIDprevious = neuronID
			
		sentenceConceptNodeList.append(conceptNode)

	if(SCANbiologicalSimulation):
		# Set the initial activation state for each neuron at time t
		HFconnectionMatrix.activation_state = pt.zeros(len(neuronNamelist), dtype=pt.float)
	
	trainSentence = True
	if(sentenceIndex == numberOfSentences-1):
		if(seedHFnetworkSubsequence):
			trainSentence = False
			seedSentenceConceptNodeList = sentenceConceptNodeList
			if(SCANbiologicalSimulation):
				HFNLPpy_SCANbiologicalSimulation.seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, seedSentenceConceptNodeList, numberOfSentences)
			else:
				HFNLPpy_SANIbiologicalSimulation.seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences)			
	if(SCANbiologicalSimulation):
		if(trainSentence):
			print("HFNLPpy_SANIbiologicalSimulation.simulateBiologicalHFnetwork")
			HFNLPpy_SCANbiologicalSimulation.trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, sentenceConceptNodeList, numberOfSentences)
	elif(SANIbiologicalSimulation):
		if(trainSentence):		
			if(useDependencyParseTree):
				print("HFNLPpy_SANIbiologicalSimulationSyntacticalGraph.simulateBiologicalHFnetworkSP")
				HFNLPpy_SANIbiologicalSimulationSyntacticalGraph.trainBiologicalHFnetworkSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations)		
			else:
				print("HFNLPpy_SANIbiologicalSimulation.simulateBiologicalHFnetwork")
				HFNLPpy_SANIbiologicalSimulation.trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences)			
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
	
	
