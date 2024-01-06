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
if(tokeniseSubwords):
	import HFNLPpy_dataTokeniser
if(useAlgorithmLayeredSANI):
	import SANIHFNLPpy_LayeredSANIGraph
	from SANIHFNLPpy_LayeredSANIGlobalDefs import SANIwordVectors
		
if(useHFconnectionMatrix):
	import torch as pt
	if(useHFconnectionMatrixPyG):
		import HFNLPpy_ConnectionMatrixPyG
	if(useHFconnectionMatrixBasic):
		import HFNLPpy_ConnectionMatrixBasic
		import HFNLPpy_ConceptsMatrix
	if(useHFconnectionMatrixAlgorithm):
		import HFNLPpy_ConnectionMatrixAlgorithm
	import HFNLPpy_ConnectionMatrixOperations

if(useAlgorithmScan):
	#from HFNLPpy_ScanGlobalDefs import *	#not currently required
	import HFNLPpy_Scan
elif(useAlgorithmDendriticSANI):
	#from HFNLPpy_DendriticSANIGlobalDefs import *	#not currently required
	import HFNLPpy_DendriticSANI
	if(useDependencyParseTree):
		import HFNLPpy_DendriticSANISyntacticalGraph
elif(useAlgorithmMatrix):
	from HFNLPpy_MatrixGlobalDefs import *	#currently required
	import HFNLPpy_Matrix
	import HFNLPpy_MatrixOperations
	if(HFconnectionMatrixAlgorithmSplitDatabase):
		import HFNLPpy_MatrixDatabase

if(useDependencyParseTree):
	import SPNLPpy_globalDefs
	import SPNLPpy_syntacticalGraph
	if(not SPNLPpy_globalDefs.useSPNLPcustomSyntacticalParser):
		SPNLPpy_syntacticalGraph.SPNLPpy_syntacticalGraphConstituencyParserFormal.initalise(spacyWordVectorGenerator)
		
if(drawHopfieldGraph):
	if(drawHopfieldGraphSentence):
		import HFNLPpy_hopfieldGraphDraw as hopfieldGraphDrawSentence
	if(drawHopfieldGraphNetwork):
		import HFNLPpy_hopfieldGraphDraw as hopfieldGraphDrawNetwork


class HFconnectionGraphClass:
	def __init__(self):
		if(useAlgorithmMatrix):
			HFNLPpy_Matrix.HFconnectionGraphMatrixHolderInitialisation(self)
		if(useHFconnectionMatrixBasic):
			self.HFconnectionGraphBasic = None
		if(useAlgorithmScan):
			self.HFconnectionGraphPyG = None
		self.connectionMatrixMaxConcepts = None
		self.neuronNamelist = None
		self.neuronIDdict = {}

networkConceptNodeDict = {}
networkSize = 0
HFconnectionGraphObject = HFconnectionGraphClass()


def generateHopfieldGraphNetwork(articles, tokenizer):
	numberOfSentences = len(articles)

	if(useHFconnectionMatrix):
		if(useHFconnectionMatrixBasic):
			HFNLPpy_ConceptsMatrix.initialiseHFconnectionMatrixBasicWrapper(HFconnectionGraphObject)
		if(useHFconnectionMatrixAlgorithm):
			HFNLPpy_Matrix.initialiseHFconnectionMatrixAlgorithmWrapper(HFconnectionGraphObject)
		if(HFconnectionMatrixAlgorithmSplitDatabase):
			HFNLPpy_MatrixDatabase.initialiseMatrixDatabase(HFconnectionGraphObject)
		regenerateGraphNodes()

	if(seedHFnetworkSubsequenceType=="lastSentence"):
		verifySeedSentenceIsReplicant(articles, numberOfSentences)

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
		print("feedPredictionSuccesses = ", HFNLPpy_Matrix.feedPredictionSuccesses)
		print("feedPredictionErrors = ", HFNLPpy_Matrix.feedPredictionErrors)
		print("feedPredictionSuccessRate = ", HFNLPpy_Matrix.feedPredictionSuccesses/(HFNLPpy_Matrix.feedPredictionSuccesses+HFNLPpy_Matrix.feedPredictionErrors))
	
	if(useHFconnectionMatrix):
		if(useHFconnectionMatrixBasic):
			HFNLPpy_ConnectionMatrixBasic.writeHFconnectionMatrixBasicWrapper(HFconnectionGraphObject)
		if(useHFconnectionMatrixAlgorithm):
			HFNLPpy_ConnectionMatrixAlgorithm.writeHFconnectionMatrixAlgorithmWrapper(HFconnectionGraphObject)
		if(HFconnectionMatrixAlgorithmSplitDatabase):
			HFNLPpy_MatrixDatabase.finaliseMatrixDatabase(HFconnectionGraphObject)

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
		if(linkSimilarConceptNodesWordnet):
			getTokenSynonyms(conceptNode, token)
		'''
		if(useAlgorithmLayeredSANI):
			print("not supported")
			conceptNode.SANIlayerNeuronID = 
			conceptNode.SANIlayerIndex = 0
		'''
		addNodeToGraph(conceptNode)
		if(printVerbose):
			print("create new conceptNode; ", conceptNode.nodeName)

def generateHopfieldGraphSentenceNodes(tokenisedSentence, sentenceIndex, sentenceConceptNodeList):
	#declare Hopfield graph nodes;	
	for w, token in enumerate(tokenisedSentence):	
		if(tokeniseSubwords):
			nodeName = token
		else:
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
			wordVector = None
			if(not tokeniseSubwords):
				wordVector = getTokenWordVector(token)	#numpy word vector
				#posTag = getTokenPOStag(token)	#not used
			
			conceptNode = HopfieldNode(networkIndex, nodeName, nodeGraphType, wordVector, w, sentenceIndex)
			if(not tokeniseSubwords):
				getTokenSynonyms(conceptNode, token)

			addNodeToGraph(conceptNode)
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.nodeName)
		sentenceConceptNodeList.append(conceptNode)
		
def generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences, trainSentence):	
	activationTime = calculateActivationTime(sentenceIndex)
			
	sentenceConceptNodeList = []
	sentenceLength = len(tokenisedSentence)
		
	SPgraphHeadNode = None
	if(useDependencyParseTree):
		performIntermediarySyntacticalTransformation = False
		generateSyntacticalGraphNetwork = False
		sentenceLeafNodeList, _, SPgraphHeadNode = SPNLPpy_syntacticalGraph.generateSyntacticalGraphSentence(sentenceIndex, tokenisedSentence, performIntermediarySyntacticalTransformation, generateSyntacticalGraphNetwork, identifySyntacticalDependencyRelations)

	#declare Hopfield graph nodes;	
	generateHopfieldGraphSentenceNodes(tokenisedSentence, sentenceIndex, sentenceConceptNodeList)

	neuronIDdictNewlyAdded = {}
	addSentenceConceptNodesToHFconnectionGraphObject(sentenceConceptNodeList, neuronIDdictNewlyAdded)
	if(HFconnectionMatrixAlgorithmSplitDatabase and useAlgorithmLayeredSANI):
		sentenceConceptNodeListOrig = sentenceConceptNodeList.copy()
			
	if(trainSentence):
		if(useHFconnectionMatrixBasic):
			HFNLPpy_ConceptsMatrix.addContextWordsToConnectionGraphLinkConcepts(tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject)
		
		#create Hopfield graph direct connections (for draw only)
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
					if(useAlgorithmDendriticPrototype):
						for w2 in range(w-1):
							previousContextConceptNodesList.append(sentenceConceptNodeList[w2]) 
					createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)
		
		if(useAlgorithmLayeredSANI):
			#print("useAlgorithmLayeredSANI:... addSentenceConceptNodesToHFconnectionGraphObject")
			#SANIHFNLPpy_LayeredSANIGraph creates Hopfield graph direct connections
			sentenceConceptNodeList = SANIHFNLPpy_LayeredSANIGraph.generateLayeredSANIGraphSentence(HFconnectionGraphObject, sentenceIndex, tokenisedSentence, sentenceConceptNodeList, networkConceptNodeDict)	#sentenceConceptNodeList is replaced with sentenceSANINodeList
			recalculateHopfieldGraphNetworkSize()
			#printConceptNodeList(sentenceConceptNodeList)
			if(not SANIwordVectors):	#already added
				addSentenceConceptNodesToHFconnectionGraphObject(sentenceConceptNodeList, neuronIDdictNewlyAdded)
			
		if(useAlgorithmDendriticSANI):
			#useAlgorithmDendriticSANI:HFNLPpy_DendriticSANIGenerate:addPredictiveSequenceToNeuron:addPredictiveSynapseToNeuron:addConnectionToNode creates connections between hopfield objects (with currentSequentialSegmentInput object)
				if(useDependencyParseTree):
					print("HFNLPpy_DendriticSANISyntacticalGraph.simulateBiologicalHFnetworkSP")
					HFNLPpy_DendriticSANISyntacticalGraph.trainBiologicalHFnetworkSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations)		
				else:
					print("HFNLPpy_DendriticSANI.simulateBiologicalHFnetwork")
					HFNLPpy_DendriticSANI.trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences)	
		elif(useAlgorithmScan):
			for w, token in enumerate(tokenisedSentence):
				conceptNode = sentenceConceptNodeList[w]
				neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
				if(w > 0):
					sourceNeuronID = neuronIDprevious
					targetNeuronID = neuronID
					HFNLPpy_ConnectionMatrixPyG.updateOrAddConnectionToGraph(HFconnectionGraphObject, sourceNeuronID, targetNeuronID)
				neuronIDprevious = neuronID
		elif(useHFconnectionMatrix):
			HFNLPpy_Matrix.addContextWordsToConnectionGraphMatrix(networkConceptNodeDict, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject)
	else:
		#predict Hopfield graph flow;
		seedSentenceConceptNodeList = sentenceConceptNodeList
		if(useAlgorithmScan):
			HFconnectionGraphObject.HFconnectionGraphPyG.activationLevel = pt.zeros(len(HFconnectionGraphObject.neuronNamelist), dtype=pt.float)	# Set the initial activation level for each neuron at time t
			HFconnectionGraphObject.HFconnectionGraphPyG.activationState = pt.zeros(len(HFconnectionGraphObject.neuronNamelist), dtype=pt.bool)	# Set the initial activation state for each neuron at time t
			HFNLPpy_Scan.seedBiologicalHFnetwork(networkConceptNodeDict, networkSize, sentenceIndex, HFconnectionGraphObject, seedSentenceConceptNodeList, numberOfSentences)
		elif(useAlgorithmDendriticSANI):
			HFNLPpy_DendriticSANI.seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences, HFconnectionGraphObject)
		elif(useHFconnectionMatrix):
			HFNLPpy_Matrix.seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences, HFconnectionGraphObject)	
		else:
			printe("HFNLPpy_hopfieldGraph:generateHopfieldGraphSentence error: !trainSentence requires useAlgorithmScan or useAlgorithmDendriticSANI")

	if(HFconnectionMatrixAlgorithmSplitDatabase):
		neuronIDalreadySaved = {}
		HFNLPpy_MatrixDatabase.finaliseMatrixDatabaseSentence(HFconnectionGraphObject, sentenceConceptNodeList, neuronIDalreadySaved)
		if(useAlgorithmLayeredSANI):
			HFNLPpy_MatrixDatabase.finaliseMatrixDatabaseSentence(HFconnectionGraphObject, sentenceConceptNodeListOrig, neuronIDalreadySaved)	#add original sentence nodes that have not been merged into SANI nodes (required for future sentences)
			
	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			hopfieldGraphDrawSentence.drawHopfieldGraphSentenceStatic(sentenceIndex, sentenceConceptNodeList, networkSize, drawHopfieldGraphPlot, drawHopfieldGraphSave)
		if(drawHopfieldGraphNetwork):
			hopfieldGraphDrawNetwork.drawHopfieldGraphNetworkStatic(sentenceIndex, networkConceptNodeDict, drawHopfieldGraphPlot, drawHopfieldGraphSave)

	result = True
	return result

#if(useDependencyParseTree):

def addSentenceConceptNodesToHFconnectionGraphObject(sentenceConceptNodeList, neuronIDdictNewlyAdded):
	if(useHFconnectionMatrix):
		neuronIDdictPrevious = HFconnectionGraphObject.neuronIDdict.copy()
		HFNLPpy_ConnectionMatrixOperations.addSentenceConceptNodesToHFconnectionGraphObject(HFconnectionGraphObject, sentenceConceptNodeList)
		if(HFconnectionMatrixAlgorithmSplit):
			HFNLPpy_Matrix.loadSentenceMatrixAlgorithmSplitMatrices(HFconnectionGraphObject, sentenceConceptNodeList, neuronIDdictPrevious, neuronIDdictNewlyAdded)
							
def connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, DPgovernorNode, spatioTemporalIndex, activationTime):
	for DPdependentNode in DPgovernorNode.DPdependentList:
		previousContextConceptNodesList = []
		conceptNode, previousConceptNode = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList)
		createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)
		connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, DPdependentNode, spatioTemporalIndex, activationTime)

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList):
	conceptNode = sentenceConceptNodeList[DPgovernorNode.w]
	previousConceptNode = sentenceConceptNodeList[DPdependentNode.w]
	if(useAlgorithmDendriticPrototype):
		for DPdependentNode2 in DPdependentNode.DPdependentList:
			previousContextConceptNode = sentenceConceptNodeList[DPdependentNode2.w]
			previousContextConceptNodesList.append(previousContextConceptNode)
			_, _ = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalPrototype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode2, previousContextConceptNodesList)
	return conceptNode, previousConceptNode

def createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime):
	HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, contextConnection=False)
	#HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, activationTime, spatioTemporalIndex)
	if(useAlgorithmDendriticPrototype):
		totalConceptsInSubsequence = 0
		for previousContextIndex, previousContextConceptNode in enumerate(previousContextConceptNodesList):
			totalConceptsInSubsequence += 1
			#multiple connections/synapses are made between current neuron and ealier neurons in sequence, and synapse weights are adjusted such that the particular combination (or permutation if SANI synapses) will fire the neuron
			weight = 1.0/totalConceptsInSubsequence	#for useAlgorithmDendriticPrototype: interpret connection as unique synapse
			#print("weight = ", weight)
			HFNLPpy_hopfieldOperations.addConnectionToNode(previousContextConceptNode, conceptNode, activationTime, spatioTemporalIndex, useAlgorithmDendriticPrototype=useAlgorithmDendriticPrototype, weight=weight, contextConnection=True, contextConnectionSANIindex=previousContextIndex)					

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
	if(tokeniseSubwords):
		sample = HFNLPpy_dataTokeniser.tokenise(sentence, tokenizer, sequenceMaxNumTokens)
		inputIdList = sample.input_ids[0].tolist()	#[0]: select first sentence (only 1 sentence available)
		tokenList = tokenizer.convert_ids_to_tokens(inputIdList)
		for specialToken in specialTokens:
			while specialToken in tokenList:
				tokenList.remove(specialToken)
	else:
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

def getTokenSynonyms(conceptNode, token=None):
	if(tokenWordnetSynonyms):
		if(tokenWordnetSynonymsFromLemma):
			token = spacyWordVectorGenerator(conceptNode.nodeName)
		conceptNode.synonymsList = HFNLPpy_hopfieldOperations.getTokenSynonyms(token)

#creation/access time:

def getNetworkIndex():
	networkIndex = len(networkConceptNodeDict)
	return networkIndex
		

#subsequence seed;
		
def verifySeedSentenceIsReplicant(articles, numberOfSentences):
	if(HFNLPnonrandomSeed):
		np.random.seed(0)
		print("np.random.randint(0,9) = ", np.random.randint(0,9))
		#random.seed(0)	#not used
		#print("random.randint(0,9) = ", random.randint(0,9))

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


#networkSize;

def recalculateHopfieldGraphNetworkSize():
	global networkSize
	networkSize = len(networkConceptNodeDict)

def printConceptNodeList(sentenceConceptNodeList):
	for conceptNode in sentenceConceptNodeList:
		print("conceptNode = ", conceptNode.nodeName)
		
