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

if(useHFconnectionMatrix):
	import torch as pt
	if(useHFconnectionMatrixPyG):
		import HFNLPpy_ConnectionMatrixPyG as HFNLPpy_ConnectionMatrix
	if(useHFconnectionMatrixBasic):
		import HFNLPpy_ConnectionMatrixBasic as HFNLPpy_ConnectionMatrix
		
if(useAlgorithmScan):
	from HFNLPpy_ScanGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_ScanGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_ScanGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_Scan
elif(useAlgorithmDendriticSANI):
	from HFNLPpy_DendriticSANIGlobalDefs import seedHFnetworkSubsequence
	if(seedHFnetworkSubsequence):
		from HFNLPpy_DendriticSANIGlobalDefs import seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant
	from HFNLPpy_DendriticSANIGlobalDefs import HFNLPnonrandomSeed
	import HFNLPpy_DendriticSANI
	if(useDependencyParseTree):
		import HFNLPpy_DendriticSANISyntacticalGraph
elif(useAlgorithmMatrix):
	from HFNLPpy_MatrixGlobalDefs import *
	import HFNLPpy_Matrix
	
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



networkConceptNodeDict = {}
networkSize = 0
if(useHFconnectionMatrix):
	class HFconnectionGraphClass:
		def __init__(self):
			if(useAlgorithmMatrix):
				if(algorithmMatrixSingleTensor):
					self.HFconnectionGraphMatrix = None
					self.HFconnectionGraphMatrixNormalised = None
				else:
					self.HFconnectionGraphMatrix = [[None for _ in range(contextSizeMax)] for _ in range(numberOfDendriticBranches)]	#[[None]*contextSizeMax]*numberOfDendriticBranches	#[[[None]*contextSizeMax] for i in range(numberOfDendriticBranches)]
					self.HFconnectionGraphMatrixNormalised = [[None for _ in range(contextSizeMax)] for _ in range(numberOfDendriticBranches)]	#[[None]*contextSizeMax]*numberOfDendriticBranches	#[[[None]*contextSizeMax] for i in range(numberOfDendriticBranches)]
			if(linkSimilarConceptNodesBagOfWords):
				self.HFconnectionGraphBasic = None
			if(useAlgorithmScan):
				self.HFconnectionGraphPyG = None
			self.neuronNamelist = None
			self.neuronIDdict = {}
	HFconnectionGraphObject = HFconnectionGraphClass()
else:
	HFconnectionGraphObject = None
	
def readHFconnectionMatrix():
	if(useAlgorithmMatrix):
		if(algorithmMatrixSingleTensor):
			HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.HFconnectionGraphMatrix = HFNLPpy_ConnectionMatrix.readHFconnectionMatrix()
			HFconnectionGraphObject.HFconnectionGraphMatrixNormalised = normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix)
		else:
			for dendriticBranchIndex in range(numberOfDendriticBranches):
				for contextSizeIndex in range(contextSizeMax):
					HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][contextSizeIndex] = HFNLPpy_ConnectionMatrix.readHFconnectionMatrix(createIndexStringDendriticBranch(dendriticBranchIndex), createIndexStringContextSizeIndex(contextSizeIndex))
					HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][contextSizeIndex] = normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][contextSizeIndex])
	else:
		HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.HFconnectionGraphBasic = HFNLPpy_ConnectionMatrix.readHFconnectionMatrix()
	regenerateGraphNodes()
	
def writeHFconnectionMatrix():
	if(useAlgorithmMatrix):
		if(algorithmMatrixSingleTensor):
			HFNLPpy_ConnectionMatrix.writeHFconnectionMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix, HFconnectionGraphObject.neuronNamelist)
		else:
			for dendriticBranchIndex in range(numberOfDendriticBranches):
				for contextSizeIndex in range(contextSizeMax):
					HFNLPpy_ConnectionMatrix.writeHFconnectionMatrix(HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchIndex][contextSizeIndex], HFconnectionGraphObject.neuronNamelist, createIndexStringDendriticBranch(dendriticBranchIndex), createIndexStringContextSizeIndex(contextSizeIndex))
	else:
		HFNLPpy_ConnectionMatrix.writeHFconnectionMatrix(HFconnectionGraphObject.HFconnectionGraphBasic, HFconnectionGraphObject.neuronNamelist)

def createIndexStringDendriticBranch(dendriticBranchIndex):
	return "dendriticBranchIndex" + str(dendriticBranchIndex)
def createIndexStringContextSizeIndex(contextSizeIndex):
	return "contextSizeIndex" + str(contextSizeIndex)	
			
def generateHopfieldGraphNetwork(articles):
	numberOfSentences = len(articles)

	if(useHFconnectionMatrix):
		readHFconnectionMatrix()

	if(seedHFnetworkSubsequence):
		verifySeedSentenceIsReplicant(articles, numberOfSentences)

	for sentenceIndex, sentence in enumerate(articles):
		generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences)	
		
	if(useHFconnectionMatrix):
		writeHFconnectionMatrix()

def generateHopfieldGraphSentenceString(sentenceIndex, sentence, numberOfSentences):
	print("\n\ngenerateHopfieldGraphSentenceString: sentenceIndex = ", sentenceIndex, "; ", sentence)

	tokenisedSentence = tokeniseSentence(sentence)
	sentenceLength = len(tokenisedSentence)
	#print("sentenceLength = ", sentenceLength)
	
	if(sentenceLength > 1):
		return generateHopfieldGraphSentence(sentenceIndex, tokenisedSentence, numberOfSentences)

def regenerateGraphNodes():
	#regenerates graph nodes from a saved list
	sentence = ' '.join(HFconnectionGraphObject.neuronNamelist)
	tokenisedSentence = tokeniseSentence(sentence)
	for neuronID, nodeName in enumerate(HFconnectionGraphObject.neuronNamelist):	
		token = tokenisedSentence[neuronID]
		networkIndex = getNetworkIndex()
		nodeGraphType = graphNodeTypeConcept
		wordVector = None	#getTokenWordVector(token)	#numpy word vector	#not used by useHFconnectionMatrix
		#posTag = getTokenPOStag(token)	#not used
		w = 0	#sentence artificial var (not used)
		sentenceIndex = 0	#sentence artificial var (not used)
		
		conceptNode = HopfieldNode(networkIndex, nodeName, nodeGraphType, wordVector, w, sentenceIndex)
		getTokenSynonyms(conceptNode, token)
		'''
		if(useAlgorithmLayeredSANI):
			print("not supported")
			conceptNode.SANIlayerNeuronID = 
			conceptNode.SANIlayerIndex = 0
		'''
		if(useHFconnectionMatrix):
			HFconnectionGraphObject.neuronIDdict[nodeName] = neuronID
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
			getTokenSynonyms(conceptNode, token)

			addNodeToGraph(conceptNode)
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.nodeName)
		sentenceConceptNodeList.append(conceptNode)
		
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
	generateHopfieldGraphSentenceNodes(tokenisedSentence, sentenceIndex, sentenceConceptNodeList)

	if(useHFconnectionMatrix):
		for conceptNode in sentenceConceptNodeList:	
			HFconnectionGraphObject.neuronNamelist.append(conceptNode.nodeName)
			neuronID = conceptNode.networkIndex
			HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName] = neuronID
		
	trainSentence = True
	if(sentenceIndex == numberOfSentences-1):
		if(seedHFnetworkSubsequence):
			trainSentence = False
			
	if(trainSentence):
		if(linkSimilarConceptNodesBagOfWords):
			addContextWordsToConnectionGraphLinkConcepts(tokenisedSentence, sentenceConceptNodeList)

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
					HFNLPpy_ConnectionMatrix.updateOrAddConnectionToGraph(HFconnectionGraphObject, sourceNeuronID, targetNeuronID)
				neuronIDprevious = neuronID
		elif(useHFconnectionMatrix):
			addContextWordsToConnectionGraphMatrix(tokenisedSentence, sentenceConceptNodeList)
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
			
	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			hopfieldGraphDrawSentence.drawHopfieldGraphSentenceStatic(sentenceIndex, sentenceConceptNodeList, networkSize, drawHopfieldGraphPlot, drawHopfieldGraphSave)
		if(drawHopfieldGraphNetwork):
			hopfieldGraphDrawNetwork.drawHopfieldGraphNetworkStatic(sentenceIndex, networkConceptNodeDict, drawHopfieldGraphPlot, drawHopfieldGraphSave)

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


#connection graph;

def addContextWordsToConnectionGraphLinkConcepts(tokenisedSentence, sentenceConceptNodeList):
	for w1, token1 in enumerate(tokenisedSentence):
		conceptNode = sentenceConceptNodeList[w1]
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		HFconnectionGraphObject.HFconnectionGraphBasic, HFconnectionGraphObject.HFconnectionGraphBasicNormalised = addContextWordsToConnectionGraph(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphBasic, linkSimilarConceptNodesBagOfWordsDistanceMax, linkSimilarConceptNodesBagOfWordsWeightStore, linkSimilarConceptNodesBagOfWordsBidirectional)

def addContextWordsToConnectionGraphMatrix(tokenisedSentence, sentenceConceptNodeList):
	for w1, token1 in enumerate(tokenisedSentence):
		#print("w1 = ", w1)
		conceptNode = sentenceConceptNodeList[w1]
		#print("addContextWordsToConnectionGraphMatrix: conceptNode.nodeName = ", conceptNode.nodeName) 
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		contextSizeMax2 = min(contextSizeMax, w1)	#min(contextSizeMax, len(tokenisedSentence))
		if(algorithmMatrixSingleTensor):
			contextSizeIndex = contextSizeMax
			_, dendriticBranchClosestIndex = HFNLPpy_hopfieldOperations.connectionMatrixCalculateConnectionStrengthIndex(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised, contextSizeIndex, contextMatrixWeightStore, False, contextSizeMax2)
		else:
			dendriticBranchClosestIndex = -1
			dendriticBranchClosestValue = 0
			for dendriticBranchIndex in range(numberOfDendriticBranches):
				#print("dendriticBranchIndex = ", dendriticBranchIndex)
				for contextSizeIndex in range(contextSizeMax2):
					#print("contextSizeIndex = ", contextSizeIndex)
					connectionStrength, _ = HFNLPpy_hopfieldOperations.connectionMatrixCalculateConnectionStrengthIndex(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][contextSizeIndex], contextSizeIndex, contextMatrixWeightStore, False)
					if(connectionStrength > dendriticBranchClosestValue):
						dendriticBranchClosestValue = connectionStrength
						dendriticBranchClosestIndex = dendriticBranchIndex
		if(debugAlgorithmMatrix):
			print("dendriticBranchClosestIndex = ", dendriticBranchClosestIndex)
		for contextSizeIndex in range(contextSizeMax2):
			#print("contextSizeIndex = ", contextSizeIndex)
			HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchClosestIndex][contextSizeIndex], HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchClosestIndex][contextSizeIndex] = addContextWordsToConnectionGraph(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphMatrix[dendriticBranchClosestIndex][contextSizeIndex], contextSizeIndex, contextMatrixWeightStore, False)
	
def addContextWordsToConnectionGraph(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraph, contextSizeIndex, weightStore, bidirectionalContext):
	contextConnectionVector = HFNLPpy_hopfieldOperations.createContextVector(w1, sentenceConceptNodeList, HFconnectionGraphObject, len(HFconnectionGraphObject.neuronNamelist), contextSizeIndex, weightStore, bidirectionalContext)
	HFNLPpy_ConnectionMatrix.addContextConnectionsToGraph(HFconnectionGraph, neuronID, contextConnectionVector)
	HFconnectionGraphNormalised = normaliseBatchedTensor(HFconnectionGraph)
	#if(debugAlgorithmMatrix):
	#	print("contextConnectionVector = ", contextConnectionVector)
	#	#print("HFconnectionGraph[neuronID] = ", HFconnectionGraph[neuronID])
	#	print("HFconnectionGraphNormalised[neuronID] = ", HFconnectionGraphNormalised[neuronID])
	return HFconnectionGraph, HFconnectionGraphNormalised

def normaliseBatchedTensor(HFconnectionGraph):
	if(useHFconnectionMatrixBasicBool):	#OLD: if(not weightStore)
		HFconnectionGraphNormalised = HFconnectionGraphFloat
	else:
		HFconnectionGraphFloat = (HFconnectionGraph).float()
		#calculate a temporary normalised version of the HFconnectionGraph	#CHECKTHIS
		if(useHFconnectionMatrixNormaliseSoftmax):
			HFconnectionGraphNormalised = pt.nn.functional.softmax(HFconnectionGraphFloat, dim=1)
		else:
			min_vals, _ = pt.min(HFconnectionGraphFloat, dim=1, keepdim=True)
			max_vals, _ = pt.max(HFconnectionGraphFloat, dim=1, keepdim=True)
			epsilon = 1e-8  # Small epsilon value
			HFconnectionGraphNormalised = (HFconnectionGraphFloat - min_vals) / (max_vals - min_vals + epsilon)
	return HFconnectionGraphNormalised
