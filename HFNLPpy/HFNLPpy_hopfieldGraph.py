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
- biologicalPrototype: add contextual connections to emulate spatiotemporal index restriction (visualise theoretical biological connections without simulation)
- biologicalSimulation: simulate sequential activation of concept neurons and their dendritic input/synapses

"""

import numpy as np
import spacy
spacyWordVectorGenerator = spacy.load('en_core_web_md')	#spacy.load('en_core_web_lg')
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations

printVerbose = False

biologicalPrototype = False	#add contextual connections to emulate primary connection spatiotemporal index restriction (visualise biological connections without simulation)
biologicalSimulation = True	#simulate sequential activation of dendritic input 
useDependencyParseTree = False

def generateHopfieldGraphFileName(sentenceOrNetwork, sentenceIndex=None):
	fileName = "hopfieldGraph"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + str(sentenceIndex)
	return fileName
	
if(biologicalSimulation):
	import HFNLPpy_biologicalSimulation
	biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False
	if(useDependencyParseTree):
		biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False	#optional	#speculative: directly encode precalculated syntactical structure in dendritic branches (rather than deriving syntax from commonly used dendritic subsequence encodings)
	
if(useDependencyParseTree):
	import SPNLPpy_syntacticalGraph
	if(not SPNLPpy_syntacticalGraph.useSPNLPcustomSyntacticalParser):
		SPNLPpy_syntacticalGraph.SPNLPpy_syntacticalGraphConstituencyParserFormal.initalise(spacyWordVectorGenerator)
	if(biologicalSimulation):
		identifySyntacticalDependencyRelations = True	#optional; only implementation coded (if use constituency parser, synapses are created in most distal branch segments only)
	else:
		identifySyntacticalDependencyRelations = True	#mandatory 

drawHopfieldGraph = True
if(drawHopfieldGraph):
	drawHopfieldGraphPlot = True
	drawHopfieldGraphSave = False
	drawHopfieldGraphSentence = True
	if(drawHopfieldGraphSentence):
		import HFNLPpy_hopfieldGraphDraw as ATNLPtf_hopfieldGraphDrawSentence
	drawHopfieldGraphNetwork = True	#default: True	#draw graph for entire network (not just sentence)
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
		
	activationTime = calculateActivationTime(sentenceIndex)

	if(drawHopfieldGraph):
		if(drawHopfieldGraphSentence):
			ATNLPtf_hopfieldGraphDrawSentence.clearHopfieldGraph()
		if(drawHopfieldGraphNetwork):
			ATNLPtf_hopfieldGraphDrawNetwork.clearHopfieldGraph()
			
	sentenceConceptNodeList = []
	sentenceLength = len(tokenisedSentence)
		
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
		else:
			#primary vars;
			wordVector = getTokenWordVector(token)	#numpy word vector
			#posTag = getTokenPOStag(token)	#not used
			activationTime = calculateActivationTime(sentenceIndex)
			nodeGraphType = graphNodeTypeConcept
			networkIndex = getNetworkIndex()
			conceptNode = HopfieldNode(networkIndex, nodeName, wordVector, nodeGraphType, activationTime, biologicalSimulation, w, sentenceIndex)
			addNodeToGraph(conceptNode)
			if(printVerbose):
				print("create new conceptNode; ", conceptNode.lemma)
		sentenceConceptNodeList.append(conceptNode)
			
	if(biologicalSimulation):
		if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
			if(identifySyntacticalDependencyRelations):
				simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode)		
			else:
				print("biologicalSimulation:identifySyntacticalDependencyRelations current implementation requires identifySyntacticalDependencyRelations")
				exit()
				#simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode)					
		else:
			HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
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
					if(biologicalPrototype):
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

#if(biologicalSimulation):
#def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPtargetNode):
#	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
#		HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceSyntacticalBranchCPTrain(networkConceptNodeDict, sentenceIndex, identifySyntacticalDependencyRelations, CPtargetNode)
#	else:
#		print("simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP error: requires biologicalSimulationEncodeSyntaxInDendriticBranchStructure")
#		exit()
#	for CPsourceNode in SPtargetNode.CPgraphNodeSourceList:
#		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPsourceNode)
def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPgovernorNode):
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain(networkConceptNodeDict, sentenceIndex, identifySyntacticalDependencyRelations, DPgovernorNode)
	else:
		contextConceptNodesList = []
		identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPgovernorNode, contextConceptNodesList)
		w = len(contextConceptNodesList) - 1	#index of DPgovernorNode in contextConceptNodesList
		HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceNodeTrain(sentenceIndex, contextConceptNodesList, w, DPgovernorNode)
	for DPdependentNode in DPgovernorNode.DPdependentList:
		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPdependentNode)

		
def connectHopfieldGraphSentenceSyntacticalBranchDP(sentenceConceptNodeList, DPgovernorNode, spatioTemporalIndex, activationTime):
	for DPdependentNode in DPgovernorNode.DPdependentList:
		previousContextConceptNodesList = []
		conceptNode, previousConceptNode = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalProtoype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList)
		createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime)
		connectHopfieldGraphSentence(sentenceConceptNodeList, DPdependentNode, spatioTemporalIndex, activationTime)

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalProtoype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode, previousContextConceptNodesList):
	conceptNode = sentenceConceptNodeList[DPgovernorNode.w]
	previousConceptNode = sentenceConceptNodeList[DPdependentNode.w]
	if(biologicalPrototype):
		for DPdependentNode2 in DPdependentNode.DPdependentList:
			previousContextConceptNode = sentenceConceptNodeList[DPdependentNode2.w]
			previousContextConceptNodesList.append(previousContextConceptNode)
			_, _ = identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalProtoype(sentenceConceptNodeList, DPgovernorNode, DPdependentNode2, previousContextConceptNodesList)
	return conceptNode, previousConceptNode

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPgovernorNode, contextConceptNodesList):
	contextConceptNodesList.append(DPgovernorNode)
	for DPdependentNode in DPgovernorNode.DPdependentList:
		identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPdependentNode, contextConceptNodesList)
	

def createConnection(conceptNode, previousConceptNode, previousContextConceptNodesList, spatioTemporalIndex, activationTime):
	HFNLPpy_hopfieldOperations.addConnectionToNode(previousConceptNode, conceptNode, activationTime, spatioTemporalIndex)
	
	if(biologicalPrototype):
		totalConceptsInSubsequence = 0
		for previousContextIndex, previousContextConceptNode in enumerate(previousContextConceptNodesList):
			totalConceptsInSubsequence += 1
			#multiple connections/synapses are made between current neuron and ealier neurons in sequence, and synapse weights are adjusted such that the particular combination (or permutation if SANI synapses) will fire the neuron
			weight = 1.0/totalConceptsInSubsequence	#for biologicalPrototype: interpret connection as unique synapse
			#print("weight = ", weight)
			HFNLPpy_hopfieldOperations.addConnectionToNode(previousContextConceptNode, conceptNode, activationTime, spatioTemporalIndex, biologicalPrototype=biologicalPrototype, weight=weight, contextConnection=True, contextConnectionSANIindex=previousContextIndex)					


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
		


