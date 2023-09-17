"""HFNLPpy_DendriticSANIDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI Draw - draw sentence/network graph with dendritic trees

definition of colour scheme: HFNLPbiologicalImplementationDevelopment-07June2022b.pdf

"""

import networkx as nx
import matplotlib.pyplot as plt
plt.ioff()	# Turn interactive plotting off
from math import cos, sin, radians
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_DendriticSANIGlobalDefs import *
from HFNLPpy_DendriticSANINode import *

if(writeBiologicalSimulation):
	import HFNLPpy_DendriticSANIXML

highResolutionFigure = True
if(highResolutionFigure):
	saveFigDPI = 300	#approx HD	#depth per inch
	saveFigSize = (16,9)	#in inches
	
debugOnlyDrawTargetNeuron = False
debugOnlyDrawActiveBranches = False

drawDendriticSANIGraphEdgeColoursWeights = True	#mandatory
drawDendriticSANIGraphNodeColours = True	#node colours not yet coded (pos type of concept node will be different depending on connectivity/instance context)
graphTransparency = 0.5

dendriticSANIGraph = nx.Graph()	#MultiDiGraph: Directed graphs with self loops and parallel edges	#https://networkx.org/documentation/stable/reference/classes/multidigraph.html
dendriticSANIGraphNodeColorMap = []
dendriticSANIGraphNodeSizeMap = []
dendriticSANIGraphConceptNodesList = []	#primary nodes for label assignment

#require calibration (depends on numberOfBranches1/numberOfBranches2/numberOfBranchSequentialSegments):
conceptNeuronIndexSeparation = 10.0*numberOfBranches2
branchIndex1Separation = 10.0/numberOfBranches1	#vertical separation
horizontalBranchSeparationDivergence = 2
branchIndex2Separation = conceptNeuronIndexSeparation/numberOfBranches2	#horizontal separation at branchIndex=1 (will decrease at higher vertical separation)
if(supportForNonBinarySubbranchSize):
	if(debugOnlyDrawTargetNeuron):
		horizontalBranchSeparationDivergence = 10
	#branchIndex2Separation = conceptNeuronIndexSeparation/numberOfBranches2/1.5

sequentialSegmentIndexSeparation = branchIndex1Separation/numberOfBranchSequentialSegments	#10.0/numberOfBranchSequentialSegments/2.0
sequentialSegmentInputIndexSeparation = 0.5

spineSeparation = 0.2
nodeSize = 0.5	#node diameter
nodeSizeDraw = 10.0	#node diameter
conceptNodeSizeDrawNetwork = 100.0	#node diameter
conceptNodeSizeDrawSentence = 1000.0	#node diameter

drawDendriticBranchOrthogonal = True


def drawDendriticSANIStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences=0):
	if(drawBiologicalSimulation):
		if(drawBiologicalSimulationSentence):
			fileName = generateDendriticSANIFileName(True, sentenceIndex, write=False)
			clearDendriticSANIGraph()
			drawDendriticSANIGraphSentence(sentenceConceptNodeList)
			print("drawBiologicalSimulationSentence: HFNLPpy_DendriticSANIDraw.displayDendriticSANIGraph()")
			displayDendriticSANIGraph(drawBiologicalSimulationPlot, drawBiologicalSimulationSave, fileName)
		if(drawBiologicalSimulationNetwork):
			fileName = generateDendriticSANIFileName(False, sentenceIndex, write=False)
			clearDendriticSANIGraph()
			drawDendriticSANIGraphNetwork(networkConceptNodeDict)
			print("drawBiologicalSimulationNetwork: HFNLPpy_DendriticSANIDraw.displayDendriticSANIGraph()")
			displayDendriticSANIGraph(drawBiologicalSimulationPlot, drawBiologicalSimulationSave, fileName)	
	if(writeBiologicalSimulation):
		if(writeBiologicalSimulationSentence):	
			fileName = generateDendriticSANIFileName(True, sentenceIndex, write=True)
			HFNLPpy_DendriticSANIXML.writeDentriticSANIGraphSentence(sentenceConceptNodeList, fileName)
		if(writeBiologicalSimulationNetwork):
			if(not outputBiologicalSimulationNetworkLastSentenceOnly or (sentenceIndex == numberOfSentences-1)):
				fileName = generateDendriticSANIFileName(False, sentenceIndex, write=True)
				HFNLPpy_DendriticSANIXML.writeDentriticSANIGraphNetwork(networkConceptNodeDict, fileName)
	
def drawBiologicalDendriticSANISequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex, activationTime, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivation or (sentenceIndex == sentenceIndexDebug and wTarget == wSource+1)):	#default
		#if(not debugCalculateNeuronActivation or (sentenceIndex == sentenceIndexDebug and wSourceDebug == wSource)):	#useful with debugOnlyDrawTargetNeuron
			if(vectoriseComputation or emulateVectorisedComputationOrder):
				print("branchIndex1 = ", branchIndex1, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
			if(drawBiologicalSimulationSentenceDynamic):
				fileName = generateDendriticSANIDynamicSequentialSegmentFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				clearDendriticSANIGraph()
				drawDendriticSANIGraphSentence(sentenceConceptNodeList, activationTime=activationTime, wTarget=wTargetDebug)
				displayDendriticSANIGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationNetworkDynamic):
				fileName = generateDendriticSANIDynamicSequentialSegmentFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				clearDendriticSANIGraph()
				drawDendriticSANIGraphNetwork(networkConceptNodeDict, activationTime=activationTime, wTarget=wTargetDebug)
				displayDendriticSANIGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)				
	if(writeBiologicalSimulationDynamic):
		if(writeBiologicalSimulationSentenceDynamic):	
			fileName = generateDendriticSANIDynamicSequentialSegmentFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
			HFNLPpy_DendriticSANIXML.writeDentriticSANIGraphSentence(sentenceConceptNodeList, fileName, activationTime=activationTime)
		if(writeBiologicalSimulationNetworkDynamic):
			#if(sentenceIndex == numberOfSentences-1):
			fileName = generateDendriticSANIDynamicSequentialSegmentFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
			HFNLPpy_DendriticSANIXML.writeDentriticSANIGraphNetwork(networkConceptNodeDict, fileName, activationTime=activationTime)
				
def drawDendriticSANIDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivation or (sentenceIndex == sentenceIndexDebug and wSource >= wSourceDebug)):	#wSource == wSourceDebug
			if(drawBiologicalSimulationSentenceDynamic):
				fileName = generateDendriticSANIDynamicNeuronFileName(True, wSource, sentenceIndex)
				clearDendriticSANIGraph()
				drawDendriticSANIGraphSentence(sentenceConceptNodeList, activationTime=activationTime, wTarget=wTargetDebug)
				displayDendriticSANIGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationNetworkDynamic):
				fileName = generateDendriticSANIDynamicNeuronFileName(False, wSource, sentenceIndex)
				clearDendriticSANIGraph()
				drawDendriticSANIGraphNetwork(networkConceptNodeDict, activationTime=activationTime, wTarget=wTargetDebug)
				displayDendriticSANIGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)	
	if(writeBiologicalSimulationDynamic):
		if(writeBiologicalSimulationSentenceDynamic):	
			fileName = generateDendriticSANIDynamicNeuronFileName(True, wSource, sentenceIndex)
			HFNLPpy_DendriticSANIXML.writeDentriticSANIGraphSentence(sentenceConceptNodeList, fileName, activationTime=activationTime)
		if(writeBiologicalSimulationNetworkDynamic):
			#if(sentenceIndex == numberOfSentences-1):
			fileName = generateDendriticSANIDynamicNeuronFileName(False, wSource, sentenceIndex)
			HFNLPpy_DendriticSANIXML.writeDentriticSANIGraphNetwork(networkConceptNodeDict, fileName, activationTime=activationTime)
								
def clearDendriticSANIGraph():
	dendriticSANIGraph.clear()	#only draw graph for single sentence
	if(drawDendriticSANIGraphNodeColours):
		dendriticSANIGraphNodeColorMap.clear()
		dendriticSANIGraphNodeSizeMap.clear()
	dendriticSANIGraphConceptNodesList.clear()	#for labels

def drawDendriticSANIGraphSentence(sentenceConceptNodeList, activationTime=None, wTarget=None):	
	sentenceConceptNodeList = list(set(sentenceConceptNodeList))	#generate a unique list from a list (in the event a sentence contains multiple instances of the same word/lemma)
	
	#print("drawDendriticSANIGraphSentence = ")
	#print("size dendriticSANIGraph.nodes = ", len(dendriticSANIGraph.nodes))
	#print("size dendriticSANIGraphNodeColorMap = ", len(dendriticSANIGraphNodeColorMap))
	drawGraphNetwork = False
	#networkSize = len(sentenceConceptNodeList)
	#need to draw all conceptNodes and their dendriticTrees before creating connections
	for conceptNode in sentenceConceptNodeList:
		if(not debugOnlyDrawTargetNeuron or (conceptNode.w==wTarget)):
			drawDendriticSANIGraphNode(conceptNode, drawGraphNetwork, activationTime)
	for conceptNode in sentenceConceptNodeList:
		if(not debugOnlyDrawTargetNeuron):
			drawDendriticSANIGraphNodeConnections(conceptNode, drawGraphNetwork, activationTime, sentenceConceptNodeList)

def drawDendriticSANIGraphNetwork(networkConceptNodeDict, activationTime=None, wTarget=None):	
	#print("drawDendriticSANIGraphNetwork = ")
	#print("size dendriticSANIGraph.nodes = ", len(dendriticSANIGraph.nodes))
	#print("size dendriticSANIGraphNodeColorMap = ", len(dendriticSANIGraphNodeColorMap))
	#generate nodes and connections
	drawGraphNetwork = True
	#networkSize = len(networkConceptNodeDict)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		if(not debugOnlyDrawTargetNeuron or (conceptNode.w==wTarget)):
			drawDendriticSANIGraphNode(conceptNode, drawGraphNetwork, activationTime)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		if(not debugOnlyDrawTargetNeuron):
			drawDendriticSANIGraphNodeConnections(conceptNode, drawGraphNetwork, activationTime)

def drawDendriticSANIGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, activationTime, sentenceConceptNodeList=None):
	for connectionKey, connectionList in hopfieldGraphNode.HFcontextTargetConnectionMultiDict.items():
		for connection in connectionList:
			drawDendriticSANIGraphConnection(connection, drawGraphNetwork, activationTime, sentenceConceptNodeList)
			
#def drawDendriticSANIGraphNodeAndConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList=None):	
#	#parse tree and generate nodes and connections
#	drawDendriticSANIGraphNode(hopfieldGraphNode, drawGraphNetwork)
#	drawDendriticSANIGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList)
		
def drawDendriticSANIGraphNode(conceptNode, drawGraphNetwork, activationTime):
	if(conceptNode.activationLevel):
		colorHtml = 'turquoise'	#soma: turquoise	
	else:
		colorHtml = 'darkgreen'	#soma: 	darkgreen	(orig: turquoise)
	#print("conceptNode.networkIndex = ", conceptNode.networkIndex)
	
	if(debugOnlyDrawTargetNeuron):
		posX = len(dendriticSANIGraphConceptNodesList)*conceptNeuronIndexSeparation
	else:
		if(drawGraphNetwork):
			posX = conceptNode.networkIndex*conceptNeuronIndexSeparation
		else:
			posX = conceptNode.w*conceptNeuronIndexSeparation
	posY = 0	#y=0: currently align concept neurons along single plane
	
	#print("drawDendriticSANIGraphNode: ", conceptNode.nodeName)
	dendriticSANIGraph.add_node(conceptNode.nodeName, pos=(posX, posY))
	if(drawDendriticSANIGraphNodeColours):
		dendriticSANIGraphNodeColorMap.append(colorHtml)
		if(drawGraphNetwork):
			dendriticSANIGraphNodeSizeMap.append(conceptNodeSizeDrawNetwork)
		else:
			dendriticSANIGraphNodeSizeMap.append(conceptNodeSizeDrawSentence)
	dendriticSANIGraphConceptNodesList.append(conceptNode.nodeName)

	#if(useAlgorithmDendriticSANIbiologicalSimulation) exclusive code:
	posYdendriticTreeBranchHead = posY+branchIndex1Separation	#position of first branching within dendritic tree
	currentBranchIndex1 = 0
	drawDendriticSANIGraphNodeDendriticBranch(conceptNode, posX, posYdendriticTreeBranchHead, conceptNode.dendriticTree, currentBranchIndex1, conceptNode, posX, posY, activationTime, drawOrthogonalBranchNode=False)

#if(useAlgorithmDendriticSANIbiologicalSimulation) exclusive code:
	
def drawDendriticSANIGraphNodeDendriticBranch(conceptNode, posX, posY, dendriticBranch, currentBranchIndex1, previousBranch, previousConceptNodePosX, previousConceptNodePosY, activationTime, drawOrthogonalBranchNode=True):
	#print("drawDendriticSANIGraphNodeDendriticBranch: , dendriticBranch.nodeName = ", dendriticBranch.nodeName, ", currentBranchIndex1 = ", currentBranchIndex1, ", posX = ", posX, ", posY = ", posY)
	
	colorHtml = 'green' #branch: green	#'OR #ffffff' invisible: white
	dendriticSANIGraph.add_node(dendriticBranch.nodeName, pos=(posX, posY))
	if(drawDendriticSANIGraphNodeColours):
		dendriticSANIGraphNodeColorMap.append(colorHtml)
		dendriticSANIGraphNodeSizeMap.append(nodeSizeDraw)
	if(drawOrthogonalBranchNode and drawDendriticBranchOrthogonal):
		colorHtml = 'white'
		orthogonalNodeName = dendriticBranch.nodeName + "Orthogonal"
		dendriticSANIGraph.add_node(orthogonalNodeName, pos=(posX, previousConceptNodePosY))	#draw another node directly below the branch head node (this should be invisible)
		if(drawDendriticSANIGraphNodeColours):
			dendriticSANIGraphNodeColorMap.append(colorHtml)
			dendriticSANIGraphNodeSizeMap.append(nodeSizeDraw)
	else:
		orthogonalNodeName = None
	if(not debugOnlyDrawActiveBranches or dendriticBranch.activationLevel):
		drawDendriticSANIGraphBranch(currentBranchIndex1, previousBranch, dendriticBranch, drawOrthogonalBranchNode, orthogonalNodeName, activationTime)	#draw branch edge
	
	if(drawOrthogonalBranchNode and drawDendriticBranchOrthogonal):
		previousSequentialSegmentNodeName = orthogonalNodeName					
	else:
		previousSequentialSegmentNodeName = previousBranch.nodeName
	for currentSequentialSegmentIndex, currentSequentialSegment in enumerate(dendriticBranch.sequentialSegments):
		posYsequentialSegment = posY+currentSequentialSegmentIndex*sequentialSegmentIndexSeparation - (sequentialSegmentIndexSeparation*(numberOfBranchSequentialSegments-1))
		previousSequentialSegmentNodeName = drawDendriticSANIGraphNodeSequentialSegment(currentBranchIndex1, conceptNode, posX, posYsequentialSegment, currentSequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime)
	
	if((currentBranchIndex1 < numberOfBranches1-1) or not expectFirstBranchSequentialSegmentConnectionStrictNumBranches1):	
		for currentBranchIndex2, subbranch in enumerate(dendriticBranch.subbranches):	
			horizontalSeparation = branchIndex2Separation/(pow(horizontalBranchSeparationDivergence, currentBranchIndex1))	#normalise/shorten at greater distance from soma
			posXsubbranch = posX-(horizontalSeparation*((numberOfBranches2-1)/2)) + currentBranchIndex2*horizontalSeparation
			#print("currentBranchIndex2 = ", currentBranchIndex2)
			#print("horizontalSeparation = ", horizontalSeparation)
			#print("posXsubbranch = ", posXsubbranch)
			posYsubbranch = posY+branchIndex1Separation
			#print("posYsubbranch = ", posYsubbranch)
			drawDendriticSANIGraphNodeDendriticBranch(conceptNode, posXsubbranch, posYsubbranch, subbranch, currentBranchIndex1+1, dendriticBranch, posX, posY, activationTime)

def drawDendriticSANIGraphBranch(currentBranchIndex1, parentBranch, currentBranch, drawOrthogonalBranchNode, orthogonalNodeName, activationTime):
	if(drawDendriticSANIGraphEdgeColoursWeights):
		color = getActivationColor(currentBranch, 'darkcyan', 'green')
		weight = 5.0/(currentBranchIndex1+1)

	if(drawDendriticBranchOrthogonal and drawOrthogonalBranchNode):
		#print("orthogonalNodeName = ", orthogonalNodeName)
		if(drawDendriticSANIGraphEdgeColoursWeights):
			dendriticSANIGraph.add_edge(parentBranch.nodeName, orthogonalNodeName, color=color, weight=weight)
			dendriticSANIGraph.add_edge(orthogonalNodeName, currentBranch.nodeName, color=color, weight=weight)
		else:
			dendriticSANIGraph.add_edge(parentBranch.nodeName, orthogonalNodeName)
			dendriticSANIGraph.add_edge(orthogonalNodeName, currentBranch.nodeName)
	else:
		if(drawDendriticSANIGraphEdgeColoursWeights):
			dendriticSANIGraph.add_edge(parentBranch.nodeName, currentBranch.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
		else:
			dendriticSANIGraph.add_edge(parentBranch.nodeName, currentBranch.nodeName)
			
def drawDendriticSANIGraphNodeSequentialSegment(currentBranchIndex1, conceptNode, posX, posY, sequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime):
	colorHtml = 'green'	#branch: green	#'OR #ffffff' invisible: white
	dendriticSANIGraph.add_node(sequentialSegment.nodeName, pos=(posX, posY))
	if(drawDendriticSANIGraphNodeColours):
		dendriticSANIGraphNodeColorMap.append(colorHtml)
		dendriticSANIGraphNodeSizeMap.append(nodeSizeDraw)
	if(not debugOnlyDrawActiveBranches or sequentialSegment.activationLevel):
		drawDendriticSANIGraphSequentialSegment(currentBranchIndex1, sequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime)	#draw sequential segment edge

	#for currentSequentialSegmentInputIndex, currentSequentialSegmentInput in enumerate(sequentialSegment.inputs):
	for currentSequentialSegmentInputIndexDynamic, currentSequentialSegmentInput in enumerate(sequentialSegment.inputs.values()):	#note currentSequentialSegmentInputIndexDynamic is valid even if inputs have been removed from dictionary (although order not guaranteed)
		if(storeSequentialSegmentInputIndexValues):
			currentSequentialSegmentInputIndex = currentSequentialSegmentInput.sequentialSegmentInputIndex
		else:
			currentSequentialSegmentInputIndex = currentSequentialSegmentInputIndexDynamic
		#print("currentSequentialSegmentInputIndex = ", currentSequentialSegmentInputIndex)
		posYsegmentInput = posY+(currentSequentialSegmentInputIndex*sequentialSegmentInputIndexSeparation*spineSeparation) - sequentialSegmentIndexSeparation + nodeSize	#+nodeSize to separate visualisation from sequential segment node	#-branchIndex1Separation to position first input of first sequential segment at base of branch
		drawDendriticSANIGraphNodeSequentialSegmentInput(conceptNode, posX, posYsegmentInput, currentSequentialSegmentInput, currentSequentialSegmentInputIndex, activationTime)
	return sequentialSegment.nodeName
	
def drawDendriticSANIGraphSequentialSegment(currentBranchIndex1, sequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime):
	if(drawDendriticSANIGraphEdgeColoursWeights):
		color = getActivationColor(sequentialSegment, 'cyan', 'green')
		weight = 5.0/(currentBranchIndex1+1)

	#print("orthogonalNodeName = ", orthogonalNodeName)
	if(drawDendriticSANIGraphEdgeColoursWeights):
		dendriticSANIGraph.add_edge(previousSequentialSegmentNodeName, sequentialSegment.nodeName, color=color, weight=weight)
	else:
		dendriticSANIGraph.add_edge(previousSequentialSegmentNodeName, sequentialSegment.nodeName)

def drawDendriticSANIGraphNodeSequentialSegmentInput(conceptNode, posX, posY, sequentialSegmentInput, currentSequentialSegmentInputIndex, activationTime):
	
	if(sequentialSegmentInput.firstInputInSequence):
		color = getActivationColor(sequentialSegmentInput, 'blue', 'orange')	
	else:
		color = getActivationColor(sequentialSegmentInput, 'blue', 'yellow')
				
	#print("sequentialSegmentInput.nodeName = ", sequentialSegmentInput.nodeName)
	#print("posX = ", posX)
	#print("posY = ", posY)
	dendriticSANIGraph.add_node(sequentialSegmentInput.nodeName, pos=(posX, posY))
	if(drawDendriticSANIGraphNodeColours):
		dendriticSANIGraphNodeColorMap.append(color)
		dendriticSANIGraphNodeSizeMap.append(nodeSizeDraw)


def drawDendriticSANIGraphConnection(connection, drawGraphNetwork, activationTime, sentenceConceptNodeList=None):
	node1 = connection.nodeSource
	node2 = connection.nodeTargetSequentialSegmentInput
	#spatioTemporalIndex = connection.spatioTemporalIndex
	if(drawGraphNetwork or (node2.conceptNode in sentenceConceptNodeList)):	#if HFNLPpy_DendriticSANIDrawSentence: ensure target node is in sentence (such that connection can be drawn) - see drawDendriticSANIGraphNodeConnections
		if(drawDendriticSANIGraphEdgeColoursWeights):
			#color = node2.sequentialSegment.branch.branchIndex1	#CHECKTHIS: assign colour of connection based on distance of target neuron synapse to soma 
			color = getActivationColor(connection, 'magenta', 'red', highlightNewActivations=False)
			weight = 1.0				
			dendriticSANIGraph.add_edge(node1.nodeName, node2.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
		else:
			dendriticSANIGraph.add_edge(node1.nodeName, node2.nodeName)
	

		
def displayDendriticSANIGraph(plot=True, save=False, fileName=None):
	pos = nx.get_node_attributes(dendriticSANIGraph, 'pos')
	
	if(highResolutionFigure):
		plt.figure(1, figsize=saveFigSize) 

	if(drawDendriticSANIGraphEdgeColoursWeights):
		edges = dendriticSANIGraph.edges()
		#colors = [dendriticSANIGraph[u][v]['color'] for u,v in edges]
		#weights = [dendriticSANIGraph[u][v]['weight'] for u,v in edges]	
		colors = nx.get_edge_attributes(dendriticSANIGraph,'color').values()
		weights = nx.get_edge_attributes(dendriticSANIGraph,'weight').values()
		#print("size dendriticSANIGraph.nodes = ", len(dendriticSANIGraph.nodes))
		#print("size dendriticSANIGraphNodeColorMap = ", len(dendriticSANIGraphNodeColorMap))
		if(drawDendriticSANIGraphNodeColours):
			nx.draw(dendriticSANIGraph, pos, with_labels=False, alpha=graphTransparency, node_color=dendriticSANIGraphNodeColorMap, edge_color=colors, width=list(weights), node_size=dendriticSANIGraphNodeSizeMap)
		else:
			nx.draw(dendriticSANIGraph, pos, with_labels=False, alpha=graphTransparency, edge_color=colors, width=list(weights), node_size=nodeSizeDraw)
	else:
		if(drawDendriticSANIGraphNodeColours):
			nx.draw(dendriticSANIGraph, pos, with_labels=False, alpha=graphTransparency, node_color=dendriticSANIGraphNodeColorMap, node_size=dendriticSANIGraphNodeSizeMap)
		else:
			nx.draw(dendriticSANIGraph, pos, with_labels=False, alpha=graphTransparency, node_size=nodeSizeDraw)

	#if(useAlgorithmDendriticSANIbiologicalSimulation) exclusive code:
	#only assign labels to conceptNeurons
	labels = {}    
	for node in dendriticSANIGraph.nodes():
		if node in dendriticSANIGraphConceptNodesList:
			#set the node name as the key and the label as its value 
			labels[node] = node
	nx.draw_networkx_labels(dendriticSANIGraph, pos, labels, font_size=8)	#font_size=16, font_color='r'
	
	if(save):
		if(highResolutionFigure):
			plt.savefig(fileName, dpi=saveFigDPI)
		else:
			plt.savefig(fileName)
	if(plot):
		plt.show()
	else:
		plt.clf()

	

def getActivationColor(neuronObject, colorActive, colorInactive, highlightNewActivations=True):
	if(drawBiologicalSimulationDynamicHighlightNewActivations and highlightNewActivations and neuronObject.activationStateNew):	#highlightNewActivations or neuronObject.objectType != objectTypeConnection
		color = highlightNewActivationColor
		neuronObject.activationStateNew = False
	elif(drawBiologicalSimulationDynamicFrozenActivations and neuronObject.objectType==objectTypeSequentialSegment and neuronObject.frozen):
		color = frozenActivationColor
	else:		
		if(neuronObject.activationLevel):
			color = colorActive
		else:
			color = colorInactive
	return color
	
	

