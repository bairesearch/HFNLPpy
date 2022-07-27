"""HFNLPpy_biologicalSimulationDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Draw - draw sentence/network graph with dendritic trees

definition of colour scheme: HFNLPbiologicalImplementationDevelopment-07June2022b.pdf

"""

import networkx as nx
import matplotlib.pyplot as plt
plt.ioff()	# Turn interactive plotting off
from math import cos, sin, radians
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_biologicalSimulationGlobalDefs import *
from HFNLPpy_biologicalSimulationNode import *

if(writeBiologicalSimulation):
	import HFNLPpy_biologicalSimulationXML

highResolutionFigure = True
if(highResolutionFigure):
	saveFigDPI = 300	#approx HD	#depth per inch
	saveFigSize = (16,9)	#in inches
	
debugOnlyDrawTargetNeuron = False
debugOnlyDrawActiveBranches = False

drawHopfieldGraphEdgeColoursWeights = True
drawHopfieldGraphNodeColours = True	#node colours not yet coded (pos type of concept node will be different depending on connectivity/instance context)
graphTransparency = 0.5

hopfieldGraph = nx.Graph()	#MultiDiGraph: Directed graphs with self loops and parallel edges	#https://networkx.org/documentation/stable/reference/classes/multidigraph.html
hopfieldGraphNodeColorMap = []
hopfieldGraphNodeSizeMap = []
hopfieldGraphConceptNodesList = []	#primary nodes for label assignment

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


def drawBiologicalSimulationStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences=0):
	if(drawBiologicalSimulation):
		if(drawBiologicalSimulationSentence):
			fileName = generateBiologicalSimulationFileName(True, sentenceIndex, write=False)
			clearHopfieldGraph()
			drawHopfieldGraphSentence(sentenceConceptNodeList)
			print("drawBiologicalSimulationSentence: HFNLPpy_biologicalSimulationDraw.displayHopfieldGraph()")
			displayHopfieldGraph(drawBiologicalSimulationPlot, drawBiologicalSimulationSave, fileName)
		if(drawBiologicalSimulationNetwork):
			fileName = generateBiologicalSimulationFileName(False, sentenceIndex, write=False)
			clearHopfieldGraph()
			drawHopfieldGraphNetwork(networkConceptNodeDict)
			print("drawBiologicalSimulationNetwork: HFNLPpy_biologicalSimulationDraw.displayHopfieldGraph()")
			displayHopfieldGraph(drawBiologicalSimulationPlot, drawBiologicalSimulationSave, fileName)	
	if(writeBiologicalSimulation):
		if(writeBiologicalSimulationSentence):	
			fileName = generateBiologicalSimulationFileName(True, sentenceIndex, write=True)
			HFNLPpy_biologicalSimulationXML.writeHopfieldGraphSentence(sentenceConceptNodeList, fileName)
		if(writeBiologicalSimulationNetwork):
			if(not outputBiologicalSimulationNetworkLastSentenceOnly or (sentenceIndex == numberOfSentences-1)):
				fileName = generateBiologicalSimulationFileName(False, sentenceIndex, write=True)
				HFNLPpy_biologicalSimulationXML.writeHopfieldGraphNetwork(networkConceptNodeDict, fileName)
	
def drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex, activationTime, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivation or (sentenceIndex == sentenceIndexDebug and wTarget == wSource+1)):
			#if(vectorisedComputation or emulateVectorisedComputationOrder):
			#	print("branchIndex1 = ", branchIndex1, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
			if(drawBiologicalSimulationSentenceDynamic):
				fileName = generateBiologicalSimulationDynamicSequentialSegmentFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				clearHopfieldGraph()
				drawHopfieldGraphSentence(sentenceConceptNodeList, activationTime=activationTime, wTarget=wTargetDebug)
				displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationNetworkDynamic):
				fileName = generateBiologicalSimulationDynamicSequentialSegmentFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				clearHopfieldGraph()
				drawHopfieldGraphNetwork(networkConceptNodeDict, activationTime=activationTime, wTarget=wTargetDebug)
				displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)				
	if(writeBiologicalSimulationDynamic):
		if(writeBiologicalSimulationSentenceDynamic):	
			fileName = generateBiologicalSimulationDynamicSequentialSegmentFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
			HFNLPpy_biologicalSimulationXML.writeHopfieldGraphSentence(sentenceConceptNodeList, fileName, activationTime=activationTime)
		if(writeBiologicalSimulationNetworkDynamic):
			#if(sentenceIndex == numberOfSentences-1):
			fileName = generateBiologicalSimulationDynamicSequentialSegmentFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
			HFNLPpy_biologicalSimulationXML.writeHopfieldGraphNetwork(networkConceptNodeDict, fileName, activationTime=activationTime)
				
def drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivation or (sentenceIndex == sentenceIndexDebug and wSource >= wSourceDebug)):	#wSource == wSourceDebug
			if(drawBiologicalSimulationSentenceDynamic):
				fileName = generateBiologicalSimulationDynamicNeuronFileName(True, wSource, sentenceIndex)
				clearHopfieldGraph()
				drawHopfieldGraphSentence(sentenceConceptNodeList, activationTime=activationTime, wTarget=wTargetDebug)
				displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationNetworkDynamic):
				fileName = generateBiologicalSimulationDynamicNeuronFileName(False, wSource, sentenceIndex)
				clearHopfieldGraph()
				drawHopfieldGraphNetwork(networkConceptNodeDict, activationTime=activationTime, wTarget=wTargetDebug)
				displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)	
	if(writeBiologicalSimulationDynamic):
		if(writeBiologicalSimulationSentenceDynamic):	
			fileName = generateBiologicalSimulationDynamicNeuronFileName(True, wSource, sentenceIndex)
			HFNLPpy_biologicalSimulationXML.writeHopfieldGraphSentence(sentenceConceptNodeList, fileName, activationTime=activationTime)
		if(writeBiologicalSimulationNetworkDynamic):
			#if(sentenceIndex == numberOfSentences-1):
			fileName = generateBiologicalSimulationDynamicNeuronFileName(False, wSource, sentenceIndex)
			HFNLPpy_biologicalSimulationXML.writeHopfieldGraphNetwork(networkConceptNodeDict, fileName, activationTime=activationTime)
								
def clearHopfieldGraph():
	hopfieldGraph.clear()	#only draw graph for single sentence
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.clear()
		hopfieldGraphNodeSizeMap.clear()
	hopfieldGraphConceptNodesList.clear()	#for labels

def drawHopfieldGraphSentence(sentenceConceptNodeList, activationTime=None, wTarget=None):	
	sentenceConceptNodeList = list(set(sentenceConceptNodeList))	#generate a unique list from a list (in the event a sentence contains multiple instances of the same word/lemma)
	
	#print("drawHopfieldGraphSentence = ")
	#print("size hopfieldGraph.nodes = ", len(hopfieldGraph.nodes))
	#print("size hopfieldGraphNodeColorMap = ", len(hopfieldGraphNodeColorMap))
	drawGraphNetwork = False
	#networkSize = len(sentenceConceptNodeList)
	#need to draw all conceptNodes and their dendriticTrees before creating connections
	for conceptNode in sentenceConceptNodeList:
		if(not debugOnlyDrawTargetNeuron or (conceptNode.w==wTarget)):
			drawHopfieldGraphNode(conceptNode, drawGraphNetwork, activationTime)
	for conceptNode in sentenceConceptNodeList:
		if(not debugOnlyDrawTargetNeuron):
			drawHopfieldGraphNodeConnections(conceptNode, drawGraphNetwork, activationTime, sentenceConceptNodeList)

def drawHopfieldGraphNetwork(networkConceptNodeDict, activationTime=None, wTarget=None):	
	#print("drawHopfieldGraphNetwork = ")
	#print("size hopfieldGraph.nodes = ", len(hopfieldGraph.nodes))
	#print("size hopfieldGraphNodeColorMap = ", len(hopfieldGraphNodeColorMap))
	#generate nodes and connections
	drawGraphNetwork = True
	#networkSize = len(networkConceptNodeDict)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		if(not debugOnlyDrawTargetNeuron or (conceptNode.w==wTarget)):
			drawHopfieldGraphNode(conceptNode, drawGraphNetwork, activationTime)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		if(not debugOnlyDrawTargetNeuron):
			drawHopfieldGraphNodeConnections(conceptNode, drawGraphNetwork, activationTime)

def drawHopfieldGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, activationTime, sentenceConceptNodeList=None):
	for connectionKey, connectionList in hopfieldGraphNode.targetConnectionDict.items():
		for connection in connectionList:
			drawHopfieldGraphConnection(connection, drawGraphNetwork, activationTime, sentenceConceptNodeList)
			
#def drawHopfieldGraphNodeAndConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList=None):	
#	#parse tree and generate nodes and connections
#	drawHopfieldGraphNode(hopfieldGraphNode, drawGraphNetwork)
#	drawHopfieldGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList)
		
def drawHopfieldGraphNode(conceptNode, drawGraphNetwork, activationTime):
	if(conceptNode.activationLevel):
		colorHtml = 'turquoise'	#soma: turquoise	
	else:
		colorHtml = 'darkgreen'	#soma: 	darkgreen	(orig: turquoise)
	#print("conceptNode.networkIndex = ", conceptNode.networkIndex)
	
	if(debugOnlyDrawTargetNeuron):
		posX = len(hopfieldGraphConceptNodesList)*conceptNeuronIndexSeparation
	else:
		if(drawGraphNetwork):
			posX = conceptNode.networkIndex*conceptNeuronIndexSeparation
		else:
			posX = conceptNode.w*conceptNeuronIndexSeparation
	posY = 0	#y=0: currently align concept neurons along single plane
	
	#print("drawHopfieldGraphNode: ", conceptNode.nodeName)
	hopfieldGraph.add_node(conceptNode.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(colorHtml)
		if(drawGraphNetwork):
			hopfieldGraphNodeSizeMap.append(conceptNodeSizeDrawNetwork)
		else:
			hopfieldGraphNodeSizeMap.append(conceptNodeSizeDrawSentence)
	hopfieldGraphConceptNodesList.append(conceptNode.nodeName)

	#if(biologicalSimulation) exclusive code:
	posYdendriticTreeBranchHead = posY+branchIndex1Separation	#position of first branching within dendritic tree
	currentBranchIndex1 = 0
	drawHopfieldGraphNodeDendriticBranch(conceptNode, posX, posYdendriticTreeBranchHead, conceptNode.dendriticTree, currentBranchIndex1, conceptNode, posX, posY, activationTime, drawOrthogonalBranchNode=False)

#if(biologicalSimulation) exclusive code:
	
def drawHopfieldGraphNodeDendriticBranch(conceptNode, posX, posY, dendriticBranch, currentBranchIndex1, previousBranch, previousConceptNodePosX, previousConceptNodePosY, activationTime, drawOrthogonalBranchNode=True):
	#print("drawHopfieldGraphNodeDendriticBranch: , dendriticBranch.nodeName = ", dendriticBranch.nodeName, ", currentBranchIndex1 = ", currentBranchIndex1, ", posX = ", posX, ", posY = ", posY)
	
	colorHtml = 'green' #branch: green	#'OR #ffffff' invisible: white
	hopfieldGraph.add_node(dendriticBranch.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(colorHtml)
		hopfieldGraphNodeSizeMap.append(nodeSizeDraw)
	if(drawOrthogonalBranchNode and drawDendriticBranchOrthogonal):
		colorHtml = 'white'
		orthogonalNodeName = dendriticBranch.nodeName + "Orthogonal"
		hopfieldGraph.add_node(orthogonalNodeName, pos=(posX, previousConceptNodePosY))	#draw another node directly below the branch head node (this should be invisible)
		if(drawHopfieldGraphNodeColours):
			hopfieldGraphNodeColorMap.append(colorHtml)
			hopfieldGraphNodeSizeMap.append(nodeSizeDraw)
	else:
		orthogonalNodeName = None
	if(not debugOnlyDrawActiveBranches or dendriticBranch.activationLevel):
		drawHopfieldGraphBranch(currentBranchIndex1, previousBranch, dendriticBranch, drawOrthogonalBranchNode, orthogonalNodeName, activationTime)	#draw branch edge
	
	if(drawOrthogonalBranchNode and drawDendriticBranchOrthogonal):
		previousSequentialSegmentNodeName = orthogonalNodeName					
	else:
		previousSequentialSegmentNodeName = previousBranch.nodeName
	for currentSequentialSegmentIndex, currentSequentialSegment in enumerate(dendriticBranch.sequentialSegments):
		posYsequentialSegment = posY+currentSequentialSegmentIndex*sequentialSegmentIndexSeparation - (sequentialSegmentIndexSeparation*(numberOfBranchSequentialSegments-1))
		previousSequentialSegmentNodeName = drawHopfieldGraphNodeSequentialSegment(currentBranchIndex1, conceptNode, posX, posYsequentialSegment, currentSequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime)
		
	for currentBranchIndex2, subbranch in enumerate(dendriticBranch.subbranches):	
		horizontalSeparation = branchIndex2Separation/(pow(horizontalBranchSeparationDivergence, currentBranchIndex1))	#normalise/shorten at greater distance from soma
		posXsubbranch = posX-(horizontalSeparation*((numberOfBranches2-1)/2)) + currentBranchIndex2*horizontalSeparation
		#print("currentBranchIndex2 = ", currentBranchIndex2)
		#print("horizontalSeparation = ", horizontalSeparation)
		#print("posXsubbranch = ", posXsubbranch)
		posYsubbranch = posY+branchIndex1Separation
		#print("posYsubbranch = ", posYsubbranch)
		drawHopfieldGraphNodeDendriticBranch(conceptNode, posXsubbranch, posYsubbranch, subbranch, currentBranchIndex1+1, dendriticBranch, posX, posY, activationTime)

def drawHopfieldGraphBranch(currentBranchIndex1, parentBranch, currentBranch, drawOrthogonalBranchNode, orthogonalNodeName, activationTime):
	if(drawHopfieldGraphEdgeColoursWeights):
		color = getActivationColor(currentBranch, 'darkcyan', 'green')
		weight = 5.0/(currentBranchIndex1+1)

	if(drawDendriticBranchOrthogonal and drawOrthogonalBranchNode):
		#print("orthogonalNodeName = ", orthogonalNodeName)
		if(drawHopfieldGraphEdgeColoursWeights):
			hopfieldGraph.add_edge(parentBranch.nodeName, orthogonalNodeName, color=color, weight=weight)
			hopfieldGraph.add_edge(orthogonalNodeName, currentBranch.nodeName, color=color, weight=weight)
		else:
			hopfieldGraph.add_edge(parentBranch.nodeName, orthogonalNodeName)
			hopfieldGraph.add_edge(orthogonalNodeName, currentBranch.nodeName)
	else:
		if(drawHopfieldGraphEdgeColoursWeights):
			hopfieldGraph.add_edge(parentBranch.nodeName, currentBranch.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
		else:
			hopfieldGraph.add_edge(parentBranch.nodeName, currentBranch.nodeName)
			
def drawHopfieldGraphNodeSequentialSegment(currentBranchIndex1, conceptNode, posX, posY, sequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime):
	colorHtml = 'green'	#branch: green	#'OR #ffffff' invisible: white
	hopfieldGraph.add_node(sequentialSegment.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(colorHtml)
		hopfieldGraphNodeSizeMap.append(nodeSizeDraw)
	if(not debugOnlyDrawActiveBranches or sequentialSegment.activationLevel):
		drawHopfieldGraphSequentialSegment(currentBranchIndex1, sequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime)	#draw sequential segment edge

	#for currentSequentialSegmentInputIndex, currentSequentialSegmentInput in enumerate(sequentialSegment.inputs):
	for currentSequentialSegmentInputIndexDynamic, currentSequentialSegmentInput in enumerate(sequentialSegment.inputs.values()):	#note currentSequentialSegmentInputIndexDynamic is valid even if inputs have been removed from dictionary (although order not guaranteed)
		if(storeSequentialSegmentInputIndexValues):
			currentSequentialSegmentInputIndex = currentSequentialSegmentInput.sequentialSegmentInputIndex
		else:
			currentSequentialSegmentInputIndex = currentSequentialSegmentInputIndexDynamic
		#print("currentSequentialSegmentInputIndex = ", currentSequentialSegmentInputIndex)
		posYsegmentInput = posY+(currentSequentialSegmentInputIndex*sequentialSegmentInputIndexSeparation*spineSeparation) - sequentialSegmentIndexSeparation + nodeSize	#+nodeSize to separate visualisation from sequential segment node	#-branchIndex1Separation to position first input of first sequential segment at base of branch
		drawHopfieldGraphNodeSequentialSegmentInput(conceptNode, posX, posYsegmentInput, currentSequentialSegmentInput, currentSequentialSegmentInputIndex, activationTime)
	return sequentialSegment.nodeName
	
def drawHopfieldGraphSequentialSegment(currentBranchIndex1, sequentialSegment, currentSequentialSegmentIndex, previousSequentialSegmentNodeName, activationTime):
	if(drawHopfieldGraphEdgeColoursWeights):
		color = getActivationColor(sequentialSegment, 'cyan', 'green')
		weight = 5.0/(currentBranchIndex1+1)

	#print("orthogonalNodeName = ", orthogonalNodeName)
	if(drawHopfieldGraphEdgeColoursWeights):
		hopfieldGraph.add_edge(previousSequentialSegmentNodeName, sequentialSegment.nodeName, color=color, weight=weight)
	else:
		hopfieldGraph.add_edge(previousSequentialSegmentNodeName, sequentialSegment.nodeName)

def drawHopfieldGraphNodeSequentialSegmentInput(conceptNode, posX, posY, sequentialSegmentInput, currentSequentialSegmentInputIndex, activationTime):
	
	if(sequentialSegmentInput.firstInputInSequence):
		color = getActivationColor(sequentialSegmentInput, 'blue', 'orange')	
	else:
		color = getActivationColor(sequentialSegmentInput, 'blue', 'yellow')
				
	#print("sequentialSegmentInput.nodeName = ", sequentialSegmentInput.nodeName)
	#print("posX = ", posX)
	#print("posY = ", posY)
	hopfieldGraph.add_node(sequentialSegmentInput.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(color)
		hopfieldGraphNodeSizeMap.append(nodeSizeDraw)


def drawHopfieldGraphConnection(connection, drawGraphNetwork, activationTime, sentenceConceptNodeList=None):
	node1 = connection.nodeSource
	node2 = connection.nodeTargetSequentialSegmentInput
	spatioTemporalIndex = connection.spatioTemporalIndex
	if(drawGraphNetwork or (node2.conceptNode in sentenceConceptNodeList)):	#if HFNLPpy_biologicalSimulationDrawSentence: ensure target node is in sentence (such that connection can be drawn) - see drawHopfieldGraphNodeConnections
		if(drawHopfieldGraphEdgeColoursWeights):
			#color = node2.sequentialSegment.branch.branchIndex1	#CHECKTHIS: assign colour of connection based on distance of target neuron synapse to soma 
			color = getActivationColor(connection, 'magenta', 'red', highlightNewActivations=False)
			weight = 1.0				
			hopfieldGraph.add_edge(node1.nodeName, node2.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
		else:
			hopfieldGraph.add_edge(node1.nodeName, node2.nodeName)
	

		
def displayHopfieldGraph(plot=True, save=False, fileName=None):
	pos = nx.get_node_attributes(hopfieldGraph, 'pos')
	
	if(highResolutionFigure):
		plt.figure(1, figsize=saveFigSize) 

	if(drawHopfieldGraphEdgeColoursWeights):
		edges = hopfieldGraph.edges()
		#colors = [hopfieldGraph[u][v]['color'] for u,v in edges]
		#weights = [hopfieldGraph[u][v]['weight'] for u,v in edges]	
		colors = nx.get_edge_attributes(hopfieldGraph,'color').values()
		weights = nx.get_edge_attributes(hopfieldGraph,'weight').values()
		#print("size hopfieldGraph.nodes = ", len(hopfieldGraph.nodes))
		#print("size hopfieldGraphNodeColorMap = ", len(hopfieldGraphNodeColorMap))
		if(drawHopfieldGraphNodeColours):
			nx.draw(hopfieldGraph, pos, with_labels=False, alpha=graphTransparency, node_color=hopfieldGraphNodeColorMap, edge_color=colors, width=list(weights), node_size=hopfieldGraphNodeSizeMap)
		else:
			nx.draw(hopfieldGraph, pos, with_labels=False, alpha=graphTransparency, edge_color=colors, width=list(weights), node_size=nodeSizeDraw)
	else:
		if(drawHopfieldGraphNodeColours):
			nx.draw(hopfieldGraph, pos, with_labels=False, alpha=graphTransparency, node_color=hopfieldGraphNodeColorMap, node_size=hopfieldGraphNodeSizeMap)
		else:
			nx.draw(hopfieldGraph, pos, with_labels=False, alpha=graphTransparency, node_size=nodeSizeDraw)

	#if(biologicalSimulation) exclusive code:
	#only assign labels to conceptNeurons
	labels = {}    
	for node in hopfieldGraph.nodes():
		if node in hopfieldGraphConceptNodesList:
			#set the node name as the key and the label as its value 
			labels[node] = node
	nx.draw_networkx_labels(hopfieldGraph, pos, labels, font_size=8)	#font_size=16, font_color='r'
	
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
	
	

