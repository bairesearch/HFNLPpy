"""HFNLPpy_biologicalSimulationDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
ATNLP Biological Simulation Draw - draw sentence/network graph with dendritic trees

definition of colour scheme: HFNLPbiologicalImplementationDevelopment-07June2022b.pdf

"""

import networkx as nx
import matplotlib.pyplot as plt
from math import cos, sin, radians
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *

drawHopfieldGraphEdgeColoursWeights = True
drawHopfieldGraphNodeColours = True	#node colours not yet coded (pos type of concept node will be different depending on connectivity/instance context)
graphTransparency = 0.5

hopfieldGraph = nx.Graph()	#MultiDiGraph: Directed graphs with self loops and parallel edges	#https://networkx.org/documentation/stable/reference/classes/multidigraph.html
hopfieldGraphNodeColorMap = []
hopfieldGraphConceptNodesList = []	#primary nodes for label assignment

#require calibration (depends on numberOfBranches1/numberOfBranches2/numberOfBranchSequentialSegments):
conceptNeuronIndexSeparation = 20.0
branchIndex1Separation = 10.0/numberOfBranches1	#vertical separation
branchIndex2Separation = conceptNeuronIndexSeparation/numberOfBranches2	#horizontal separation at branchIndex=1 (will decrease at higher vertical separation)
sequentialSegmentIndexSeparation = 10.0/numberOfBranchSequentialSegments/2.0
sequentialSegmentInputIndexSeparation = 0.5

nodeSize = 0.5	#node diameter
nodeSizeDraw = 10.0	#node diameter

def clearHopfieldGraph():
	hopfieldGraph.clear()	#only draw graph for single sentence
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.clear()
	hopfieldGraphConceptNodesList.clear()	#for labels

def drawHopfieldGraphNode(conceptNode, drawGraphNetwork):
	colorHtml = '#00cccc'	#soma: turquoise
	#print("conceptNode.networkIndex = ", conceptNode.networkIndex)
	if(drawGraphNetwork):
		posX, posY = (conceptNode.networkIndex*conceptNeuronIndexSeparation, 0)	#y=0: currently align concept neurons along single plane
	else:
		posX, posY = (conceptNode.w*conceptNeuronIndexSeparation, 0)	#y=0: currently align concept neurons along single plane
	#print("drawHopfieldGraphNode: ", conceptNode.nodeName)
	hopfieldGraph.add_node(conceptNode.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(colorHtml)
	hopfieldGraphConceptNodesList.append(conceptNode.nodeName)

	#if(biologicalSimulation) exclusive code:
	posYdendriticTreeBranchHead = posY+branchIndex1Separation	#position of first branching within dendritic tree
	drawHopfieldGraphNodeDendriticBranch(conceptNode, posX, posYdendriticTreeBranchHead, conceptNode.dendriticTree, 0)
	drawHopfieldGraphBranch(conceptNode, conceptNode.dendriticTree)	#draw branch edge

#if(biologicalSimulation) exclusive code:
	
def drawHopfieldGraphNodeDendriticBranch(conceptNode, posX, posY, dendriticBranch, currentBranchIndex1):
	#print("drawHopfieldGraphNodeDendriticBranch: , currentBranchIndex1 = ", currentBranchIndex1, ", posX = ", posX)
	#print("posY = ", posY)
	
	colorHtml = 'green' #branch: green	#'OR #ffffff' invisible: white
	hopfieldGraph.add_node(dendriticBranch.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(colorHtml)
					
	for currentSequentialSegmentIndex, currentSequentialSegment in enumerate(dendriticBranch.sequentialSegments):
		posYsequentialSegment = posY+currentSequentialSegmentIndex*sequentialSegmentIndexSeparation - branchIndex1Separation/2	#-(branchIndex1Separation/2) to position sequentialSegment in middle of branch	#OLD: +nodeSize to separate visualisation from branch node	
		drawHopfieldGraphNodeSequentialSegment(conceptNode, posX, posYsequentialSegment, currentSequentialSegment, currentSequentialSegmentIndex)
		
	for currentBranchIndex2, subbranch in enumerate(dendriticBranch.subbranches):	
		horizontalSeparation = branchIndex2Separation/(pow(2, currentBranchIndex1))	#normalise/shorten at greater distance from soma
		posXsubbranch = posX+currentBranchIndex2*horizontalSeparation
		#print("currentBranchIndex2 = ", currentBranchIndex2)
		#print("horizontalSeparation = ", horizontalSeparation)
		#print("posXsubbranch = ", posXsubbranch)
		posYsubbranch = posY+branchIndex1Separation
		#print("posYsubbranch = ", posYsubbranch)
		drawHopfieldGraphNodeDendriticBranch(conceptNode, posXsubbranch, posYsubbranch, subbranch, currentBranchIndex1+1)	
		drawHopfieldGraphBranch(dendriticBranch, subbranch)	#draw branch edge

def drawHopfieldGraphNodeSequentialSegment(conceptNode, posX, posY, sequentialSegment, currentSequentialSegmentIndex):
	if(numberOfBranchSequentialSegments > 1):	#only draw sequential segments if more than 1 sequential segment allowed in branch
		colorHtml = 'magenta'	#sequentialSegment: orange	or white/invisible
		hopfieldGraph.add_node(sequentialSegment.nodeName, pos=(posX, posY))
		if(drawHopfieldGraphNodeColours):
			hopfieldGraphNodeColorMap.append(colorHtml)
	
	for currentSequentialSegmentInputIndex, currentSequentialSegmentInput in enumerate(sequentialSegment.inputs):
		posYsegmentInput = posY+currentSequentialSegmentInputIndex*sequentialSegmentInputIndexSeparation + nodeSize	#+nodeSize to separate visualisation from sequential segment node
		drawHopfieldGraphNodeSequentialSegmentInput(conceptNode, posX, posYsegmentInput, currentSequentialSegmentInput, currentSequentialSegmentInputIndex)
		
def drawHopfieldGraphNodeSequentialSegmentInput(conceptNode, posX, posY, sequentialSegmentInput, currentSequentialSegmentInputIndex):
	colorHtml = 'yellow'	#synapse: yellow
	#print("sequentialSegmentInput.nodeName = ", sequentialSegmentInput.nodeName)
	#print("posX = ", posX)
	#print("posY = ", posY)
	hopfieldGraph.add_node(sequentialSegmentInput.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(colorHtml)

def drawHopfieldGraphBranch(parentBranch, currentBranch):
	if(drawHopfieldGraphEdgeColoursWeights):
		color = 'green'	#dendrite: green
		weight = 1.0
		hopfieldGraph.add_edge(parentBranch.nodeName, currentBranch.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
	else:
		hopfieldGraph.add_edge(parentBranch.nodeName, currentBranch.nodeName)
		
		
def drawHopfieldGraphConnection(connection, drawGraphNetwork, sentenceConceptNodeList=None):
	node1 = connection.nodeSource
	node2 = connection.nodeTargetSequentialSegmentInput
	spatioTemporalIndex = connection.spatioTemporalIndex
	if(drawGraphNetwork or (node2.conceptNode in sentenceConceptNodeList)):	#if HFNLPpy_biologicalSimulationDrawSentence: ensure target node is in sentence (such that connection can be drawn) - see drawHopfieldGraphNodeConnections
		if(drawHopfieldGraphEdgeColoursWeights):
			#color = node2.sequentialSegment.branch.branchIndex1	#CHECKTHIS: assign colour of connection based on distance of target neuron synapse to soma 
			color = 'red'	#axon: red
			weight = 1.0
			hopfieldGraph.add_edge(node1.nodeName, node2.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
		else:
			hopfieldGraph.add_edge(node1.nodeName, node2.nodeName)
	

def displayHopfieldGraph():
	pos = nx.get_node_attributes(hopfieldGraph, 'pos')
	if(drawHopfieldGraphEdgeColoursWeights):
		edges = hopfieldGraph.edges()
		#colors = [hopfieldGraph[u][v]['color'] for u,v in edges]
		#weights = [hopfieldGraph[u][v]['weight'] for u,v in edges]	
		colors = nx.get_edge_attributes(hopfieldGraph,'color').values()
		weights = nx.get_edge_attributes(hopfieldGraph,'weight').values()
		#print("size hopfieldGraph.nodes = ", len(hopfieldGraph.nodes))
		#print("size hopfieldGraphNodeColorMap = ", len(hopfieldGraphNodeColorMap))
		if(drawHopfieldGraphNodeColours):
			nx.draw(hopfieldGraph, pos, with_labels=False, alpha=graphTransparency, node_color=hopfieldGraphNodeColorMap, edge_color=colors, width=list(weights), node_size=nodeSizeDraw)
		else:
			nx.draw(hopfieldGraph, pos, with_labels=False, alpha=graphTransparency, edge_color=colors, width=list(weights), node_size=nodeSizeDraw)
	else:
		if(drawHopfieldGraphNodeColours):
			nx.draw(hopfieldGraph, pos, with_labels=False, alpha=graphTransparency, node_color=hopfieldGraphNodeColorMap, node_size=nodeSizeDraw)
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
	
	plt.show()

def drawHopfieldGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList=None):
	for connectionKey, connectionList in hopfieldGraphNode.targetConnectionDict.items():
		for connection in connectionList:
			drawHopfieldGraphConnection(connection, drawGraphNetwork, sentenceConceptNodeList)
			
def drawHopfieldGraphNodeAndConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList=None):	
	#parse tree and generate nodes and connections
	drawHopfieldGraphNode(hopfieldGraphNode, drawGraphNetwork)
	drawHopfieldGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList)

def drawHopfieldGraphSentence(sentenceConceptNodeList):	
	sentenceConceptNodeList = list(set(sentenceConceptNodeList))	#generate a unique list from a list (in the event a sentence contains multiple instances of the same word/lemma)
	
	#print("drawHopfieldGraphSentence = ")
	#print("size hopfieldGraph.nodes = ", len(hopfieldGraph.nodes))
	#print("size hopfieldGraphNodeColorMap = ", len(hopfieldGraphNodeColorMap))
	drawGraphNetwork = False
	#networkSize = len(sentenceConceptNodeList)
	#need to draw all conceptNodes and their dendriticTrees before creating connections
	for conceptNode in sentenceConceptNodeList:
		drawHopfieldGraphNode(conceptNode, drawGraphNetwork)
	for conceptNode in sentenceConceptNodeList:
		drawHopfieldGraphNodeConnections(conceptNode, drawGraphNetwork, sentenceConceptNodeList)

def drawHopfieldGraphNetwork(networkConceptNodeDict):	
	#print("drawHopfieldGraphNetwork = ")
	#print("size hopfieldGraph.nodes = ", len(hopfieldGraph.nodes))
	#print("size hopfieldGraphNodeColorMap = ", len(hopfieldGraphNodeColorMap))
	#generate nodes and connections
	drawGraphNetwork = True
	#networkSize = len(networkConceptNodeDict)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		drawHopfieldGraphNode(conceptNode, drawGraphNetwork)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		drawHopfieldGraphNodeConnections(conceptNode, drawGraphNetwork)
		
def pointOnCircle(radius, angleDegrees, centre=[0,0]):
	angle = radians(angleDegrees)
	x = centre[0] + (radius * cos(angle))
	y = centre[1] + (radius * sin(angle))
	return x, y
	
