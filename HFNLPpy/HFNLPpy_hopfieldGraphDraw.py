"""HFNLPpy_hopfieldGraphDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Hopfield Graph Draw Class

"""

import networkx as nx
import matplotlib.pyplot as plt
from math import cos, sin, radians
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_globalDefs import *

highResolutionFigure = True
if(highResolutionFigure):
	saveFigDPI = 300	#approx HD	#depth per inch
	saveFigSize = (16,9)	#(9,9)	#in inches
	
drawHopfieldGraphEdgeColours = True
drawHopfieldGraphEdgeWeights = False
if(drawHopfieldGraphEdgeColours or drawHopfieldGraphEdgeWeights):
	drawHopfieldGraphEdgeColoursWeights = True
if(useAlgorithmScanBiologicalSimulation):
	drawHopfieldGraphNodeColours = True	#colour activated nodes
else:
	drawHopfieldGraphNodeColours = False	#node colours not yet coded (pos type of concept node will be different depending on connectivity/instance context)
graphTransparency = 0.5

hopfieldGraph = nx.MultiDiGraph()	#Directed graphs with self loops and parallel edges	#https://networkx.org/documentation/stable/reference/classes/multidigraph.html
hopfieldGraphNodeColorMap = []
hopfieldGraphRadius = 100
hopfieldGraphCentre = [0, 0]
	
def drawHopfieldGraphSentenceStatic(sentenceIndex, sentenceConceptNodeList, networkSize, drawHopfieldGraphPlot, drawHopfieldGraphSave):
	print("HFNLPpy_hopfieldGraphDraw.drawHopfieldGraphSentenceStatic()")
	sentenceOrNetwork = True
	clearHopfieldGraph()
	fileName = generateHopfieldGraphFileName(True, sentenceIndex)
	drawHopfieldGraphSentence(sentenceConceptNodeList, networkSize)
	displayHopfieldGraph(drawHopfieldGraphPlot, drawHopfieldGraphSave, fileName)

def drawHopfieldGraphNetworkStatic(sentenceIndex, networkConceptNodeDict, drawHopfieldGraphPlot, drawHopfieldGraphSave):
	print("HFNLPpy_hopfieldGraphDraw.drawHopfieldGraphNetworkStatic()")
	sentenceOrNetwork = False
	clearHopfieldGraph()
	fileName = generateHopfieldGraphFileName(False, sentenceIndex)
	drawHopfieldGraphNetwork(networkConceptNodeDict)
	displayHopfieldGraph(drawHopfieldGraphPlot, drawHopfieldGraphSave, fileName)
				
def setColourHopfieldNodes(value):
    global drawHopfieldGraphNodeColours
    drawHopfieldGraphNodeColours = value

def clearHopfieldGraph():
	hopfieldGraph.clear()	#only draw graph for single sentence
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.clear()

def drawHopfieldGraphSentence(sentenceConceptNodeList, networkSize):
	#need to draw all conceptNodes before creating connections
	drawGraphNetwork = False
	for conceptNode in sentenceConceptNodeList:
		drawHopfieldGraphNode(conceptNode, networkSize)
	for conceptNode in sentenceConceptNodeList:
		drawHopfieldGraphNodeConnections(conceptNode, drawGraphNetwork, sentenceConceptNodeList)	

def drawHopfieldGraphNetwork(networkConceptNodeDict):	
	#generate nodes and connections
	drawGraphNetwork = True
	networkSize = len(networkConceptNodeDict)
	#need to draw all conceptNodes before creating connections
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		drawHopfieldGraphNode(conceptNode, networkSize)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
		drawHopfieldGraphNodeConnections(conceptNode, drawGraphNetwork)	
			
def drawHopfieldGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList=None):
	if(assignSingleConnectionBetweenUniqueConceptPair):
		for connectionKey, connection in hopfieldGraphNode.HFcontextTargetConnectionDict.items():
			drawHopfieldGraphConnection(connection, drawGraphNetwork, sentenceConceptNodeList)
		for connectionKey, connection in hopfieldGraphNode.HFcausalTargetConnectionDict.items():
			drawHopfieldGraphConnection(connection, drawGraphNetwork, sentenceConceptNodeList)
	else:
		for connectionKey, connectionList in hopfieldGraphNode.HFcontextTargetConnectionDict.items():
			for connection in connectionList:
				drawHopfieldGraphConnection(connection, drawGraphNetwork, sentenceConceptNodeList)
		for connectionKey, connectionList in hopfieldGraphNode.HFcausalTargetConnectionDict.items():
			for connection in connectionList:
				drawHopfieldGraphConnection(connection, drawGraphNetwork, sentenceConceptNodeList)
				
#def drawHopfieldGraphNodeAndConnections(hopfieldGraphNode, networkSize, drawGraphNetwork, sentenceConceptNodeList=None):	
#	#parse tree and generate nodes and connections
#	drawHopfieldGraphNode(hopfieldGraphNode, networkSize)
#	drawHopfieldGraphNodeConnections(hopfieldGraphNode, drawGraphNetwork, sentenceConceptNodeList)
	
def drawHopfieldGraphNode(node, networkSize):
	hopfieldGraphAngle = node.networkIndex/networkSize*360
	#print("hopfieldGraphAngle = ", hopfieldGraphAngle)
	posX, posY = pointOnCircle(hopfieldGraphRadius, hopfieldGraphAngle, hopfieldGraphCentre)	#generate circular graph
	hopfieldGraph.add_node(node.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		if(useAlgorithmScanBiologicalSimulation):
			#~use SANIgenDemoVideo1 simulation colour scheme (FUTURE: vary colours based on node.activationLevel)
			if(node.activationStateFiltered):
				colorHtml = 'yellow'
			elif(node.activationState):
				colorHtml = 'orange'
			else:
				colorHtml = 'blue'
		else:
			printe("drawHopfieldGraphNodeColours currently requires useAlgorithmScanBiologicalSimulation")
		hopfieldGraphNodeColorMap.append(colorHtml)

def drawHopfieldGraphConnection(connection, drawGraphNetwork, sentenceConceptNodeList=None):
	node1 = connection.nodeSource
	node2 = connection.nodeTarget
	#print("drawHopfieldGraphConnection: node1 = ", node1.nodeName, ", node2 = ", node2.nodeName)
	#spatioTemporalIndex = connection.spatioTemporalIndex
	if(drawGraphNetwork or (node2 in sentenceConceptNodeList)):	#if HFNLPpy_hopfieldGraphDraw: ensure target node is in sentence (such that connection can be drawn) - see drawHopfieldGraphNodeConnections
		if(drawHopfieldGraphEdgeColours):
			if(connection.contextConnection):
				color = 'blue'
			else:
				color = 'red'
		else:
			color = 'black'
		if(drawHopfieldGraphEdgeWeights):
			weight = connection.weight
		else:
			weight = 1.0
		if(drawHopfieldGraphEdgeColoursWeights):
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
		if(drawHopfieldGraphNodeColours):
			nx.draw(hopfieldGraph, pos, with_labels=True, alpha=graphTransparency, node_color=hopfieldGraphNodeColorMap, edge_color=colors, width=list(weights))
		else:
			nx.draw(hopfieldGraph, pos, with_labels=True, alpha=graphTransparency, edge_color=colors, width=list(weights))
	else:
		if(drawHopfieldGraphNodeColours):
			nx.draw(hopfieldGraph, pos, with_labels=True, alpha=graphTransparency, node_color=hopfieldGraphNodeColorMap)
		else:
			nx.draw(hopfieldGraph, pos, with_labels=True, alpha=graphTransparency)

	if(save):
		if(highResolutionFigure):
			plt.savefig(fileName, dpi=saveFigDPI)
		else:
			plt.savefig(fileName)
	if(plot):
		plt.show()
	else:	
		plt.clf()

def pointOnCircle(radius, angleDegrees, centre=[0,0]):
	angle = radians(angleDegrees)
	x = centre[0] + (radius * cos(angle))
	y = centre[1] + (radius * sin(angle))
	return x, y

def generateHopfieldGraphFileName(sentenceOrNetwork, sentenceIndex=None):
	fileName = "hopfieldGraph"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + str(sentenceIndex)
	return fileName
