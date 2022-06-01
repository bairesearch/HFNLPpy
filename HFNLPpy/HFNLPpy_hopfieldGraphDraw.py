"""HFNLPpy_hopfieldGraphDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
ATNLP Hopfield Graph Draw Class

"""

import networkx as nx
import matplotlib.pyplot as plt
from math import cos, sin, radians
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *

drawHopfieldGraphEdgeStrengths = False	#TODO (incomplete)
drawHopfieldGraphNodeColours = False	#node colours yet coded (pos type of node will be different depending on connectivity/instance context)
if(not drawHopfieldGraphEdgeStrengths):
	graphTransparency = 0.5

hopfieldGraph = nx.MultiDiGraph()	#Directed graphs with self loops and parallel edges	#https://networkx.org/documentation/stable/reference/classes/multidigraph.html
hopfieldGraphNodeColorMap = []
hopfieldGraphRadius = 100
hopfieldGraphCentre = [0, 0]

def setColourHopfieldNodes(value):
    global drawHopfieldGraphNodeColours
    drawHopfieldGraphNodeColours = value

def clearHopfieldGraph():
	hopfieldGraph.clear()	#only draw graph for single sentence
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.clear()

def drawHopfieldGraphNode(node, networkSize, sentenceIndex=0):
	colorHtml = "NA"	#node colours yet coded (pos type of node will be different depending on connectivity/instance context)
	hopfieldGraphAngle = node.networkIndex/networkSize*360
	#print("hopfieldGraphAngle = ", hopfieldGraphAngle)
	posX, posY = pointOnCircle(hopfieldGraphRadius, hopfieldGraphAngle, hopfieldGraphCentre)	#generate circular graph
	hopfieldGraph.add_node(node.nodeName, pos=(posX, posY))
	if(drawHopfieldGraphNodeColours):
		hopfieldGraphNodeColorMap.append(colorHtml)

def drawHopfieldGraphConnection(node1, node2, spatioTemporalIndex):
	hopfieldGraph.add_edge(node1.nodeName, node2.nodeName)	#FUTURE: consider setting color based on spatioTemporalIndex

def displayHopfieldGraph():
	pos = nx.get_node_attributes(hopfieldGraph, 'pos')
	if(drawHopfieldGraphEdgeStrengths):
		#TODO;
		widths = nx.get_edge_attributes(hopfieldGraph, 'weight')
		nodelist = hopfieldGraph.nodes()
		plt.figure(figsize=(10,10))
		#pos = nx.shell_layout(hopfieldGraph)
		nx.draw_networkx_nodes(hopfieldGraph, pos, nodelist=nodelist, node_size=0,  node_color=0,  alpha=0)
		nx.draw_networkx_edges(hopfieldGraph, pos, edgelist = widths.keys(), width=list(widths.values()), edge_color=0, alpha=0)
		nx.draw_networkx_labels(hopfieldGraph, pos=pos, labels=dict(zip(nodelist,nodelist)), font_color=0)
		plt.box(False)	
	else:
		if(drawHopfieldGraphNodeColours):
			nx.draw(hopfieldGraph, pos, node_color=hopfieldGraphNodeColorMap, with_labels=True, alpha=graphTransparency)	#nx.draw_networkx	
		else:
			nx.draw(hopfieldGraph, pos, with_labels=True, alpha=graphTransparency)
	plt.show()

def drawHopfieldGraphNodeAndConnections(hopfieldGraphNode, networkSize, drawGraph=False):	
	#parse tree and generate nodes and connections
	drawHopfieldGraphNode(hopfieldGraphNode, networkSize)
	#for connection in hopfieldGraphNode.targetConnectionList:
	for connectionKey, connectionList in hopfieldGraphNode.targetConnectionDict.items():
		for connection in connectionList:
			targetNode = connection.nodeTarget
			spatioTemporalIndex = connection.spatioTemporalIndex
			drawHopfieldGraphConnection(hopfieldGraphNode, targetNode, spatioTemporalIndex)

def drawHopfieldGraphNetwork(networkConceptNodeDict):	
	#generate nodes and connections
	networkSize = len(networkConceptNodeDict)
	for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
	#for conceptNode in conceptNodeList:
		#print("conceptNode.lemma = ", conceptNode.lemma)
		drawHopfieldGraphNodeAndConnections(conceptNode, networkSize, drawGraph=True)

def pointOnCircle(radius, angleDegrees, centre=[0,0]):
	angle = radians(angleDegrees)
	x = centre[0] + (radius * cos(angle))
	y = centre[1] + (radius * sin(angle))
	return x, y
	
