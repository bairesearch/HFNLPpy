"""HFNLPpy_LayeredSANIDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Layered SANI Draw - draw sentence/network SANI graph

"""

import networkx as nx
import matplotlib.pyplot as plt
plt.ioff()	# Turn interactive plotting off
from math import cos, sin, radians
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_globalDefs import drawSaveSvg
from HFNLPpy_LayeredSANIGlobalDefs import *

highResolutionFigure = True
if(highResolutionFigure):
	saveFigDPI = 300	#approx HD	#depth per inch
	saveFigSize = (16,9)	#in inches
	
drawLayeredSANIGraphEdgeColours = True
drawLayeredSANIGraphEdgeWeights = False
if(drawLayeredSANIGraphEdgeColours or drawLayeredSANIGraphEdgeWeights):
	drawLayeredSANIGraphEdgeColoursWeights = True
drawLayeredSANIGraphNodeColours = True	#node colours not yet coded (pos type of SANI node will be different depending on connectivity/instance context)
graphTransparency = 0.5

layeredSANIGraph = nx.Graph()	#MultiDiGraph: Directed graphs with self loops and parallel edges	#https://networkx.org/documentation/stable/reference/classes/multidigraph.html
layeredSANIGraphNodeColorMap = []
layeredSANIGraphNodeSizeMap = []
layeredSANIGraphSANINodesList = []	#primary nodes for label assignment

#require calibration (depends on numberOfLayers):
#numberOfLayersMax = 10
SANINeuronIndexSeparation = 10.0
layerSeparation = 10.0

spineSeparation = 0.2
nodeSize = 0.5	#node diameter
nodeSizeDraw = 10.0	#node diameter
SANINodeSizeDrawNetwork = 100.0	#node diameter
SANINodeSizeDrawSentence = 1000.0	#node diameter


def drawLayeredSANIStatic(SANIlayerList, sentenceIndex):
	#print("drawLayeredSANIStatic")
	if(drawBiologicalSimulation):
		if(drawBiologicalSimulationSentence):
			fileName = generateLayeredSANIFileName(True, sentenceIndex, write=False)
			clearLayeredSANIGraph()
			drawLayeredSANIGraphSentence(SANIlayerList)
			print("drawBiologicalSimulationSentence: HFNLPpy_LayeredSANIDraw.displayLayeredSANIGraph()")
			displayLayeredSANIGraph(drawBiologicalSimulationPlot, drawBiologicalSimulationSave, fileName)
		if(drawBiologicalSimulationNetwork):
			fileName = generateLayeredSANIFileName(False, sentenceIndex, write=False)
			clearLayeredSANIGraph()
			drawLayeredSANIGraphNetwork(SANIlayerList)
			print("drawBiologicalSimulationNetwork: HFNLPpy_LayeredSANIDraw.displayLayeredSANIGraph()")
			displayLayeredSANIGraph(drawBiologicalSimulationPlot, drawBiologicalSimulationSave, fileName)	

'''
def drawLayeredSANIDynamicNeuronActivation(wSource, SANIlayerList, sentenceIndex, activationTime, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivation or (sentenceIndex == sentenceIndexDebug and wSource >= wSourceDebug)):	#wSource == wSourceDebug
			if(drawBiologicalSimulationSentenceDynamic):
				fileName = generateLayeredSANIDynamicNeuronFileName(True, wSource, sentenceIndex)
				clearLayeredSANIGraph()
				drawLayeredSANIGraphSentence(SANIlayerList, activationTime=activationTime, wTarget=wTargetDebug)
				displayLayeredSANIGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationNetworkDynamic):
				fileName = generateLayeredSANIDynamicNeuronFileName(False, wSource, sentenceIndex)
				clearLayeredSANIGraph()
				drawLayeredSANIGraphNetwork(SANIlayerList, activationTime=activationTime, wTarget=wTargetDebug)
				displayLayeredSANIGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)	
'''

def clearLayeredSANIGraph():
	layeredSANIGraph.clear()	#only draw graph for single sentence
	if(drawLayeredSANIGraphNodeColours):
		layeredSANIGraphNodeColorMap.clear()
		layeredSANIGraphNodeSizeMap.clear()
	layeredSANIGraphSANINodesList.clear()	#for labels

def drawLayeredSANIGraphSentence(SANIlayerList):	
	#print("drawLayeredSANIGraphSentence = ")
	#print("size layeredSANIGraph.nodes = ", len(layeredSANIGraph.nodes))
	#print("size layeredSANIGraphNodeColorMap = ", len(layeredSANIGraphNodeColorMap))
	drawGraphNetwork = False
	#need to draw all SANINodes and their dendriticTrees before creating connections
	for layerIndex, layer in enumerate(SANIlayerList): 
		for nodeIndex, SANINode in enumerate(layer.sentenceSANINodeList):
			#print("drawLayeredSANIGraphSentence:drawLayeredSANIGraphNode, layerIndex = ", layerIndex)
			drawLayeredSANIGraphNode(layerIndex, nodeIndex, SANINode, drawGraphNetwork)
	for layerIndex, layer in enumerate(SANIlayerList): 
		for nodeIndex, SANINode in enumerate(layer.sentenceSANINodeList):
			drawLayeredSANIGraphNodeConnections(layerIndex, nodeIndex, SANINode, drawGraphNetwork, layer.sentenceSANINodeList)

def drawLayeredSANIGraphNetwork(SANIlayerList, activationTime=None, wTarget=None):	
	#print("drawLayeredSANIGraphNetwork = ")
	#print("size layeredSANIGraph.nodes = ", len(layeredSANIGraph.nodes))
	#print("size layeredSANIGraphNodeColorMap = ", len(layeredSANIGraphNodeColorMap))
	#generate nodes and connections
	drawGraphNetwork = True
	for layerIndex, layer in enumerate(SANIlayerList): 
		for nodeIndex, SANINode in enumerate(layer.networkSANINodeList):
			drawLayeredSANIGraphNode(layerIndex, nodeIndex, SANINode, drawGraphNetwork)
	for layerIndex, layer in enumerate(SANIlayerList): 
		for nodeIndex, SANINode in enumerate(layer.networkSANINodeList):
			drawLayeredSANIGraphNodeConnections(layerIndex, nodeIndex, SANINode, drawGraphNetwork)

def drawLayeredSANIGraphNodeConnections(layerIndex, nodeIndex, layeredSANIGraphNode, drawGraphNetwork, sentenceSANINodeList=None):
	for connectionKey, connection in layeredSANIGraphNode.HFcontextTargetConnectionLayeredDict.items():
		if(not LayeredSANIHierarchicalGraphOnlyShowParentChildLinks):
			drawHFGraphConnection(layerIndex, connection, drawGraphNetwork, sentenceSANINodeList)
		if(connection.SANInodeAssigned):
			if(drawGraphNetwork or connection.SANIactivationState):
				if(drawGraphNetwork or ((connection.nodeSource in sentenceSANINodeList) and (connection.nodeTarget in sentenceSANINodeList))):
					drawLayeredSANIGraphConnection(connection, connection.nodeSource, connection.SANInode, drawGraphNetwork, sentenceSANINodeList)
					drawLayeredSANIGraphConnection(connection, connection.nodeTarget, connection.SANInode, drawGraphNetwork, sentenceSANINodeList)
	for connectionKey, connection in layeredSANIGraphNode.HFcausalTargetConnectionLayeredDict.items():
		if(not LayeredSANIHierarchicalGraphOnlyShowParentChildLinks):
			drawHFGraphConnection(layerIndex, connection, drawGraphNetwork, sentenceSANINodeList)
	#for outputNode in layeredSANIGraphNode.SANIoutputNodeList:
	#	drawLayeredSANIGraphConnection(layeredSANIGraphNode, outputNode, drawGraphNetwork, sentenceSANINodeList)
			
#def drawLayeredSANIGraphNodeAndConnections(layeredSANIGraphNode, drawGraphNetwork, sentenceSANINodeList=None):	
#	#parse tree and generate nodes and connections
#	drawLayeredSANIGraphNode(layeredSANIGraphNode, drawGraphNetwork)
#	drawLayeredSANIGraphNodeConnections(layeredSANIGraphNode, drawGraphNetwork, sentenceSANINodeList)
		
def drawLayeredSANIGraphNode(layerIndex, nodeIndex, SANINode, drawGraphNetwork):
	#print("layerIndex = ", layerIndex)
	#print("drawLayeredSANIGraphNode: SANINode.nodeName = ", SANINode.nodeName)
	colorHtml = getActivationColor(SANINode, 'yellow', 'orange', 'blue', highlightPartialActivations=True)

	posX = nodeIndex*SANINeuronIndexSeparation
	posY = layerIndex*layerSeparation
	
	#print("drawLayeredSANIGraphNode: ", SANINode.nodeName)
	layeredSANIGraph.add_node(SANINode.nodeName, pos=(posX, posY))
	if(drawLayeredSANIGraphNodeColours):
		layeredSANIGraphNodeColorMap.append(colorHtml)
		if(drawGraphNetwork):
			layeredSANIGraphNodeSizeMap.append(SANINodeSizeDrawNetwork)
		else:
			layeredSANIGraphNodeSizeMap.append(SANINodeSizeDrawSentence)
	layeredSANIGraphSANINodesList.append(SANINode.nodeName)


def drawHFGraphConnection(layerIndex, connection, drawGraphNetwork, sentenceSANINodeList=None):
	node1 = connection.nodeSource
	node2 = connection.nodeTarget
	if(drawGraphNetwork or (node2 in sentenceSANINodeList)):	#if HFNLPpy_DendriticSANIDrawSentence: ensure target node is in sentence (such that connection can be drawn) - see drawLayeredSANIGraphNodeConnections
		if(drawLayeredSANIGraphEdgeColours):
			if(connection.contextConnection):
				color = getActivationColor(connection, 'gold', 'orange', 'darkorange', highlightPartialActivations=True)
			else:
				color = getActivationColor(connection, 'lightred', 'red', 'darkred', highlightPartialActivations=True)
		else:
			color = 'black'
		if(drawLayeredSANIGraphEdgeWeights):
			weight = connection.weight
		else:
			weight = 1.0
		if(drawLayeredSANIGraphEdgeColoursWeights):
			layeredSANIGraph.add_edge(node1.nodeName, node2.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
		else:
			layeredSANIGraph.add_edge(node1.nodeName, node2.nodeName)

def drawLayeredSANIGraphConnection(connection, inputNode, outputNode, drawGraphNetwork, sentenceSANINodeList=None):
	#if(drawGraphNetwork or (outputNode in sentenceSANINodeList)):	#if HFNLPpy_DendriticSANIDrawSentence: ensure target node is in sentence (such that connection can be drawn) - see drawLayeredSANIGraphNodeConnections
	if(drawLayeredSANIGraphEdgeColours):
		color = getActivationColor(outputNode, 'lightblue', 'blue', 'darkblue', highlightPartialActivations=False)
	else:
		color = 'black'
	if(drawLayeredSANIGraphEdgeWeights):
		weight = connection.weight
	else:
		weight = 1.0
	if(drawLayeredSANIGraphEdgeColoursWeights):
		layeredSANIGraph.add_edge(inputNode.nodeName, outputNode.nodeName, color=color, weight=weight)	#FUTURE: consider setting color based on spatioTemporalIndex
	else:
		layeredSANIGraph.add_edge(inputNode.nodeName, outputNode.nodeName)
			

def displayLayeredSANIGraph(plot=True, save=False, fileName=None):
	pos = nx.get_node_attributes(layeredSANIGraph, 'pos')
	#print("pos = ", pos)
	
	if(highResolutionFigure):
		plt.figure(1, figsize=saveFigSize) 

	if(drawLayeredSANIGraphEdgeColoursWeights):
		edges = layeredSANIGraph.edges()
		#colors = [layeredSANIGraph[u][v]['color'] for u,v in edges]
		#weights = [layeredSANIGraph[u][v]['weight'] for u,v in edges]	
		colors = nx.get_edge_attributes(layeredSANIGraph,'color').values()
		weights = nx.get_edge_attributes(layeredSANIGraph,'weight').values()
		#print("size layeredSANIGraph.nodes = ", len(layeredSANIGraph.nodes))
		#print("size layeredSANIGraphNodeColorMap = ", len(layeredSANIGraphNodeColorMap))
		if(drawLayeredSANIGraphNodeColours):
			nx.draw(layeredSANIGraph, pos, with_labels=False, alpha=graphTransparency, node_color=layeredSANIGraphNodeColorMap, edge_color=colors, width=list(weights), node_size=layeredSANIGraphNodeSizeMap)
		else:
			nx.draw(layeredSANIGraph, pos, with_labels=False, alpha=graphTransparency, edge_color=colors, width=list(weights), node_size=nodeSizeDraw)
	else:
		if(drawLayeredSANIGraphNodeColours):
			nx.draw(layeredSANIGraph, pos, with_labels=False, alpha=graphTransparency, node_color=layeredSANIGraphNodeColorMap, node_size=layeredSANIGraphNodeSizeMap)
		else:
			nx.draw(layeredSANIGraph, pos, with_labels=False, alpha=graphTransparency, node_size=nodeSizeDraw)

	#if(useAlgorithmDendriticSANI) exclusive code:
	#only assign labels to SANINeurons
	labels = {}    
	for node in layeredSANIGraph.nodes():
		if node in layeredSANIGraphSANINodesList:
			#set the node name as the key and the label as its value 
			labels[node] = node
	nx.draw_networkx_labels(layeredSANIGraph, pos, labels, font_size=8)	#font_size=16, font_color='r'
	
	if(save):
		fileExtension = ".svg" if drawSaveSvg else ".png"
		saveFormat = 'svg' if drawSaveSvg else 'png'
		fileName = fileName + fileExtension
		if(highResolutionFigure):
			plt.savefig(fileName, dpi=saveFigDPI, format=saveFormat)
		else:
			plt.savefig(fileName, format=saveFormat)
	if(plot):
		plt.show()
	else:
		plt.clf()

	

def getActivationColor(neuronObject, colorActiveNew, colorActive, colorInactive, highlightPartialActivations=False):
	#~use SANIgenDemoVideo1 simulation colour scheme (FUTURE: vary colours based on node.activationLevel)
	if(highlightPartialActivations and neuronObject.activationStatePartial):
		color = colorActive
	elif(neuronObject.SANIactivationState):
		color = colorActive
	else:
		color = colorInactive
	return color
	
	

def generateLayeredSANIFileName(sentenceOrNetwork, sentenceIndex=None, write=False):
	fileName = "useAlgorithmDendriticSANI"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	if(outputFileNameComputationType):
		if(vectoriseComputation):
			fileName = fileName + "VectorisedComputation"
		else:
			fileName = fileName + "StandardComputation"
	return fileName
	
def generateLayeredSANIDynamicNeuronFileName(sentenceOrNetwork, wSource, sentenceIndex=None):
	fileName = "biologicalSimulationDynamic"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + convertIntToString(sentenceIndex)
	fileName = fileName + "Wsource" + convertIntToString(wSource)
	if(outputFileNameComputationType):
		if(vectoriseComputation):
			fileName = fileName + "VectorisedComputation"
		else:
			fileName = fileName + "StandardComputation"
	return fileName
	
def convertIntToString(integer, zeroPadLength=3):
	string = str(integer).zfill(zeroPadLength)
	return string
		
