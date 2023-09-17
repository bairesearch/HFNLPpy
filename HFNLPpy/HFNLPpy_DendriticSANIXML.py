"""HFNLPpy_DendriticSANIXML.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI XML - read/write graph XML file

"""

from yattag import Doc, indent	#pythonic xml api
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_DendriticSANIGlobalDefs import *
from HFNLPpy_DendriticSANINode import *

def writeDentriticSANIGraphSentence(sentenceConceptNodeList, fileName, activationTime=None):
	drawGraphNetwork = False
	doc, tag, text, line = Doc().ttl()
	#doc, tag, text = Doc().tagtext()
	
	with tag('HFNLPgraphSentence'):
		with tag('conceptNodes'):
			for conceptNode in sentenceConceptNodeList:
				writeHopfieldGraphNode(doc, tag, text, line, conceptNode, drawGraphNetwork, activationTime, sentenceConceptNodeList)
				
	string = indent(doc.getvalue(), indentation = '\t', newline = '\n')
	#string = doc.getvalue()
	fileName = fileName + '.xml'
	writeStringToFile(fileName, string)
			
def writeDentriticSANIGraphNetwork(networkConceptNodeDict, fileName, activationTime=None):
	drawGraphNetwork = True
	doc, tag, text, line = Doc().ttl()
	#doc, tag, text = Doc().tagtext()

	with tag('HFNLPgraphNetwork'):
		with tag('conceptNodes'):
			for conceptNodeKey, conceptNode in networkConceptNodeDict.items():
				writeHopfieldGraphNode(doc, tag, text, line, conceptNode, drawGraphNetwork, activationTime)

	string = indent(doc.getvalue(), indentation = '\t', newline = '\n')
	#string = doc.getvalue()
	fileName = fileName + '.xml'
	writeStringToFile(fileName, string)
	
def writeStringToFile(fileName, string):
	#print(string)
	text_file = open(fileName, "w")
	n = text_file.write(string)
	text_file.close()

def writeHopfieldGraphNode(doc, tag, text, line, conceptNode, drawGraphNetwork, activationTime, sentenceConceptNodeList=None):
	activationState = getActivationState(conceptNode)
	with tag('conceptNode', name=conceptNode.nodeName, activationState=activationState):
		with tag('connections'):
			writeHopfieldGraphNodeConnections(doc, tag, text, line, conceptNode, drawGraphNetwork, activationTime, sentenceConceptNodeList)
		with tag('dendriticTree'):
			currentBranchIndex1 = 0
			currentBranchIndex2 = 0
			writeHopfieldGraphNodeDendriticBranch(doc, tag, text, line, conceptNode, conceptNode.dendriticTree, currentBranchIndex1, currentBranchIndex2, activationTime)

def writeHopfieldGraphNodeDendriticBranch(doc, tag, text, line, conceptNode, dendriticBranch, currentBranchIndex1, currentBranchIndex2, activationTime):
	#print("writeHopfieldGraphNodeDendriticBranch: , dendriticBranch.nodeName = ", dendriticBranch.nodeName, ", currentBranchIndex1 = ", currentBranchIndex1)
	activationState = getActivationState(dendriticBranch)
	with tag('branch', branchIndex1=currentBranchIndex1, branchIndex2=currentBranchIndex2, horizontalBranchIndex=dendriticBranch.horizontalBranchIndex, activationState=activationState):	#with tag('branch', branchIndex1=currentBranchIndex1, activationState=activationState):
		with tag('sequentialSegments'):
			for currentSequentialSegmentIndex, currentSequentialSegment in enumerate(dendriticBranch.sequentialSegments):
				writeHopfieldGraphNodeSequentialSegment(doc, tag, text, line, currentBranchIndex1, conceptNode, currentSequentialSegment, currentSequentialSegmentIndex, activationTime)
		with tag('subbranches'):	#redundant (could be removed)
			for currentBranchIndex2, subbranch in enumerate(dendriticBranch.subbranches):
				#with tag('branchIndex2', id=currentBranchIndex2):	#redundant (could be removed)
				writeHopfieldGraphNodeDendriticBranch(doc, tag, text, line, conceptNode, subbranch, currentBranchIndex1+1, currentBranchIndex2, activationTime)

def writeHopfieldGraphNodeSequentialSegment(doc, tag, text, line, currentBranchIndex1, conceptNode, sequentialSegment, currentSequentialSegmentIndex, activationTime):
	activationState = getActivationState(sequentialSegment)
	#with tag('sequentialSegment', sequentialSegmentIndex=currentSequentialSegmentIndex, activationState=activationState):	#default
	with tag('sequentialSegment', sequentialSegmentIndex=currentSequentialSegmentIndex, branchIndex1=currentBranchIndex1, branchIndex2=sequentialSegment.branch.branchIndex2, horizontalBranchIndex=sequentialSegment.branch.horizontalBranchIndex, activationState=activationState):	#additional attributes displayed for debug only
		for currentSequentialSegmentInputIndexDynamic, currentSequentialSegmentInput in enumerate(sequentialSegment.inputs.values()):	#note currentSequentialSegmentInputIndexDynamic is valid even if inputs have been removed from dictionary (although order not guaranteed)
			if(storeSequentialSegmentInputIndexValues):
				currentSequentialSegmentInputIndex = currentSequentialSegmentInput.sequentialSegmentInputIndex
			else:
				currentSequentialSegmentInputIndex = currentSequentialSegmentInputIndexDynamic	
			currentSequentialSegmentInputText = name=currentSequentialSegmentInput.nodeName 	#name attribute is added for axon connectivity lookup
			with tag('sequentialSegmentInput', name=currentSequentialSegmentInputText, sequentialSegmentInputIndex=currentSequentialSegmentInputIndex, activationState=activationState):
				writeHopfieldGraphNodeSequentialSegmentInput(doc, tag, text, line, conceptNode, currentSequentialSegmentInput, currentSequentialSegmentInputIndex, activationTime)
	
def writeHopfieldGraphNodeSequentialSegmentInput(doc, tag, text, line, conceptNode, sequentialSegmentInput, currentSequentialSegmentInputIndex, activationTime):
	activationState = getActivationState(sequentialSegmentInput)
	#NA

def writeHopfieldGraphNodeConnections(doc, tag, text, line, hopfieldGraphNode, drawGraphNetwork, activationTime, sentenceConceptNodeList=None):
	for connectionKey, connectionList in hopfieldGraphNode.HFcontextTargetConnectionMultiDict.items():
		with tag('connectionList', key=connectionKey):
			for connectionIndex, connection in enumerate(connectionList):
				writeHopfieldGraphConnection(doc, tag, text, line, connection, connectionIndex, drawGraphNetwork, activationTime, sentenceConceptNodeList)

def writeHopfieldGraphConnection(doc, tag, text, line, connection, connectionIndex, drawGraphNetwork, activationTime, sentenceConceptNodeList=None):
	activationState = getActivationState(connection)
	node1 = connection.nodeSource
	node2 = connection.nodeTargetSequentialSegmentInput
	nodeTargetSequentialSegmentInputText = node2.nodeName	#name attribute is added for axon connectivity lookup
	if(drawGraphNetwork or (node2.conceptNode in sentenceConceptNodeList)):	#if HFNLPpy_DendriticSANIDrawSentence: ensure target node is in sentence (such that connection can be drawn) - see drawHopfieldGraphNodeConnections
		with tag('connection', connectionIndex=connectionIndex, nodeTargetSequentialSegmentInput=nodeTargetSequentialSegmentInputText, activationState=activationState):	
			pass
			#doc.stag('connection', nodeTargetSequentialSegmentInput=nodeTargetSequentialSegmentInputText, activationState=activationState) 	#inline tag
			#line('connection', 'Salt')
		

def getActivationState(neuronObject):
	if(writeBiologicalSimulationActivationStates):
		if(neuronObject.activationLevel):
			activationState = 'True'
		else:
			activationState = 'False'
	else:
		activationState = ''
	return activationState
	
						
