"""HFNLPpy_ConnectionMatrixOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Connection Matrix Operations

"""

from HFNLPpy_globalDefs import *	
from HFNLPpy_MatrixGlobalDefs import *
import csv
import os

if(pt.cuda.is_available()):
	device = pt.device("cuda")
else:
	device = pt.device("cpu")

def addSentenceConceptNodesToHFconnectionGraphObject(HFconnectionGraphObject, sentenceConceptNodeList):
	if(useHFconnectionMatrix):
		for conceptNodeIndex, conceptNode in enumerate(sentenceConceptNodeList):
			addSentenceConceptNodeToHFconnectionGraphObject(HFconnectionGraphObject, conceptNode)
			
def addSentenceConceptNodeToHFconnectionGraphObject(HFconnectionGraphObject, conceptNode):
	#print("conceptNodeIndex = ", conceptNodeIndex)
	if(not conceptNode.nodeName in HFconnectionGraphObject.neuronIDdict):
		HFconnectionGraphObject.neuronNamelist.append(conceptNode.nodeName)
		neuronID = conceptNode.networkIndex
		HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName] = neuronID
		
def getNetworkSize(HFconnectionGraphObject):
	#networkSize = stored databaseSize (RAM HFconnectionNetwork size is slightly larger to accomodate for new concepts added in current sentence)
	networkSize = len(HFconnectionGraphObject.neuronNamelist)	#or len(HFconnectionGraphObject.neuronIDdict)
	return networkSize

def setConnectionMatrixMaxConcepts(HFconnectionGraphObject):
	#need to set connectionMatrixMaxConcepts to be sufficiently high such that all new sentence nodes can be added to array
	networkSize = getNetworkSize(HFconnectionGraphObject)
	print("networkSize = ", networkSize)
	HFconnectionGraphObject.connectionMatrixMaxConcepts = networkSize + HFconnectionMatrixBasicMaxConceptsInArticle
	print("HFconnectionGraphObject.connectionMatrixMaxConcepts = ", HFconnectionGraphObject.connectionMatrixMaxConcepts)

def initialiseNeuronNameList(HFconnectionGraphObject):
	if(HFconnectionGraphObject.neuronNamelist is None):	#ensure neuronNamelist has not already been read (eg via another HFNLPpy_ConnectionMatrix implementation)
		conceptNeuronListFileExists = False
		HFconceptNeuronListPathName = generateConceptListFileName()
		if(os.path.exists(HFconceptNeuronListPathName)):
			conceptNeuronListFileExists = True
		if(HFreadSavedConceptList and conceptNeuronListFileExists):
			print("found existing neuronNamelist, loading concepts")
			HFconnectionGraphObject.neuronNamelist = readConceptNeuronList(HFconceptNeuronListPathName)
		else:
			HFconnectionGraphObject.neuronNamelist = []
		createNeuronIDdictFromNameList(HFconnectionGraphObject)
		setConnectionMatrixMaxConcepts(HFconnectionGraphObject)

def createNeuronIDdictFromNameList(HFconnectionGraphObject):
	for neuronID, neuronName in enumerate(HFconnectionGraphObject.neuronNamelist):
		HFconnectionGraphObject.neuronIDdict[neuronName] = neuronID
		
def readConceptNeuronList(filePath):
	names = []
	try:
		with open(filePath, 'r') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if row:
					names.append(row[0])
	except FileNotFoundError:
		print("File not found.")
	return names

def writeConceptNeuronList(names, filePath):
	try:
		with open(filePath, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			for name in names:
				writer.writerow([name])
		print("Names written to file successfully.")
	except Exception as e:
		print("Error:", e)
		
def writeHFConceptList(neuronNamelist):
	HFconceptNeuronListPathName = generateConceptListFileName()
	writeConceptNeuronList(neuronNamelist, HFconceptNeuronListPathName)
	
def generateConceptListFileName():
	if(HFconnectionMatrixAlgorithmSplitDatabase):
		HFconceptNeuronListPathName = matrixDatabasePathName + "/" + HFconceptNeuronsBasicFileName + HFconceptNeuronsBasicExtensionName
	else:
		HFconceptNeuronListPathName = datasetFolderRelative + "/" + HFconceptNeuronsBasicFileName + HFconceptNeuronsBasicExtensionName
	return HFconceptNeuronListPathName

def readGraphFromCsv(filePath):
	connections = []
	with open(filePath, 'r') as f:
		reader = csv.reader(f)
		for row in (reader):
			connections.append(row)
	connections = [[int(value) for value in row] for row in connections]
	graph = pt.tensor(connections, dtype=HFconnectionsMatrixAlgorithmType)
	return graph

def writeGraphToCsv(graph, filePath):
	graph = graph.cpu()		
	connections = graph.numpy()
	with open(filePath, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(connections)

