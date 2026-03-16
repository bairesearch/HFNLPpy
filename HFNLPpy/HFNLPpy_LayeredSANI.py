"""HFNLPpy_LayeredSANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Layered SANI - simulate training/inference of biological hybrid SANI-hopfield graph/network based on textual input

training;
	activate concept neurons in order of sentence word order
	detect association between concept neurons
	assign a higher level SANI node for highly associated concept neurons

nodeName = neuronID
"""


import numpy as np
import torch as pt

from HFNLPpy_LayeredSANIGlobalDefs import *
import HFNLPpy_hopfieldOperations
import HFNLPpy_LayeredSANIDraw
from HFNLPpy_hopfieldNodeClass import *
if(LayeredSANIwordVectors):
	from HFNLPpy_MatrixGlobalDefs import *
	import HFNLPpy_Matrix
	if(useHFconnectionMatrixBasic):
		import HFNLPpy_ConceptsMatrixOperations
		import HFNLPpy_ConnectionMatrixBasic
	elif(useHFconnectionMatrixAlgorithm):
		import HFNLPpy_MatrixOperations
		
if(drawBiologicalSimulation):
	import HFNLPpy_LayeredSANIDraw

def updateLayeredSANIgraph(HFconnectionGraphObject, networkConceptNodeDict, SANIlayerList, sentenceIndex):
	for layerIndex, layer in enumerate(SANIlayerList):
		if(layerIndex < LayeredSANInumberOfLayersMax):	#ensure that higher layer SANI nodes can still be added
			#print("layerIndex = ", layerIndex)
			if(layerIndex < LayeredSANInumberOfLayersMax-1):
				if(vectoriseComputation):
					updateLayeredSANIlayerVectorised(HFconnectionGraphObject, networkConceptNodeDict, SANIlayerList, layerIndex, layer, generateSANINetwork=True)
				else:
					updateLayeredSANIlayerStandard(HFconnectionGraphObject, networkConceptNodeDict, SANIlayerList, layerIndex, layer, generateSANINetwork=True)
			else:
				layerBufferSANINodeListSorted = sortSANInodeList(layer.sentenceSANINodeList)
				if(enableBasicNextWordCausalPredictions):
					recordNextWordCausalPredictions(layerBufferSANINodeListSorted)
			
	if(drawBiologicalSimulation):
		HFNLPpy_LayeredSANIDraw.drawLayeredSANIStatic(SANIlayerList, sentenceIndex)

	clearNetworkActivations(SANIlayerList)
	
	return layerBufferSANINodeListSorted

def sortSANInodeList(sentenceSANINodeList):
	layerBufferSANINodeListSorted = sentenceSANINodeList
	layerBufferSANINodeListSorted.sort(key=lambda x: x.wMin)
	return layerBufferSANINodeListSorted
	
def recordNextWordCausalPredictions(layerBufferSANINodeListSorted):
	SANINeuronPrev = None
	for x1, SANINeuron in enumerate(layerBufferSANINodeListSorted):
		if(x1 > 0):
			if(not SANINeuron.noncontinguousInputNodeAboveCentralContentsStart):
				if(not connectionExists(SANINeuronPrev, SANINeuron, False)):
					HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuronPrev, SANINeuron, contextConnection=False, useAlgorithmLayeredSANI=True)
				connection = SANINeuronPrev.HFcausalTargetConnectionLayeredDict[SANINeuron.SANIlayerNeuronID]
				connection.weight += 1
			if(SANINeuron.noncontinguousInputNodeAboveCentralContentsEnd):
				if(not connectionExists(SANINeuron, SANINeuron.noncontinguousInputNodeAbove, False)):
					HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuron, SANINeuron.noncontinguousInputNodeAbove, contextConnection=False, useAlgorithmLayeredSANI=True)
				connection = SANINeuron.HFcausalTargetConnectionLayeredDict[SANINeuron.noncontinguousInputNodeAbove.SANIlayerNeuronID]
				connection.SANIoptionalCausalConnection = True
				connection.weight += 1
		SANINeuronPrev = SANINeuron
	
def updateLayeredSANIlayerStandard(HFconnectionGraphObject, networkConceptNodeDict, SANIlayerList, layerIndex, layer, generateSANINetwork=True):
	layerBufferSANINodeList = layer.sentenceSANINodeList
	#print("len(layerBufferSANINodeList) = ", len(layerBufferSANINodeList))
	sentenceSANINodeList = []
	networkSANINodeList = SANIlayerList[layerIndex+1].networkSANINodeList
	layerNetworkIndex = len(networkSANINodeList)
	networkIndex = len(networkConceptNodeDict)
	for x1, SANINeuron1 in enumerate(layerBufferSANINodeList):
		if(selectActivatedTop):
			layerSANINodeListAssociated = []
		for x2, SANINeuron2 in enumerate(layerBufferSANINodeList):
			#print("1 updateLayeredSANIlayerStandard")
			if(checkValidAssociation(x1, x2, SANINeuron1, SANINeuron2, layerIndex, layerBufferSANINodeList)):
				if(not connectionExists(SANINeuron1, SANINeuron2, True)):
					HFNLPpy_hopfieldOperations.addConnectionToNode(SANINeuron1, SANINeuron2, contextConnection=True, useAlgorithmLayeredSANI=True)
				connection = SANINeuron1.HFcontextTargetConnectionLayeredDict[SANINeuron2.SANIlayerNeuronID]
				if(selectActivatedTop):
					foundSANIneuronCompound = False
					if(LayeredSANIwordVectors):
						foundNeuronCompound, SANINeuronCompound, SANINeuronCompoundWeight = findNeuronCompound(x1, layerBufferSANINodeList, HFconnectionGraphObject, networkConceptNodeDict, SANINeuron1, SANINeuron2)
						if(foundNeuronCompound):
							if(SANINeuronCompoundWeight > LayeredSANInodeGenerationHFassociationThresholdWordVector):
								if(SANINeuronCompound.graphNodeType == graphNodeTypeSANIhidden):
									layerSANINodeListAssociated.append(SANINeuronCompound)
									SANINeuronCompound.wTemp = x2
									SANINeuronCompound.weight = SANINeuronCompoundWeight	#temp artificial param	#TODO: SANINeuronCompound "weights" must be normalised wrt connection.weights
									SANINeuronCompound.isSANIcompoundNode = True	#temp artificial param
									foundSANIneuronCompound = True
					if(not foundSANIneuronCompound):
						if(connection.weight > LayeredSANInodeGenerationHFassociationThreshold):
							layerSANINodeListAssociated.append(connection)
							connection.nodeTarget.wTemp = x2	#temp artificial param
				if(HFassociationStrengthProximityBias):
					HFproximity = calculateHFproximity(layerBufferSANINodeList, SANINeuron1, SANINeuron2)
					#print("HFproximity = ", HFproximity)
					connection.weight += HFproximity
				else:
					connection.weight += 1
		if(selectActivatedTop):
			layerSANINodeListAssociated.sort(key=lambda x: x.weight)
			selectActivatedTopKChecked = min([selectActivatedTopK, len(layerSANINodeListAssociated)])
			layerSANINodeListAssociated = layerSANINodeListAssociated[0:selectActivatedTopKChecked]
			layerSANINodeListAssociatedTopK = []
			for association in layerSANINodeListAssociated:
				if(association.isSANIcompoundNode):
					layerSANINodeListAssociatedTopK.append(association)
				else:
					layerSANINodeListAssociatedTopK.append(association.nodeTarget)
		else:
			layerSANINodeListAssociatedTopK = layerBufferSANINodeList
		foundAssociation = False
		for SANINeuron2 in layerSANINodeListAssociatedTopK:
			x2 = SANINeuron2.wTemp
			if(SANINeuron2.isSANIcompoundNode):
				sentenceSANINodeList.append(SANINeuron2)
				SANINeuron2.isSANIcompoundNode = False
			else:
				if(checkValidAssociation(x1, x2, SANINeuron1, SANINeuron2, layerIndex, layerBufferSANINodeList)):
					connection = SANINeuron1.HFcontextTargetConnectionLayeredDict[SANINeuron2.SANIlayerNeuronID]
					if(connection.weight >= LayeredSANInodeGenerationHFassociationThreshold):
						addNodeToNextLayer = True
						foundAssociation = True
						if(not connection.SANInodeAssigned):
							if(generateSANINetwork):
								connection.SANInodeAssigned = True
								nodeName = generateSANInodeName(SANINeuron1, SANINeuron2, layerIndex+1, networkIndex)
								if(printVerbose):
									print("updateLayeredSANIlayerStandard: create new conceptNode; ", nodeName)
								nodeGraphType = graphNodeTypeSANIhidden
								connection.SANInode = HopfieldNode(networkIndex, nodeName, nodeGraphType)
								assignWindex(connection.SANInode, SANINeuron1, SANINeuron2)
								connection.SANInode.SANIlayerIndex = layerIndex+1
								connection.SANInode.SANIlayerNeuronID = layerNetworkIndex
								if(checkNodeContiguity(SANINeuron1, SANINeuron2, False)):  
									connection.SANIcontiguousInput = True
								networkSANINodeList.append(connection.SANInode)
								layerNetworkIndex += 1
								networkIndex = addNodeToGraph(connection.SANInode, networkConceptNodeDict, networkIndex)
							else:
								addToNextLayer = False
						if(addNodeToNextLayer):
							connection.SANIactivationState = True
							connection.SANInode.SANIactivationState = True
							sentenceSANINodeList.append(connection.SANInode)
						if(LayeredSANIwordVectors):
							HFNLPpy_Matrix.addSentenceConceptNodeToHFconnectionGraphObject(HFconnectionGraphObject, connection.SANInode)
		if(enableSkipLayerConnectivity):
			if(not foundAssociation):
				sentenceSANINodeList.append(SANINeuron1)	#append unassociated node to buffer so that it can still be referenced in future layers
				
	SANIlayerList[layerIndex+1].sentenceSANINodeList = sentenceSANINodeList

def getSANIlayerNeuronID(nodeName, networkIndex):
	SANIlayerNeuronID = nodeName	#orig: networkIndex (not compatible with generalised HF..ConnectionDict code)
	return SANIlayerNeuronID
	
def assignWindex(SANInode, SANINeuron1, SANINeuron2):
	wApprox = ((SANINeuron1.w + SANINeuron2.w)/2)
	SANInode.w=wApprox
	SANInode.wMin=SANINeuron1.w
	SANInode.wMax=SANINeuron2.w
								
def checkValidAssociation(x1, x2, SANINeuron1, SANINeuron2, layerIndex, layerBufferSANINodeList):
	result = False
	if(checkNonReplicateAssociation(x1, x2)):
		if(checkSkipLayerConnectivity(SANINeuron1, SANINeuron2, layerIndex)):
			if(checkNodeContiguity(SANINeuron1, SANINeuron2)):
				if(checkNonContiguousAssociationContents(SANINeuron1, SANINeuron2, layerBufferSANINodeList)):
					result = True
	return result

def checkNonReplicateAssociation(x1, x2):
	result = False
	if(x1 != x2 and x2 > x1):	#x2 > x1; ensures that replicate SANI nodes are not added to network (with input swapped)
		result = True	
	return result
	
def checkSkipLayerConnectivity(SANINeuron1, SANINeuron2, layerIndex):
	result = False
	if(enableSkipLayerConnectivity):
		if((SANINeuron1.SANIlayerIndex == layerIndex) or (SANINeuron2.SANIlayerIndex == layerIndex)):	#at least one node is on the current layer
			result = True
	else:
		result = True
	return result

def checkNodeContiguity(SANINeuron1, SANINeuron2, supportPermutationInvariance=HFassociationPermutationInvariance):
	wContiguityChecks = False
	if(supportPermutationInvariance):
		wContiguityChecks = True
	else:	#wContiguityEnforced
		if(SANINeuron1.wMax+1 == SANINeuron2.wMin):
			wContiguityChecks = True
	return wContiguityChecks	

def checkNonContiguousAssociationContents(SANINeuron1, SANINeuron2, layerBufferSANINodeList):
	result = True
	if(enableNextWordCausalPredictionsPermutationInvariance):
		for node in layerBufferSANINodeList:
			if(not node.SANIcontiguousInput):
				candidateInNodeWindexWindow1 = candidateInNodeWindexCentralWindow(SANINeuron1, node)
				candidateInNodeWindexWindow2 = candidateInNodeWindexCentralWindow(SANINeuron2, node)
				if(candidateInNodeWindexWindow1 or candidateInNodeWindexWindow2):
					if(candidateInNodeWindexWindow1 and candidateInNodeWindexWindow2):
						if(SANINeuron1.wMin == node.wMin+1):	#CHECKTHIS; +1
							SANINeuron1.noncontinguousInputNodeAboveCentralContentsStart = True
						if(SANINeuron2.wMax == node.wMax-1):	#CHECKTHIS; -1
							SANINeuron1.noncontinguousInputNodeAboveCentralContentsEnd = True
							SANINeuron1.noncontinguousInputNodeAbove = node
					else:
						result = False
	return result

def candidateInNodeWindexCentralWindow(candidateNode, nonContiguousNode):
	result = False
	if((candidateNode.wMin > nonContiguousNode.wMin) and (candidateNode.wMax < nonContiguousNode.wMax)):	#CHECKTHIS; >= <=
		result = True
	return result
	
def generateSANInodeName(SANINeuron1, SANINeuron2, layerIndex, layerNetworkIndex):
	#nodeName = "l" + str(layerIndex) + "n" + str(layerNetworkIndex)
	nodeName = SANINeuron1.nodeName + "_" + SANINeuron2.nodeName
	return nodeName

def calculateHFproximity(layerBufferSANINodeList, SANINeuron1, SANINeuron2):
	HFproximity = ((len(layerBufferSANINodeList) - min(len(layerBufferSANINodeList), abs(SANINeuron1.w - SANINeuron2.w))) / len(layerBufferSANINodeList)) * HFassociationStrengthProximityBiasLevel
	return HFproximity
	
def clearNetworkActivations(SANIlayerList):
	for layerIndex, layer in enumerate(SANIlayerList): 
		for nodeIndex, SANINode in enumerate(layer.networkSANINodeList):
			#print("layerIndex = ", layerIndex)
			SANINode.SANIactivationState = False
			for connectionKey, connection in SANINode.HFcontextTargetConnectionLayeredDict.items():
				if(HFassociationStrengthAtrophy):
					connection.weight += HFassociationStrengthAtrophy
				connection.SANIactivationState = False
				

def addNodeToGraph(conceptNode, networkConceptNodeDict, networkSize):
	if(conceptNode.nodeName not in networkConceptNodeDict):
		#print("addNodeToGraph: conceptNode.nodeName = ", conceptNode.nodeName)
		networkConceptNodeDict[conceptNode.nodeName] = conceptNode
		networkSize += 1
	else:
		print("addNodeToGraph error: conceptNode.nodeName already in networkConceptNodeDict")
		exit()
	return networkSize
		
def connectionExists(nodeSource, nodeTarget, contextConnection):
	result = False
	if(contextConnection):
		if(nodeTarget.SANIlayerNeuronID in nodeSource.HFcontextTargetConnectionLayeredDict):
			result = True
	else:
		if(nodeTarget.SANIlayerNeuronID in nodeSource.HFcausalTargetConnectionLayeredDict):
			result = True
	return result

if(LayeredSANIwordVectors):
	def findNeuronCompound(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, SANINeuron1, SANINeuron2):
		foundNeuronCompound = False
		SANINeuronCompound = None
		SANINeuronCompoundWeight = None
		if(useHFconnectionMatrixBasic):
			compoundWordVector = calculateCompoundWordVectorBasic(HFconnectionGraphObject, SANINeuron1, SANINeuron2)
			SANINeuronCompoundSet, SANINeuronCompoundWeight = getNearestWordVectorBasic(HFconnectionGraphObject, networkConceptNodeDict, compoundWordVector)
		elif(useHFconnectionMatrixAlgorithm):
			dendriticBranchIndex = None
			secondDataIndex = None
			secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax()
			weightStore = contextMatrixWeightStore
			bidirectionalContext = False	#not supported for prediction
			matrixTensorDim4 = (algorithmMatrixTensorDim == 4)
			assert(matrixTensorDim4)
			SANINeuronCompoundSet, SANINeuronCompoundWeight, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(w1, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, secondDataIndexMax, weightStore, bidirectionalContext, matrixTensorDim4, k=LayeredSANIwordVectorsTopK)
		else:
			printe("findNeuronCompound error: LayeredSANIwordVectors requires useHFconnectionMatrixBasic or useHFconnectionMatrixAlgorithm")
		if(len(SANINeuronCompoundSet) > 0):
			SANINeuronCompound = (list(SANINeuronCompoundSet))[0]	#only works with LayeredSANIwordVectorsTopK=1
			foundNeuronCompound = True
		return foundNeuronCompound, SANINeuronCompound, SANINeuronCompoundWeight

	def calculateCompoundWordVectorBasic(HFconnectionGraphObject, SANINeuron1, SANINeuron2):
		neuronID1 = HFconnectionGraphObject.neuronIDdict[SANINeuron1.nodeName]
		neuronID2 = HFconnectionGraphObject.neuronIDdict[SANINeuron1.nodeName]
		neuronWordVector1 = HFconnectionGraphObject.HFconnectionGraphBasic[neuronID1]
		neuronWordVector2 = HFconnectionGraphObject.HFconnectionGraphBasic[neuronID2]
		compoundWordVector = neuronWordVector1 + neuronWordVector2
		compoundWordVector = HFNLPpy_ConnectionMatrixBasic.normaliseBatchedTensor(compoundWordVector)
		return compoundWordVector

	def getNearestWordVectorBasic(HFconnectionGraphObject, networkConceptNodeDict, compoundWordVector):
		HFconnectionGraphBasicNormalised = HFconnectionGraphObject.HFconnectionGraphBasicNormalised	#more inefficient: HFconnectionGraphBasicNormalised = HFNLPpy_ConnectionMatrixBasic.normaliseBatchedTensor(HFconnectionGraphObject.HFconnectionGraphBasic)
		SANINeuronCompoundSet, SANINeuronCompoundWeight, _ = HFNLPpy_ConceptsMatrixOperations.connectionMatrixCalculateConnectionTargetSetBasic(HFconnectionGraphObject, HFconnectionGraphBasicNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, compoundWordVector, LayeredSANIwordVectorsTopK)
		return SANINeuronCompoundSet, SANINeuronCompoundWeight
