"""HFNLPpy_MatrixPropagate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Matrix Propagate

"""

import numpy as np

import HFNLPpy_hopfieldOperations
import HFNLPpy_MatrixOperations
from HFNLPpy_globalDefs import *
from HFNLPpy_MatrixGlobalDefs import *
from collections import Counter
if(HFconnectionMatrixAlgorithmSplitDatabase):
	import HFNLPpy_MatrixDatabase

printVerbose = False
printConnectionTargetActivations = False


	
def simulateBiologicalHFnetworkSequenceNodesPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions):
	somaActivationFound = False
	connectionTargetNeuronSetLocal = set()
	for conceptNeuronSource in conceptNeuronSourceList:
		somaActivationFoundS, connectionTargetNeuronSetLocalS = simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions)
		connectionTargetNeuronSetLocal = connectionTargetNeuronSetLocal.union(connectionTargetNeuronSetLocalS)
		if(somaActivationFoundS):	
			somaActivationFound = True
	return somaActivationFound, connectionTargetNeuronSetLocal
	
def simulateBiologicalHFnetworkSequenceNodePropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject, useReversePredictions):
	if(printPredictions):
		print("simulateBiologicalHFnetworkSequenceNodePropagateStandard: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
	
	connectionTargetNeuronList = []
	secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax(getContextSizeSource=True, wSource=wSource+1)
	conceptNeuronID = HFconnectionGraphObject.neuronIDdict[conceptNeuronSource.nodeName]
	
	foundClosestBranchIndex = False
	foundDendriticBranchClosest = False
	dendriticBranchClosestTargetSet = set()
	closestConnectionStrength = 0
	#dendriticBranchClosestIndex = 0
	
	if(algorithmMatrixTensorDim==4):
		connectionTargetNeuronSet0, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, None, None, secondDataIndexMax, contextMatrixWeightStore, predictionBidirectionalContext, True, useReversePredictions=useReversePredictions, connectionTargetNeuronSet=connectionTargetNeuronSet)
		foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, _ = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, None, connectionTargetNeuronSet0, connectionStrength, None)
		if(debugAlgorithmMatrix):
			print("connectionTargetNeuronListC[0] = ", list(connectionTargetNeuronSet0)[0].nodeName)
	else:
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			if(algorithmMatrixTensorDim==3):
				connectionTargetNeuronSet1, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, None, secondDataIndexMax, contextMatrixWeightStore, predictionBidirectionalContext, True, useReversePredictions=useReversePredictions, connectionTargetNeuronSet=connectionTargetNeuronSet)
				foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, _ = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, None, connectionTargetNeuronSet1, connectionStrength, None)
				#connectionTargetNeuronList.extend(list(connectionTargetNeuronSet1))
			else:
				connectionTargetNeuronList1 = []
				for secondDataIndex in range(secondDataIndexMax):
					connectionTargetNeuronSet2, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, dendriticBranchIndex, secondDataIndex, None, contextMatrixWeightStore, predictionBidirectionalContext, False, useReversePredictions=useReversePredictions, connectionTargetNeuronSet=connectionTargetNeuronSet)
					foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, _ = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosestBranchIndex, dendriticBranchClosestTargetSet, closestConnectionStrength, None, connectionTargetNeuronSet2, connectionStrength, None)
					#connectionTargetNeuronList1.extend(list(connectionTargetNeuronSet2))
				#connectionTargetNeuronList.extend(connectionTargetNeuronList1)
		#connectionTargetNeuronListTopKneurons = performListTopK(connectionTargetNeuronList, matrixPropagateTopKdendriticBranches)
		#connectionTargetNeuronSet.update(set(connectionTargetNeuronListTopKneurons))
	if(foundClosestBranchIndex):
		#OLD: connectionTargetNeuronSetLocal = .update(list(dendriticBranchClosestTargetSet))
		addPredictedConceptNodesToHFconnectionGraphObject(HFconnectionGraphObject, sentenceConceptNodeList, list(dendriticBranchClosestTargetSet))	#OLD:  list(connectionTargetNeuronSet)
		
	somaActivationFound = False
	if(conceptNeuronTarget in dendriticBranchClosestTargetSet):
		somaActivationFound = True
		
	return somaActivationFound, dendriticBranchClosestTargetSet

def performListTopK(connectionTargetNeuronList, k):
	#calculate top k prediction
	connectionTargetNeuronListTopK = Counter(connectionTargetNeuronList).most_common(k)
	connectionTargetNeuronListTopKkeys = [i[0] for i in connectionTargetNeuronListTopK]
	if(debugAlgorithmMatrix):
		print("connectionTargetNeuronListTopKkeys[0] = ", connectionTargetNeuronListTopKkeys[0].nodeName)
	return connectionTargetNeuronListTopKkeys

def addPredictedConceptNodesToHFconnectionGraphObject(HFconnectionGraphObject, sentenceConceptNodeList, predictedConceptNodeList):
	if(HFconnectionMatrixAlgorithmSplitDatabase):
		neuronIDdictNewlyAdded = {}
		for conceptNodeIndex, conceptNode in enumerate(sentenceConceptNodeList):
			neuronID = conceptNode.networkIndex
			neuronIDdictNewlyAdded[conceptNode.nodeName] = neuronID	
			neuronIDdictNewlyAdded = {}
		for conceptNodeIndex, conceptNode in enumerate(predictedConceptNodeList):
			#print("conceptNodeIndex = ", conceptNodeIndex)
			if(not conceptNode.nodeName in HFconnectionGraphObject.neuronIDdict):
				printe("addPredictedConceptNodesToHFconnectionGraphObject error: not conceptNode.nodeName in HFconnectionGraphObject.neuronIDdict; predicted concepts should be in neuronIDdict")				
			else:
				#load HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID] into RAM
				neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
				#print("neuronID = ", neuronID)
				#print("conceptNode.nodeName = ", conceptNode.nodeName)
				if(conceptNode.nodeName not in neuronIDdictNewlyAdded):
					#load a predicted neuron matrix column that is not in the current sentence
					HFconnectionGraphObject.HFconnectionGraphMatrix[neuronID] = HFNLPpy_MatrixDatabase.loadMatrixDatabaseFile(HFconnectionGraphObject, neuronID)
