"""HFNLPpy_MatrixPropagate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

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
from HFNLPpy_MatrixGlobalDefs import *
from HFNLPpy_globalDefs import *
from collections import Counter

printVerbose = False
printConnectionTargetActivations = False


def simulateBiologicalHFnetworkSequenceNodesPropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject):
	somaActivationFound = False
	for conceptNeuronSource in conceptNeuronSourceList:
		if(simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject)):
			somaActivationFound = True
	return somaActivationFound
	
def simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, HFconnectionGraphObject):
	#if(printVerbose):
	print("simulateBiologicalHFnetworkSequenceNodePropagateStandard: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
	
	connectionTargetNeuronList = []
	secondDataIndexMax = HFNLPpy_MatrixOperations.getSecondDataIndexMax(getContextSizeSource=True, wSource=wSource+1)
	
	conceptNeuronID = HFconnectionGraphObject.neuronIDdict[conceptNeuronSource.nodeName]
	if(algorithmMatrixSingleTensor):
		conceptNeuronContextVector = HFNLPpy_MatrixOperations.createContextVectorWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, None, secondDataIndexMax, contextMatrixWeightStore, False, algorithmMatrixSingleTensor)	#HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[contextSizeIndex][conceptNeuronID]
		connectionTargetNeuronSetC, _, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphMatrixNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, matrixPropagateTopKconceptNodes, algorithmMatrixSingleTensor)
		connectionTargetNeuronListC = list(connectionTargetNeuronSetC)
		connectionTargetNeuronSet.update(connectionTargetNeuronSetC)
		if(debugAlgorithmMatrix):
			print("connectionTargetNeuronListC[0] = ", connectionTargetNeuronListC[0].nodeName)
	else:
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			connectionTargetNeuronList1 = []
			for secondDataIndex in range(secondDataIndexMax):
				conceptNeuronContextVector = HFNLPpy_MatrixOperations.createContextVectorWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, secondDataIndex, None, contextMatrixWeightStore, False, algorithmMatrixSingleTensor)	#HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[contextSizeIndex][conceptNeuronID]
				connectionTargetNeuronSet2, _, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][secondDataIndex], HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, matrixPropagateTopKconceptNodes, algorithmMatrixSingleTensor)
				connectionTargetNeuronList2 = list(connectionTargetNeuronSet2)
				connectionTargetNeuronList1.extend(connectionTargetNeuronList2)
				if(debugAlgorithmMatrix):
					print("connectionTargetNeuronList2[0] = ", connectionTargetNeuronList2[0].nodeName)
			if(algorithmMatrixSANItopkActivationAcrossSegments):
				connectionTargetNeuronListTopKneurons = performListTopK(connectionTargetNeuronList1, matrixPropagateTopKsequentialSegments)
				connectionTargetNeuronList.extend(connectionTargetNeuronListTopKneurons)
			else:
				connectionTargetNeuronList.extend(connectionTargetNeuronList1)
		connectionTargetNeuronListTopKneurons = performListTopK(connectionTargetNeuronList, matrixPropagateTopKdendriticBranches)
		connectionTargetNeuronSet.update(set(connectionTargetNeuronListTopKneurons))

	somaActivationFound = False
	if(conceptNeuronTarget in connectionTargetNeuronSet):
		somaActivationFound = True
	
	return somaActivationFound

def performListTopK(connectionTargetNeuronList, k):
	#calculate top k prediction
	connectionTargetNeuronListTopK = Counter(connectionTargetNeuronList).most_common(k)
	connectionTargetNeuronListTopKkeys = [i[0] for i in connectionTargetNeuronListTopK]
	if(debugAlgorithmMatrix):
		print("connectionTargetNeuronListTopKkeys[0] = ", connectionTargetNeuronListTopKkeys[0].nodeName)
	return connectionTargetNeuronListTopKkeys
