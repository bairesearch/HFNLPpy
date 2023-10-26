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
		connectionTargetNeuronSet0, _, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised, None, secondDataIndexMax, contextMatrixWeightStore, False, algorithmMatrixSingleTensor)
		connectionTargetNeuronSet.update(list(connectionTargetNeuronSet0))
		if(debugAlgorithmMatrix):
			print("connectionTargetNeuronListC[0] = ", list(connectionTargetNeuronSet0)[0].nodeName)
	else:
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			if(algorithmMatrixSANImethodUseActivationAcrossSegments):
				HFconnectionGraphMatrixNormalisedDB = HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex].unsqueeze(dim=0)
				connectionTargetNeuronSet1, _, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphMatrixNormalisedDB, None, secondDataIndexMax, contextMatrixWeightStore, False, True)
				connectionTargetNeuronList.extend(list(connectionTargetNeuronSet1))
			else:
				connectionTargetNeuronList1 = []
				for secondDataIndex in range(secondDataIndexMax):
					connectionTargetNeuronSet2, _, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][secondDataIndex], secondDataIndex, None, contextMatrixWeightStore, False, algorithmMatrixSingleTensor)
					connectionTargetNeuronList1.extend(list(connectionTargetNeuronSet2))
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
