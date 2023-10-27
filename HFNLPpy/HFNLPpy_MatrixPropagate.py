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
	
	foundClosest = False
	foundDendriticBranchClosest = False
	dendriticBranchClosestTargetSet = set()
	closestConnectionStrength = 0
	#dendriticBranchClosestIndex = 0

	if(algorithmMatrixTensorDim4):
		connectionTargetNeuronSet0, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised, None, secondDataIndexMax, contextMatrixWeightStore, False, algorithmMatrixTensorDim4)
		foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, _ = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, None, connectionTargetNeuronSet0, connectionStrength, None)
		if(debugAlgorithmMatrix):
			print("connectionTargetNeuronListC[0] = ", list(connectionTargetNeuronSet0)[0].nodeName)
	else:
		for dendriticBranchIndex in range(numberOfIndependentDendriticBranches):
			if(algorithmMatrixTensorDim3):
				HFconnectionGraphMatrixNormalisedDB = HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex].unsqueeze(dim=0)
				connectionTargetNeuronSet1, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphMatrixNormalisedDB, None, secondDataIndexMax, contextMatrixWeightStore, False, True)
				foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, _ = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, None, connectionTargetNeuronSet1, connectionStrength, None)
				#connectionTargetNeuronList.extend(list(connectionTargetNeuronSet1))
			else:
				connectionTargetNeuronList1 = []
				for secondDataIndex in range(secondDataIndexMax):
					connectionTargetNeuronSet2, connectionStrength, _ = HFNLPpy_MatrixOperations.connectionMatrixCalculateConnectionTargetSetWrapper(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, networkConceptNodeDict, HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][secondDataIndex], secondDataIndex, None, contextMatrixWeightStore, False, algorithmMatrixTensorDim4)
					foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, _ = HFNLPpy_MatrixOperations.updateDendriticBranchClosestValue(foundClosest, dendriticBranchClosestTargetSet, closestConnectionStrength, None, connectionTargetNeuronSet2, connectionStrength, None)
					#connectionTargetNeuronList1.extend(list(connectionTargetNeuronSet2))
				#connectionTargetNeuronList.extend(connectionTargetNeuronList1)
		#connectionTargetNeuronListTopKneurons = performListTopK(connectionTargetNeuronList, matrixPropagateTopKdendriticBranches)
		#connectionTargetNeuronSet.update(set(connectionTargetNeuronListTopKneurons))
	if(foundClosest):
		connectionTargetNeuronSet.update(list(dendriticBranchClosestTargetSet))
	
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
