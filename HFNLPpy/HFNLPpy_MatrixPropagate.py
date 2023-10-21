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
	contextSizeMax2 = min(contextSizeMax, wSource+1)	#len(sentenceConceptNodeList)
	conceptNeuronID = HFconnectionGraphObject.neuronIDdict[conceptNeuronSource.nodeName]
	if(algorithmMatrixSingleTensor):
		contextSizeIndex = contextSizeMax2
		conceptNeuronContextVector = HFNLPpy_hopfieldOperations.createContextVector(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, contextSizeIndex, contextMatrixWeightStore, False)	#HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[contextSizeIndex][conceptNeuronID]
		connectionTargetNeuronSetC, _, _ = HFNLPpy_hopfieldOperations.connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphMatrixNormalised, HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, matrixPropagateTopKconceptNodes, algorithmMatrixSingleTensor, contextSizeMax2)
		connectionTargetNeuronListC = list(connectionTargetNeuronSetC)
		connectionTargetNeuronSet.update(connectionTargetNeuronSetC)
		if(debugAlgorithmMatrix):
			print("connectionTargetNeuronListC[0] = ", connectionTargetNeuronListC[0].nodeName)
	else:
		for dendriticBranchIndex in range(numberOfDendriticBranches):
			for contextSizeIndex in range(contextSizeMax2):
				conceptNeuronContextVector = HFNLPpy_hopfieldOperations.createContextVector(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, contextSizeIndex, contextMatrixWeightStore, False)	#HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[contextSizeIndex][conceptNeuronID]
				connectionTargetNeuronSetC, _, _ = HFNLPpy_hopfieldOperations.connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[dendriticBranchIndex][contextSizeIndex], HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, matrixPropagateTopKconceptNodes, algorithmMatrixSingleTensor)
				connectionTargetNeuronListC = list(connectionTargetNeuronSetC)
				connectionTargetNeuronList.extend(connectionTargetNeuronListC)
				if(debugAlgorithmMatrix):
					print("connectionTargetNeuronListC[0] = ", connectionTargetNeuronListC[0].nodeName)
		#calculate top k prediction
		connectionTargetNeuronListTopK = Counter(connectionTargetNeuronList).most_common(matrixPropagateTopKcontextSize)
		connectionTargetNeuronListTopKkeys = [i[0] for i in connectionTargetNeuronListTopK]
		connectionTargetNeuronSet.update(set(connectionTargetNeuronListTopKkeys))
		if(debugAlgorithmMatrix):
			print("connectionTargetNeuronListTopKkeys[0] = ", connectionTargetNeuronListTopKkeys[0].nodeName)
	
	somaActivationFound = False
	if(conceptNeuronTarget in connectionTargetNeuronSet):
		somaActivationFound = True
	
	return somaActivationFound
