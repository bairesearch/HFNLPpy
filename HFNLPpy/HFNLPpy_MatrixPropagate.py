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
	contextSizeMax2 = min(contextSizeMax, len(sentenceConceptNodeList))
	for contextSize in range(contextSizeMax2):
		#calculate top k prediction
		conceptNeuronID = HFconnectionGraphObject.neuronIDdict[conceptNeuronSource.nodeName]
		conceptNeuronContextVector = HFNLPpy_hopfieldOperations.createContextVector(wTarget, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionMatrixBasicMaxConcepts, contextSize, contextMatrixWeightStore, False)	#HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[contextSize][conceptNeuronID]
		#print("conceptNeuronContextVector = ", conceptNeuronContextVector)
		connectionTargetNeuronSetC = HFNLPpy_hopfieldOperations.connectionMatrixCalculateConnectionTargetSet(HFconnectionGraphObject.HFconnectionGraphMatrixNormalised[contextSize], HFconnectionGraphObject.neuronNamelist, networkConceptNodeDict, conceptNeuronContextVector, matrixPropagateTopK1)
		connectionTargetNeuronList.extend(list(connectionTargetNeuronSetC))
	
	connectionTargetNeuronListTopK = Counter(connectionTargetNeuronList).most_common(matrixPropagateTopK2)
	connectionTargetNeuronListTopKkeys = [i[0] for i in connectionTargetNeuronListTopK]
	connectionTargetNeuronSet.update(set(connectionTargetNeuronListTopKkeys))
	
	#print("connectionTargetNeuronList = ", connectionTargetNeuronList)
	#print("connectionTargetNeuronListTopKkeys = ", connectionTargetNeuronListTopKkeys)
	print("connectionTargetNeuronListTop = ", connectionTargetNeuronListTopKkeys[0].nodeName)

	somaActivationFound = False
	if(conceptNeuronTarget in connectionTargetNeuronSet):
		somaActivationFound = True
	
	return somaActivationFound
