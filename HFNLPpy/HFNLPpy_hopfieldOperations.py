"""HFNLPpy_hopfieldOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Hopfield Operations

Contains shared HFNLP operations on HFNLPpy_hopfieldNodeClass/HFNLPpy_hopfieldConnectionClass

"""

import numpy as np
from HFNLPpy_globalDefs import *
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *


def addConnectionToNode(nodeSource, nodeTarget, activationTime=-1, spatioTemporalIndex=-1, useAlgorithmDendriticSANIbiologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, useAlgorithmDendriticSANIbiologicalSimulation=False, nodeTargetSequentialSegmentInput=None):
	connection = HopfieldConnection(nodeSource, nodeTarget, spatioTemporalIndex, activationTime, useAlgorithmDendriticSANIbiologicalPrototype)
	if(assignSingleConnectionBetweenUniqueConceptPair):
		nodeSource.HFtargetConnectionDict[nodeTarget.SANIlayerNeuronID] = connection
		nodeTarget.HFsourceConnectionDict[nodeSource.SANIlayerNeuronID] = connection
	else:
		createConnectionKeyIfNonExistant(nodeSource.HFtargetConnectionDict, nodeTarget.nodeName)
		createConnectionKeyIfNonExistant(nodeTarget.HFsourceConnectionDict, nodeSource.nodeName)
		nodeSource.HFtargetConnectionDict[nodeTarget.nodeName].append(connection)
		nodeTarget.HFsourceConnectionDict[nodeSource.nodeName].append(connection)
		#connection.subsequenceConnection = subsequenceConnection
	if(useAlgorithmDendriticSANIbiologicalPrototype):
		connection.useAlgorithmDendriticSANIbiologicalPrototype = useAlgorithmDendriticSANIbiologicalPrototype
		connection.weight = weight
		connection.contextConnection = contextConnection
		connection.contextConnectionSANIindex = contextConnectionSANIindex
	if(useAlgorithmDendriticSANIbiologicalSimulation):
		connection.useAlgorithmDendriticSANIbiologicalSimulation = useAlgorithmDendriticSANIbiologicalSimulation
		connection.nodeTargetSequentialSegmentInput = nodeTargetSequentialSegmentInput
		connection.weight = weight

def connectionExists(nodeSource, nodeTarget):
	result = False
	if(assignSingleConnectionBetweenUniqueConceptPair):
		if(nodeTarget.SANIlayerNeuronID in nodeSource.HFtargetConnectionDict):
			result = True
	else:
		if(nodeTarget.nodeName in nodeSource.HFtargetConnectionDict):
			result = True
	return result
		
