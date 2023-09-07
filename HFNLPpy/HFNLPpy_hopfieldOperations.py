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
	connection.contextConnection = contextConnection
	
	if(contextConnection):
		if(assignSingleConnectionBetweenUniqueConceptPair):
			nodeSource.HFcontextTargetConnectionDict[nodeTarget.SANIlayerNeuronID] = connection
			nodeTarget.HFcontextSourceConnectionDict[nodeSource.SANIlayerNeuronID] = connection
		else:
			createConnectionKeyIfNonExistant(nodeSource.HFcontextTargetConnectionDict, nodeTarget.nodeName)
			createConnectionKeyIfNonExistant(nodeTarget.HFcontextSourceConnectionDict, nodeSource.nodeName)
			nodeSource.HFcontextTargetConnectionDict[nodeTarget.nodeName].append(connection)
			nodeTarget.HFcontextSourceConnectionDict[nodeSource.nodeName].append(connection)
			#connection.subsequenceConnection = subsequenceConnection
	else:
		if(assignSingleConnectionBetweenUniqueConceptPair):
			nodeSource.HFcausalTargetConnectionDict[nodeTarget.SANIlayerNeuronID] = connection
			nodeTarget.HFcausalSourceConnectionDict[nodeSource.SANIlayerNeuronID] = connection
		else:
			createConnectionKeyIfNonExistant(nodeSource.HFcausalTargetConnectionDict, nodeTarget.nodeName)
			createConnectionKeyIfNonExistant(nodeTarget.HFcausalSourceConnectionDict, nodeSource.nodeName)
			nodeSource.HFcausalTargetConnectionDict[nodeTarget.nodeName].append(connection)
			nodeTarget.HFcausalSourceConnectionDict[nodeSource.nodeName].append(connection)
			#connection.subsequenceConnection = subsequenceConnection
		
	if(useAlgorithmDendriticSANIbiologicalPrototype):
		connection.useAlgorithmDendriticSANIbiologicalPrototype = useAlgorithmDendriticSANIbiologicalPrototype
		connection.weight = weight
		connection.contextConnectionSANIindex = contextConnectionSANIindex
	if(useAlgorithmDendriticSANIbiologicalSimulation):
		connection.useAlgorithmDendriticSANIbiologicalSimulation = useAlgorithmDendriticSANIbiologicalSimulation
		connection.nodeTargetSequentialSegmentInput = nodeTargetSequentialSegmentInput
		connection.weight = weight
	return connection

def connectionExists(nodeSource, nodeTarget, contextConnection):
	result = False
	if(contextConnection):
		if(assignSingleConnectionBetweenUniqueConceptPair):
			if(nodeTarget.SANIlayerNeuronID in nodeSource.HFcontextTargetConnectionDict):
				result = True
		else:
			if(nodeTarget.nodeName in nodeSource.HFcontextTargetConnectionDict):
				result = True
	else:
		if(assignSingleConnectionBetweenUniqueConceptPair):
			if(nodeTarget.SANIlayerNeuronID in nodeSource.HFcausalTargetConnectionDict):
				result = True
		else:
			if(nodeTarget.nodeName in nodeSource.HFcausalTargetConnectionDict):
				result = True
	return result
		
