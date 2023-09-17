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
		nodeSource.HFcontextTargetConnectionDict[nodeTarget.nodeName] = connection
		nodeTarget.HFcontextSourceConnectionDict[nodeSource.nodeName] = connection
		if(useAlgorithmLayeredSANIbiologicalSimulation):
			nodeSource.HFcontextTargetConnectionLayeredDict[nodeTarget.SANIlayerNeuronID] = connection
			nodeTarget.HFcontextSourceConnectionLayeredDict[nodeSource.SANIlayerNeuronID] = connection
		if(useAlgorithmDendriticSANIbiologicalSimulation):
			createConnectionKeyIfNonExistant(nodeSource.HFcontextTargetConnectionMultiDict, nodeTarget.nodeName)
			createConnectionKeyIfNonExistant(nodeTarget.HFcontextSourceConnectionMultiDict, nodeSource.nodeName)
			nodeSource.HFcontextTargetConnectionMultiDict[nodeTarget.nodeName].append(connection)
			nodeTarget.HFcontextSourceConnectionMultiDict[nodeSource.nodeName].append(connection)
			#connection.subsequenceConnection = subsequenceConnection
	else:
		nodeSource.HFcausalTargetConnectionDict[nodeTarget.nodeName] = connection
		nodeTarget.HFcausalSourceConnectionDict[nodeSource.nodeName] = connection
		if(useAlgorithmLayeredSANIbiologicalSimulation):
			nodeSource.HFcausalTargetConnectionLayeredDict[nodeTarget.SANIlayerNeuronID] = connection
			nodeTarget.HFcausalSourceConnectionLayeredDict[nodeSource.SANIlayerNeuronID] = connection
		if(useAlgorithmDendriticSANIbiologicalSimulation):
			createConnectionKeyIfNonExistant(nodeSource.HFcausalTargetConnectionMultiDict, nodeTarget.nodeName)
			createConnectionKeyIfNonExistant(nodeTarget.HFcausalSourceConnectionMultiDict, nodeSource.nodeName)
			nodeSource.HFcausalTargetConnectionMultiDict[nodeTarget.nodeName].append(connection)
			nodeTarget.HFcausalSourceConnectionMultiDict[nodeSource.nodeName].append(connection)
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
