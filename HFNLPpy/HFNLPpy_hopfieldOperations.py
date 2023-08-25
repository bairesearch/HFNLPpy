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
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *


def addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, SANIbiologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, SANIbiologicalSimulation=False, nodeTargetSequentialSegmentInput=None):
	connection = HopfieldConnection(nodeSource, nodeTarget, spatioTemporalIndex, activationTime, SANIbiologicalPrototype, SANIbiologicalSimulation)
	#nodeSource.targetConnectionList.append(connection)
	#nodeTarget.sourceConnectionList.append(connection)
	#print("addConnectionToNode: nodeTarget.nodeName = ", nodeTarget.nodeName)
	#print("addConnectionToNode: nodeSource.nodeName = ", nodeSource.nodeName)
	createConnectionKeyIfNonExistant(nodeSource.targetConnectionDict, nodeTarget.nodeName)
	createConnectionKeyIfNonExistant(nodeTarget.sourceConnectionDict, nodeSource.nodeName)
	nodeSource.targetConnectionDict[nodeTarget.nodeName].append(connection)
	nodeTarget.sourceConnectionDict[nodeSource.nodeName].append(connection)
	#connection.subsequenceConnection = subsequenceConnection
	if(SANIbiologicalPrototype):
		connection.SANIbiologicalPrototype = SANIbiologicalPrototype
		connection.weight = weight
		connection.contextConnection = contextConnection
		connection.contextConnectionSANIindex = contextConnectionSANIindex
	if(SANIbiologicalSimulation):
		connection.SANIbiologicalSimulation = SANIbiologicalSimulation
		connection.nodeTargetSequentialSegmentInput = nodeTargetSequentialSegmentInput
		connection.weight = weight
