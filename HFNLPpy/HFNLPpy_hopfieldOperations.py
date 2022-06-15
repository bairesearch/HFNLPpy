"""HFNLPpy_hopfieldOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

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


def addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalImplementation=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=False, biologicalSynapse=False, nodeTargetBranchIndex1=None, nodeTargetBranchIndex2=None, nodeTargetSequentialSegmentIndex=None, nodeTargetSequentialSegmentInputIndex=None):
	connection = HopfieldConnection(nodeSource, nodeTarget, spatioTemporalIndex, activationTime)
	#nodeSource.targetConnectionList.append(connection)
	#nodeTarget.sourceConnectionList.append(connection)
	#print("addConnectionToNode: nodeTarget.nodeName = ", nodeTarget.nodeName)
	#print("addConnectionToNode: nodeSource.nodeName = ", nodeSource.nodeName)
	createConnectionKeyIfNonExistant(nodeSource.targetConnectionDict, nodeTarget.nodeName)
	createConnectionKeyIfNonExistant(nodeTarget.sourceConnectionDict, nodeSource.nodeName)
	nodeSource.targetConnectionDict[nodeTarget.nodeName].append(connection)
	nodeTarget.sourceConnectionDict[nodeSource.nodeName].append(connection)
	#connection.subsequenceConnection = subsequenceConnection
	if(biologicalImplementation):
		connection.weight = weight
		connection.contextConnection = contextConnection
		connection.contextConnectionSANIindex = contextConnectionSANIindex
	if(biologicalSimulation):
		connection.biologicalSynapse = biologicalSynapse
		connection.nodeTargetBranchIndex1 = nodeTargetBranchIndex1
		connection.nodeTargetBranchIndex2 = nodeTargetBranchIndex2
		connection.nodeTargetSequentialSegmentIndex = nodeTargetSequentialSegmentIndex
		connection.nodeTargetSequentialSegmentInputIndex = nodeTargetSequentialSegmentInputIndex
		
