"""HFNLPpy_biologicalSimulationGenerate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Generate

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_biologicalSimulationNode import *
import HFNLPpy_hopfieldOperations

printVerbose = False

def addPredictiveSequenceToNeuron(conceptNeuron, sentenceIndex, sentenceConceptNodeList, dendriticBranch, dendriticBranchMaxW, branchIndex1, expectFurtherSubbranches=True):
	
	activationTime = 0	#unactivated
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
	numberOfWordsInSequence = len(sentenceConceptNodeList)
	
	#no prediction found for previous sequence; generate prediction for conceptNeuron (encode subsequences in dendrite)
	#print("addPredictiveSequenceToNeuron:")

	previousContextConceptNode = sentenceConceptNodeList[dendriticBranchMaxW]
	currentSequentialSegmentIndex = 0	#biologicalSimulation implementation does not currently use local sequential segments (encode sequentiality in branch structure only)
	currentSequentialSegment = dendriticBranch.sequentialSegments[currentSequentialSegmentIndex]
	createNewConnection, existingSequentialSegmentInput = verifyCreateNewConnection(currentSequentialSegment, previousContextConceptNode)
	if(createNewConnection):
		newSequentialSegmentSegmentInputIndex = calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
		#print("addPredictiveSynapseToNeuron ", conceptNeuron.nodeName, " branchIndex1 = ", branchIndex1)
		currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex, previousContextConceptNode)
		#currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
		if(preventGenerationOfDuplicateConnections):
			currentSequentialSegment.inputs[previousContextConceptNode.nodeName] = currentSequentialSegmentInput			
		else:
			#print("newSequentialSegmentSegmentInputIndex = ", newSequentialSegmentSegmentInputIndex)
			currentSequentialSegment.inputs[newSequentialSegmentSegmentInputIndex] = currentSequentialSegmentInput
		addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)
	else:
		currentSequentialSegmentInput = existingSequentialSegmentInput
		
	if(expectFurtherSubbranches):
		for subbranchIndex, subbranch in enumerate(dendriticBranch.subbranches):
			#lengthOfSubsequenceScale = random.uniform(0, 1)
			#dendriticSubBranchMaxW = int(lengthOfSubsequenceScale*dendriticBranchMaxW)	
			lengthOfSubsequence = int(np.random.exponential()*subsequenceLengthCalibration)	#the more proximal the previous context, the more likely to form a synapse
			#lengthOfSubsequence = max(1, lengthOfSubsequence)
			lengthOfSubsequence = lengthOfSubsequence + 1	#ensure >= 1
			dendriticSubBranchMaxW = dendriticBranchMaxW-lengthOfSubsequence
			#print("dendriticSubBranchMaxW = ", dendriticSubBranchMaxW)
			dendriticSubBranchMaxW = max(dendriticSubBranchMaxW, 0)	#ensure >=0 (if zero then subbranch.seqentialSegment.firstInputInSequence will be set to true)	
			if(dendriticSubBranchMaxW == 0):
				expectFurtherSubbranches = False			
			if(len(subbranch.subbranches) == 0):
				expectFurtherSubbranches = False
				#print("no further subbranches")
			addPredictiveSequenceToNeuron(conceptNeuron, sentenceIndex, sentenceConceptNodeList, subbranch, dendriticSubBranchMaxW, branchIndex1+1, expectFurtherSubbranches)
	else:
		currentSequentialSegmentInput.firstInputInSequence = True
		#print("setting currentSequentialSegmentInput.firstInputInSequence, branchIndex1 = ", branchIndex1)


#adds predictive synapse such that subsequences occur in order
def addPredictiveSynapseToNeuron(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=False, biologicalSynapse=False, nodeTargetSequentialSegmentInput=None):
	HFNLPpy_hopfieldOperations.addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalPrototype=biologicalPrototype, weight=weight, subsequenceConnection=subsequenceConnection, contextConnection=contextConnection, contextConnectionSANIindex=contextConnectionSANIindex, biologicalSimulation=biologicalSimulation, biologicalSynapse=biologicalSynapse, nodeTargetSequentialSegmentInput=nodeTargetSequentialSegmentInput)
																							
def calculateNewSequentialSegmentInputIndex(currentSequentialSegment):
	newSequentialSegmentSegmentInputIndex = len(currentSequentialSegment.inputs)
	newSequentialSegmentSegmentInputIndex += 1	
	#print("newSequentialSegmentSegmentInputIndex = ", newSequentialSegmentSegmentInputIndex)
	return newSequentialSegmentSegmentInputIndex	

def verifyCreateNewConnection(sequentialSegment, sourceConceptNode):
	createNewConnection = True
	existingSequentialSegmentInput = None
	if(preventGenerationOfDuplicateConnections):
		foundSequentialSegmentInput, existingSequentialSegmentInput = findSequentialSegmentInputBySourceNode(sequentialSegment, sourceConceptNode)
		if(foundSequentialSegmentInput):
			createNewConnection = False
	return createNewConnection, existingSequentialSegmentInput
	



			
