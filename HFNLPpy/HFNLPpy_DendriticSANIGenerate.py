"""HFNLPpy_DendriticSANIGenerate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI Generate

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations
from HFNLPpy_DendriticSANIGlobalDefs import *
from HFNLPpy_DendriticSANINode import *

printVerbose = False

debugSubsequenceGeneration = True

def addPredictiveSequenceToNeuron(conceptNeuron, sentenceIndex, sentenceConceptNodeList, dendriticBranch, predictiveSequenceLength, dendriticBranchMaxW, branchIndex1, sequentialSegmentIndex, expectFurtherSubbranches=True):
	
	activationTime = 0	#unactivated
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
	numberOfWordsInSequence = len(sentenceConceptNodeList)
	
	#no prediction found for previous sequence; generate prediction for conceptNeuron (encode subsequences in dendrite)
	
	if((branchIndex1 > 0) or expectFirstBranchSequentialSegmentConnection):
		previousContextConceptNode = sentenceConceptNodeList[dendriticBranchMaxW]	
		currentSequentialSegment = dendriticBranch.sequentialSegments[sequentialSegmentIndex]
		createNewConnection, existingSequentialSegmentInput = verifyCreateNewConnection(currentSequentialSegment, previousContextConceptNode)
		if(createNewConnection):
			if(printVerbose):
				print("addPredictiveSequenceToNeuron: conceptNeuron = ", conceptNeuron.nodeName, ", previousContextConceptNode = ", previousContextConceptNode.nodeName)
			weight = sequentialSegmentMinActivationLevel
			newSequentialSegmentSegmentInputIndex = calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
			#print("addPredictiveSynapseToNeuron ", conceptNeuron.nodeName, " branchIndex1 = ", branchIndex1)
			currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex, previousContextConceptNode)
			#currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
			if(preventGenerationOfDuplicateConnections):
				currentSequentialSegment.inputs[previousContextConceptNode.nodeName] = currentSequentialSegmentInput			
			else:
				#print("newSequentialSegmentSegmentInputIndex = ", newSequentialSegmentSegmentInputIndex)
				currentSequentialSegment.inputs[newSequentialSegmentSegmentInputIndex] = currentSequentialSegmentInput
			addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, useAlgorithmDendriticSANIbiologicalPrototype=False, weight=weight, subsequenceConnection=False, contextConnection=True, contextConnectionSANIindex=0, useAlgorithmDendriticSANIbiologicalSimulation=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)
		else:
			currentSequentialSegmentInput = existingSequentialSegmentInput
		
	if(expectFurtherSubbranches):
		if(isMostDistalSequentialSegmentInBranch(sequentialSegmentIndex)):
			if((branchIndex1 < numberOfBranches1-1) or not expectFirstBranchSequentialSegmentConnectionStrictNumBranches1):
				if(trainSubsetOfHorizontalSubbranches):
					subbranchIndices = list(range(len(dendriticBranch.subbranches)))
					np.random.shuffle(subbranchIndices)	#random.shuffle(subbranchIndices)
				#for subbranchIndex, subbranch in enumerate(dendriticBranch.subbranches):
				for i in range(numberOfHorizontalSubBranchesTrained):
					if(trainSubsetOfHorizontalSubbranches):
						subbranchIndex = subbranchIndices[i]
						subbranch = dendriticBranch.subbranches[subbranchIndex]
					else:
						subbranch = dendriticBranch.subbranches[i]
					expectFurtherSubbranches2 = True
					if(len(subbranch.subbranches) == 0):
						expectFurtherSubbranches2 = False
					#print("expectFurtherSubbranches2 = ", expectFurtherSubbranches2)
					addPredictiveSequenceToNeuronSubsequenceGeneration(conceptNeuron, sentenceIndex, sentenceConceptNodeList, subbranch, predictiveSequenceLength, dendriticBranchMaxW, branchIndex1+1, 0, expectFurtherSubbranches2)
		else:
			addPredictiveSequenceToNeuronSubsequenceGeneration(conceptNeuron, sentenceIndex, sentenceConceptNodeList, dendriticBranch, predictiveSequenceLength, dendriticBranchMaxW, branchIndex1, sequentialSegmentIndex+1, expectFurtherSubbranches)
	else:
		currentSequentialSegmentInput.firstInputInSequence = True
		#print("setting currentSequentialSegmentInput.firstInputInSequence, branchIndex1 = ", branchIndex1)

def addPredictiveSequenceToNeuronSubsequenceGeneration(conceptNeuron, sentenceIndex, sentenceConceptNodeList, dendriticBranch, predictiveSequenceLength, dendriticBranchMaxW, branchIndex1, sequentialSegmentIndex, expectFurtherSubbranches):
	if(subsequenceLengthRandExponential):
		lengthOfSubsequence = np.random.exponential()*subsequenceLengthCalibration	#the more proximal the previous context, the more likely to form a synapse	
	else:
		lengthOfSubsequenceScale = np.random.uniform(0, 1) #random.uniform(0, 1)
		lengthOfSubsequence = lengthOfSubsequenceScale*subsequenceLengthCalibration

	if(reduceCompletenessOfEncodingWithPreviousContextDistance):
		distanceOfPreviousContextNode = (conceptNeuron.w-dendriticBranchMaxW)
		#print("distanceOfPreviousContextNode = ", distanceOfPreviousContextNode)
		lengthOfSubsequence = lengthOfSubsequence*distanceOfPreviousContextNode
	if(reduceCompletenessOfEncodingWithSequenceLength):
		lengthOfSubsequence = lengthOfSubsequence*(predictiveSequenceLength/reduceCompletenessOfEncodingWithSequenceLengthCalibration) #the shorter the sequence, the more likely to form a synapse

	lengthOfSubsequence = int(lengthOfSubsequence)
	
	lengthOfSubsequence = lengthOfSubsequence + 1	#ensure >= 1
	if(verifyPropagationTime):
		lengthOfSubsequence = min(lengthOfSubsequence, int(activationPropagationTimeMax))
	
	if(printVerbose):
		print("lengthOfSubsequence = ", lengthOfSubsequence)
		
	dendriticSubBranchMaxW = dendriticBranchMaxW-lengthOfSubsequence
	dendriticSubBranchMaxW = max(dendriticSubBranchMaxW, 0)	#ensure >=0 (if zero then subbranch.seqentialSegment.firstInputInSequence will be set to true)	
	if(dendriticSubBranchMaxW == 0):
		expectFurtherSubbranches = False			
	#print("no further subbranches")
	
	addPredictiveSequenceToNeuron(conceptNeuron, sentenceIndex, sentenceConceptNodeList, dendriticBranch, predictiveSequenceLength, dendriticSubBranchMaxW, branchIndex1, sequentialSegmentIndex, expectFurtherSubbranches)

#adds predictive synapse such that subsequences occur in order
def addPredictiveSynapseToNeuron(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, useAlgorithmDendriticSANIbiologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=True, contextConnectionSANIindex=0, useAlgorithmDendriticSANIbiologicalSimulation=False, nodeTargetSequentialSegmentInput=None):
	HFNLPpy_hopfieldOperations.addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, useAlgorithmDendriticSANIbiologicalPrototype=useAlgorithmDendriticSANIbiologicalPrototype, weight=weight, subsequenceConnection=subsequenceConnection, contextConnection=contextConnection, contextConnectionSANIindex=contextConnectionSANIindex, useAlgorithmDendriticSANIbiologicalSimulation=useAlgorithmDendriticSANIbiologicalSimulation, nodeTargetSequentialSegmentInput=nodeTargetSequentialSegmentInput)
																							
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
	



			
