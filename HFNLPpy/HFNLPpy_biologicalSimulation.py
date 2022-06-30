"""HFNLPpy_biologicalSimulation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation - simulate training/inference of biological hopfield graph/network based on textual input

pseudo code;
for every time step/concept neuron (word w):
	for every branch in dendriticTree (2+, recursively parsed from outer to inner; sequentially dependent):
		for every sequential segment in branch (1+, sequentially dependent):
			for every non-sequential synapse (input) in segment (1+, sequentially independent):
				calculate local dendritic activation
					subject to readiness (repolarisation time at dendrite location)
	calculate neuron activation
		subject to readiness (repolarisation time)

training;
	activate concept neurons in order of sentence word order
	strengthen those synapses which directly precede/contribute to firing
		weaken those that do not
	this will enable neuron to fire when specific contextual instances are experienced
inference;
	calculate neuron firing exclusively from prior/contextual subsequence detections

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_biologicalSimulationNode import *
import HFNLPpy_hopfieldOperations
if(vectoriseComputation):
	import HFNLPpy_biologicalSimulationVectorised
else:
	import HFNLPpy_biologicalSimulationStandard

printVerbose = False

debugCalculateNeuronActivationParallel = False
if(debugCalculateNeuronActivationParallel):
	wSourceDebug = 5
	wTargetDebug = wSourceDebug+1

#alwaysAddPredictionInputFromPreviousConcept = False
#if(vectoriseComputation):
#	alwaysAddPredictionInputFromPreviousConcept = True #ensures that simulateBiologicalHFnetworkSequenceNodeTrainParallel:conceptNeuronBatchIndexFound

drawBiologicalSimulation = False	#default: True
if(drawBiologicalSimulation):
	drawBiologicalSimulationDendriticTreeSentence = True	#default: True	#draw graph for sentence neurons and their dendritic tree
	if(drawBiologicalSimulationDendriticTreeSentence):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentence
	drawBiologicalSimulationDendriticTreeNetwork = True	#default: True	#draw graph for entire network (not just sentence)
	if(drawBiologicalSimulationDendriticTreeNetwork):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetwork


def simulateBiologicalHFnetworkSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations):
	simulateBiologicalHFnetworkSequenceTrainSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations)					

def simulateBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):
	simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	


#if (!biologicalSimulation:useDependencyParseTree):

def simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):

	#cannot clear now as HFNLPpy_biologicalSimulationDrawSentence/HFNLPpy_biologicalSimulationDrawNetwork memory structure is not independent (diagnose reason for this);
	
	sentenceLength = len(sentenceConceptNodeList)
	
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	
	for wTarget in range(1, sentenceLength):	#wTarget>=1: do not create (recursive) connection from conceptNode to conceptNode branchIndex1=0
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainWrapper(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, connectionTargetNeuronSet)

		if(somaActivationFound):
			#if(printVerbose):
			print("somaActivationFound")
		else:
			#if(printVerbose):
			print("!somaActivationFound: addPredictiveSequenceToNeuron")
			dendriticBranchMaxW = wTarget-1
			expectFurtherSubbranches = True
			if(wTarget == 1):
				expectFurtherSubbranches = False
			conceptNeuronTarget = sentenceConceptNodeList[wTarget]
			addPredictiveSequenceToNeuron(conceptNeuronTarget, sentenceIndex, sentenceConceptNodeList, conceptNeuronTarget.dendriticTree, dendriticBranchMaxW, 0, expectFurtherSubbranches)
	
	#reset dendritic trees
	if(biologicalSimulationForward):
		for conceptNeuron in connectionTargetNeuronSet:
			resetDendriticTreeActivation(conceptNeuron)

	drawBiologicalSimulationStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)

			
def simulateBiologicalHFnetworkSequenceNodeTrainWrapper(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, connectionTargetNeuronSet):
	somaActivationFound = False
	if(biologicalSimulationForward):
		wSource = wTarget-1
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		conceptNeuronTarget = sentenceConceptNodeList[wTarget]
		activationTime = calculateActivationTimeSequence(wSource)
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)
	else:
		conceptNeuronTarget = sentenceConceptNodeList[wTarget]
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)
	
	return somaActivationFound


def simulateBiologicalHFnetworkSequenceNodeTrainForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet):
	if(vectoriseComputationCurrentDendriticInput):
		somaActivationFound = HFNLPpy_biologicalSimulationVectorised.simulateBiologicalHFnetworkSequenceNodeTrainParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)
	else:
		somaActivationFound = HFNLPpy_biologicalSimulationStandard.simulateBiologicalHFnetworkSequenceNodeTrainStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)			
	return somaActivationFound

def simulateBiologicalHFnetworkSequenceNodeTrainReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
	somaActivationFound = HFNLPpy_biologicalSimulationStandard.simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)
	return somaActivationFound
	
#independent method (does not need to be executed in order of wSource)
def simulateBiologicalHFnetworkSequenceNodeTrainForwardFull(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
	somaActivationFound = False
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	
	for wSource, conceptNeuronSource in enumerate(sentenceConceptNodeList):	#support for simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain:!biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	#orig for wSource in range(0, wTarget):
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		activationTime = calculateActivationTimeSequence(wSource)
		somaActivationFoundTemp = simulateBiologicalHFnetworkSequenceNodeTrainForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)
		if(wSource == len(sentenceConceptNodeList)-1):
			if(somaActivationFoundTemp):
				somaActivationFound = True
			
	for conceptNeuron in connectionTargetNeuronSet:
		resetDendriticTreeActivation(conceptNeuron)
		
	return somaActivationFound

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
	




#else if(biologicalSimulation:useDependencyParseTree):

def simulateBiologicalHFnetworkSequenceTrainSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations):
	connectionTargetNeuronSet = set()	#for posthoc network deactivation

	activationTime = 0
	if(identifySyntacticalDependencyRelations):
		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, None, activationTime, connectionTargetNeuronSet)		
	else:
		print("biologicalSimulation:identifySyntacticalDependencyRelations current implementation requires identifySyntacticalDependencyRelations")
		exit()
		#simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, None, activationTime, connectionTargetNeuronSet)

	#reset dendritic trees
	if(biologicalSimulationForward):
		for conceptNeuron in connectionTargetNeuronSet:
			resetDendriticTreeActivation(conceptNeuron)
	
	drawBiologicalSimulationStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
			

#def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPtargetNode, DPparentNode, activationTime, connectionTargetNeuronSet):
#	for CPsourceNode in SPtargetNode.CPgraphNodeSourceList:
#		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPsourceNode, SPtargetNode, activationTime-1, connectionTargetNeuronSet)
#	HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceSyntacticalBranchCPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPtargetNode, DPparentNode, activationTime, connectionTargetNeuronSet)

def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, DPparentNode, activationTime, connectionTargetNeuronSet):
	for DPdependentNode in DPgovernorNode.DPdependentList:
		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPdependentNode, DPgovernorNode, activationTime-1, connectionTargetNeuronSet)
	simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, DPparentNode, activationTime, connectionTargetNeuronSet)

#def simulateBiologicalHFnetworkSequenceSyntacticalBranchCPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPbranchHeadNode, DPparentNode, connectionTargetNeuronSet):
#	somaActivationFound = False
#	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
#		activationTime = 0
#		if(calculateNeuronActivationSyntacticalBranchCP(sentenceConceptNodeList, CPbranchHeadNode, CPbranchHeadNode, activationTime, connectionTargetNeuronSet)):
#			somaActivationFound = True	
#		conceptNode = sentenceConceptNodeList[SPbranchHeadNode.w]
#		resetDendriticTreeActivation(conceptNode)
#	else:
#		print("simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP error: requires biologicalSimulationEncodeSyntaxInDendriticBranchStructure")
#		exit()
#	if(not somaActivationFound):
#		addPredictiveSequenceToNeuronSyntacticalBranchCP(sentenceIndex, sentenceConceptNodeList, CPbranchHeadNode)
	
def simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchHeadNode, DPbranchHeadParentNode, activationTime, connectionTargetNeuronSet):
	if(DPbranchHeadParentNode is not None):	#treat DPbranchHeadParentNode as target
		somaActivationFound = False
		if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
			activationTime = 0
			if(biologicalSimulationForward):
				if(calculateNeuronActivationSyntacticalBranchDPforward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchHeadNode, DPbranchHeadParentNode, activationTime, connectionTargetNeuronSet)):
					somaActivationFound = True
			else:
				if(calculateNeuronActivationSyntacticalBranchDPreverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchHeadParentNode, DPbranchHeadParentNode, activationTime, connectionTargetNeuronSet)):
					somaActivationFound = True			
		else:
			contextConceptNodesList = []
			if(calculateNeuronActivationSyntacticalBranchDPflat(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchHeadNode, DPbranchHeadParentNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList)):
				somaActivationFound = True

		if(somaActivationFound):
			#if(printVerbose):
			print("somaActivationFound")
		else:
			print("!somaActivationFound: addPredictiveSequenceToNeuron")
			w = DPbranchHeadParentNode.w
			conceptNode = sentenceConceptNodeList[w]
			currentBranchIndex1 = 0
			if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
				addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNode, sentenceIndex, sentenceConceptNodeList, DPbranchHeadParentNode, conceptNode.dendriticTree, currentBranchIndex1)		
			else:
				dendriticBranchMaxW = len(contextConceptNodesList) - 1	#index of last predictive neuron in artificial contextConceptNodesList sequence (not index of target concept)
				expectFurtherSubbranches = True
				if(dendriticBranchMaxW == 0):
					expectFurtherSubbranches = False
				addPredictiveSequenceToNeuron(conceptNode, sentenceIndex, contextConceptNodesList, conceptNode.dendriticTree, dendriticBranchMaxW, currentBranchIndex1, expectFurtherSubbranches)

def calculateNeuronActivationSyntacticalBranchDPforward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet):
	somaActivationFound = False
	#DPgovernorNode is not used
	wTarget = DPbranchTargetNode.w	#not used (for draw only)	
	conceptNeuronTarget = sentenceConceptNodeList[wTarget]
	#somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainForwardFull(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)
	wSource = DPbranchSourceNode.w	#not used (for draw only)	
	conceptNeuronSource = sentenceConceptNodeList[wSource]
	somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)	#assumes simulateBiologicalHFnetworkSequenceNodeTrainForward was executed for contiguous wSource
	return somaActivationFound
	
def calculateNeuronActivationSyntacticalBranchDPreverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchTargetNode, DPgovernorNode, activationTime, connectionTargetNeuronSet):
	somaActivationFound = False
	wTarget = DPbranchTargetNode.w	#not used (for draw only)	
	conceptNodeTarget = sentenceConceptNodeList[wTarget]
	for DPdependentNode in DPgovernorNode.DPdependentList:
		if(calculateNeuronActivationSyntacticalBranchDPreverseLookup(sentenceIndex, sentenceConceptNodeList, DPbranchTargetNode, DPdependentNode, activationTime-1, connectionTargetNeuronSet)):
			somaActivationFound = True

		wSource = DPdependentNode.w	#not used (for draw only)			
		conceptNeuronSource = sentenceConceptNodeList[wSource]	#previousContextConceptNode
		if(HFNLPpy_biologicalSimulationStandard.simulateBiologicalHFnetworkSequenceNodeTrainStandardSpecificTarget(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNodeTarget)):
			somaActivationFound = True

	return somaActivationFound

def calculateNeuronActivationSyntacticalBranchDPflat(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList):
	somaActivationFound = False
	identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPbranchSourceNode, contextConceptNodesList)
	wTarget = DPbranchTargetNode.w
	conceptNeuronTarget = sentenceConceptNodeList[wTarget]
	#print("wTarget = ", wTarget)
	#print("conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
	if(biologicalSimulationForward):
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainForwardFull(networkConceptNodeDict, sentenceIndex, contextConceptNodesList, wTarget, conceptNeuronTarget)
		#wSource = DPbranchSourceNode.w
		#conceptNeuronSource = sentenceConceptNodeList[wSource]
		#somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainForward(networkConceptNodeDict, sentenceIndex, contextConceptNodesList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)	#assumes simulateBiologicalHFnetworkSequenceNodeTrainForward was executed for contiguous wSource)		
	else:
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodeTrainReverseLookup(networkConceptNodeDict, sentenceIndex, contextConceptNodesList, wTarget, conceptNeuronTarget)	
	return somaActivationFound

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPgovernorNode, contextConceptNodesList):
	wSource = DPgovernorNode.w
	conceptNeuronSource = sentenceConceptNodeList[wSource]
	contextConceptNodesList.append(conceptNeuronSource)
	for DPdependentNode in DPgovernorNode.DPdependentList:
		identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPdependentNode, contextConceptNodesList)
	

def addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNeuron, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, dendriticBranch, currentBranchIndex1):

	#headNode in DP = current conceptNode (so not encoded in dendritic tree)

	activationTime = 0	#unactivated
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)

	currentBranchIndex2 = 0
	if(len(dendriticBranch.subbranches) > 0): 	#ensure dendriticTree has a sufficient number of branches to store the SPgraph
		for DPdependentNodeIndex in range(len(DPgovernorNode.DPdependentList)):
			print("DPdependentNodeIndex = ", DPdependentNodeIndex)
			print("len(dendriticBranch.subbranches) = ", len(dendriticBranch.subbranches))
			DPdependentNode = DPgovernorNode.DPdependentList[DPdependentNodeIndex]
			dendriticBranchSub = dendriticBranch.subbranches[DPdependentNodeIndex]

			previousContextConceptNode = sentenceConceptNodeList[DPdependentNode.w]
			currentSequentialSegmentIndex = 0	#SyntacticalBranchDP/SyntacticalBranchSP biologicalSimulation implementation does not use local sequential segments (encode sequentiality in branch structure only)
			currentSequentialSegment = dendriticBranchSub.sequentialSegments[currentSequentialSegmentIndex]
			createNewConnection, existingSequentialSegmentInput = verifyCreateNewConnection(currentSequentialSegment, previousContextConceptNode)
			if(createNewConnection):
				newSequentialSegmentSegmentInputIndex = calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
				currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex, previousContextConceptNode)
				#currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
				if(preventGenerationOfDuplicateConnections):
					currentSequentialSegment.inputs[previousContextConceptNode.nodeName] = currentSequentialSegmentInput			
				else:
					currentSequentialSegment.inputs[newSequentialSegmentSegmentInputIndex] = currentSequentialSegmentInput
				addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)
			else:
				currentSequentialSegmentInput = existingSequentialSegmentInput

			if(len(DPdependentNode.DPdependentList) == 0):
				expectFurtherSubbranches = False
				currentSequentialSegmentInput.firstInputInSequence = True

			addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNeuron, sentenceIndex, sentenceConceptNodeList, DPdependentNode, dendriticBranchSub, currentBranchIndex1+1)
			currentBranchIndex2 += 1

def drawBiologicalSimulationStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):
	if(drawBiologicalSimulation):
		if(drawBiologicalSimulationDendriticTreeSentence):
			HFNLPpy_biologicalSimulationDrawSentence.clearHopfieldGraph()
			HFNLPpy_biologicalSimulationDrawSentence.drawHopfieldGraphSentence(sentenceConceptNodeList)
			print("HFNLPpy_biologicalSimulationDrawSentence.displayHopfieldGraph()")
			HFNLPpy_biologicalSimulationDrawSentence.displayHopfieldGraph()
		if(drawBiologicalSimulationDendriticTreeNetwork):
			HFNLPpy_biologicalSimulationDrawNetwork.clearHopfieldGraph()
			HFNLPpy_biologicalSimulationDrawNetwork.drawHopfieldGraphNetwork(networkConceptNodeDict)
			print("HFNLPpy_biologicalSimulationDrawNetwork.displayHopfieldGraph()")
			HFNLPpy_biologicalSimulationDrawNetwork.displayHopfieldGraph()
			
			
