"""HFNLPpy_biologicalSimulationSyntacticalGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Syntactical Graph

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_biologicalSimulationNode import *
import HFNLPpy_hopfieldOperations
import HFNLPpy_biologicalSimulation
import HFNLPpy_biologicalSimulationGenerate
if(vectoriseComputation):
	import HFNLPpy_biologicalSimulationPropagateVectorised
else:
	import HFNLPpy_biologicalSimulationPropagateStandard

printVerbose = False


biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False	#incomplete - requires dendriticTree and syntacticalTree branches to have matching number of subnodes (2)	#speculative: directly encode precalculated syntactical structure in dendritic branches (rather than deriving syntax from commonly used dendritic subsequence encodings)	#requires useDependencyParseTree


def simulateBiologicalHFnetworkSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations):
	simulateBiologicalHFnetworkSequenceTrainSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations)					

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
	
	HFNLPpy_biologicalSimulation.drawBiologicalSimulationStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
			

def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPtargetNode, DPparentNode, activationTime, connectionTargetNeuronSet):
	for CPsourceNode in SPtargetNode.CPgraphNodeSourceList:
		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPsourceNode, SPtargetNode, activationTime-1, connectionTargetNeuronSet)
	simulateBiologicalHFnetworkSequenceSyntacticalBranchCPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPtargetNode, DPparentNode, activationTime, connectionTargetNeuronSet)

def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, DPparentNode, activationTime, connectionTargetNeuronSet):
	for DPdependentNode in DPgovernorNode.DPdependentList:
		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPdependentNode, DPgovernorNode, activationTime-1, connectionTargetNeuronSet)
	simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, DPparentNode, activationTime, connectionTargetNeuronSet)

def simulateBiologicalHFnetworkSequenceSyntacticalBranchCPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPbranchHeadNode, DPparentNode, activationTime, connectionTargetNeuronSet):
	somaActivationFound = False
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		activationTime = 0
		if(calculateNeuronActivationSyntacticalBranchCP(sentenceConceptNodeList, CPbranchHeadNode, CPbranchHeadNode, activationTime, connectionTargetNeuronSet)):
			somaActivationFound = True  
	else:
		print("simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP error: requires biologicalSimulationEncodeSyntaxInDendriticBranchStructure")
		exit()
	if(not somaActivationFound):
		addPredictiveSequenceToNeuronSyntacticalBranchCP(sentenceIndex, sentenceConceptNodeList, CPbranchHeadNode)
	
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
			w = DPbranchHeadParentNode.w
			conceptNode = sentenceConceptNodeList[w]
			currentBranchIndex1 = 0
			if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
				print("!somaActivationFound: addPredictiveSequenceToNeuronSyntacticalBranchDP")
				addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNode, sentenceIndex, sentenceConceptNodeList, DPbranchHeadParentNode, conceptNode.dendriticTree, currentBranchIndex1)		
			else:
				print("!somaActivationFound: addPredictiveSequenceToNeuron")
				dendriticBranchMaxW = len(contextConceptNodesList) - 1	#index of last predictive neuron in artificial contextConceptNodesList sequence (not index of target concept)
				expectFurtherSubbranches = True
				if(dendriticBranchMaxW == 0):
					expectFurtherSubbranches = False
				HFNLPpy_biologicalSimulationGenerate.addPredictiveSequenceToNeuron(conceptNode, sentenceIndex, contextConceptNodesList, conceptNode.dendriticTree, dendriticBranchMaxW, currentBranchIndex1, expectFurtherSubbranches)

def calculateNeuronActivationSyntacticalBranchDPforward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet):
	somaActivationFound = False
	#DPgovernorNode is not used
	wTarget = DPbranchTargetNode.w	#not used (for draw only)	
	conceptNeuronTarget = sentenceConceptNodeList[wTarget]
	#somaActivationFound = HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceNodeTrainForwardFull(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)
	wSource = DPbranchSourceNode.w	#not used (for draw only)	
	conceptNeuronSource = sentenceConceptNodeList[wSource]
	somaActivationFound = HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceNodeTrainForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)	#assumes simulateBiologicalHFnetworkSequenceNodeTrainForward was executed for contiguous wSource
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
		if(HFNLPpy_biologicalSimulationPropagateStandard.simulateBiologicalHFnetworkSequenceNodeTrainPropagateSpecificTarget(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNodeTarget)):
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
		somaActivationFound = HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceNodeTrainForwardFull(networkConceptNodeDict, sentenceIndex, contextConceptNodesList, wTarget, conceptNeuronTarget)
		#wSource = DPbranchSourceNode.w
		#conceptNeuronSource = sentenceConceptNodeList[wSource]
		#somaActivationFound = HFNLPpy_biologicalSimulation.simulateBiologicalHFnetworkSequenceNodeTrainForward(networkConceptNodeDict, sentenceIndex, contextConceptNodesList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)	#assumes simulateBiologicalHFnetworkSequenceNodeTrainForward was executed for contiguous wSource)		
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
			createNewConnection, existingSequentialSegmentInput = HFNLPpy_biologicalSimulationGenerate.verifyCreateNewConnection(currentSequentialSegment, previousContextConceptNode)
			if(createNewConnection):
				newSequentialSegmentSegmentInputIndex = HFNLPpy_biologicalSimulationGenerate.calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
				currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex, previousContextConceptNode)
				#currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
				if(preventGenerationOfDuplicateConnections):
					currentSequentialSegment.inputs[previousContextConceptNode.nodeName] = currentSequentialSegmentInput			
				else:
					currentSequentialSegment.inputs[newSequentialSegmentSegmentInputIndex] = currentSequentialSegmentInput
				HFNLPpy_biologicalSimulationGenerate.addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)
			else:
				currentSequentialSegmentInput = existingSequentialSegmentInput

			if(len(DPdependentNode.DPdependentList) == 0):
				expectFurtherSubbranches = False
				currentSequentialSegmentInput.firstInputInSequence = True

			addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNeuron, sentenceIndex, sentenceConceptNodeList, DPdependentNode, dendriticBranchSub, currentBranchIndex1+1)
			currentBranchIndex2 += 1

