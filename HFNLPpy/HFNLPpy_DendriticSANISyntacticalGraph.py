"""HFNLPpy_DendriticSANISyntacticalGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI Syntactical Graph

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations
from HFNLPpy_DendriticSANIGlobalDefs import *
from HFNLPpy_DendriticSANINode import *
import HFNLPpy_DendriticSANI
import HFNLPpy_DendriticSANIGenerate
if(vectoriseComputation):
	import HFNLPpy_DendriticSANIPropagateVectorised
else:
	import HFNLPpy_DendriticSANIPropagateStandard

printVerbose = False

debugDrawAfterAddPredictiveSequence = True
	

def trainBiologicalHFnetworkSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations):
	simulateBiologicalHFnetworkSequenceTrainSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations)					

def simulateBiologicalHFnetworkSequenceTrainSP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, identifySyntacticalDependencyRelations):
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	contextConceptNodesList = []
	#if(not biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect and not biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearHierarchical):
	#	identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, SPgraphHeadNode, contextConceptNodesList)

	activationTime = 0
	if(identifySyntacticalDependencyRelations):
		simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList)		
	else:
		print("useAlgorithmDendriticSANIbiologicalSimulation:identifySyntacticalDependencyRelations current implementation requires identifySyntacticalDependencyRelations")
		exit()
		#simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPgraphHeadNode, activationTime, connectionTargetNeuronSet)

	#reset dendritic trees
	if(biologicalSimulationForward):
		for conceptNeuron in connectionTargetNeuronSet:
			resetDendriticTreeActivation(conceptNeuron)
	
	HFNLPpy_DendriticSANIDraw.drawDendriticSANIStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
		
def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList=None):
	somaActivationFound = False
	if(len(DPgovernorNode.DPdependentList) > 0):
		for DPdependentNode in DPgovernorNode.DPdependentList:
			#print("activationTime = ", activationTime)
			simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchDP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPdependentNode, activationTime-1, connectionTargetNeuronSet, contextConceptNodesList)
			if(simulateBiologicalHFnetworkSequenceSyntacticalBranchDPPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPdependentNode, DPgovernorNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList)):
				somaActivationFound = True
	else:
		somaActivationFound = True	#DPgovernorNode is leaf node (do not add predictive sequence)
	if(somaActivationFound):
		#if(printVerbose):
		print("somaActivationFound, DPbranchHeadNode = ", DPgovernorNode.word)
	else:
		#if(printVerbose):
		print("!somaActivationFound: simulateBiologicalHFnetworkSequenceSyntacticalBranchDPAdd; DPbranchHeadNode = ", DPgovernorNode.word)
		simulateBiologicalHFnetworkSequenceSyntacticalBranchDPAdd(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList)

def simulateBiologicalHFnetworkSequenceSyntacticalBranchDPPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList=None):
	print("simulateBiologicalHFnetworkSequenceSyntacticalBranchDPPropagate: DPbranchTargetNode = ", DPbranchTargetNode.word, ", DPbranchSourceNode = ", DPbranchSourceNode.word)
	somaActivationFound = False
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect):
		if(biologicalSimulationForward):
			if(calculateNeuronActivationSyntacticalBranchDPforward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet)):
				somaActivationFound = True
		else:
			if(calculateNeuronActivationSyntacticalBranchDPreverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchTargetNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet)):
				somaActivationFound = True			
	else:
		if(calculateNeuronActivationSyntacticalBranchDPlinear(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList)):
			somaActivationFound = True
	
	return somaActivationFound

def simulateBiologicalHFnetworkSequenceSyntacticalBranchDPAdd(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchHeadNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList=None):
	w = DPbranchHeadNode.w
	conceptNode = sentenceConceptNodeList[w]
	currentBranchIndex1 = 0
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect):
		addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNode, sentenceIndex, sentenceConceptNodeList, DPbranchHeadNode, conceptNode.dendriticTree, currentBranchIndex1)		
	else:	
		contextConceptNodesListLocal = []	
		if(biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearHierarchical):
			identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulationHierarchical(sentenceConceptNodeList, DPbranchHeadNode, contextConceptNodesListLocal)
		else:
			identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPbranchHeadNode, contextConceptNodesListLocal)			
		predictiveSequenceLength = len(contextConceptNodesListLocal)
		dendriticBranchMaxW = len(contextConceptNodesListLocal) - 1	#index of last predictive neuron in artificial contextConceptNodesListLocal sequence (index of target concept is not in contextConceptNodesListLocal)
		expectFurtherSubbranches = True
		if(dendriticBranchMaxW == 0):
			expectFurtherSubbranches = False
		HFNLPpy_DendriticSANIGenerate.addPredictiveSequenceToNeuron(conceptNode, sentenceIndex, contextConceptNodesListLocal, conceptNode.dendriticTree, predictiveSequenceLength, dendriticBranchMaxW, currentBranchIndex1, 0, expectFurtherSubbranches)

	if(debugDrawAfterAddPredictiveSequence):
		HFNLPpy_DendriticSANIDraw.drawDendriticSANIStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	#draw for debugging

	
def calculateNeuronActivationSyntacticalBranchDPforward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet):
	somaActivationFound = False
	wTarget = DPbranchTargetNode.w	#not used (for draw only)	
	conceptNeuronTarget = sentenceConceptNodeList[wTarget]
	wSource = DPbranchSourceNode.w	#not used (for draw only)	
	conceptNeuronSource = sentenceConceptNodeList[wSource]
	somaActivationFound = HFNLPpy_DendriticSANI.simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)	#assumes simulateBiologicalHFnetworkSequenceNodePropagateForward was executed for contiguous wSource
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
		if(HFNLPpy_DendriticSANIPropagateStandard.simulateBiologicalHFnetworkSequenceNodeTrainPropagateSpecificTarget(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNodeTarget)):
			somaActivationFound = True

	return somaActivationFound

	
def addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNeuron, sentenceIndex, sentenceConceptNodeList, DPgovernorNode, dendriticBranch, currentBranchIndex1):

	if(expectFirstBranchSequentialSegmentConnection):
		print("addPredictiveSequenceToNeuronSyntacticalBranchDP error: currently requires !expectFirstBranchSequentialSegmentConnection")
		exit()
		
	#headNode in DP = current conceptNode (so not encoded in dendritic tree)

	activationTime = 0	#unactivated
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)

	currentBranchIndex2 = 0
	if(len(dendriticBranch.subbranches) > 0): 	#ensure dendriticTree has a sufficient number of branches to store the SPgraph
		numberOfSubbranchesToConnect = len(DPgovernorNode.DPdependentList)
		if(numberOfSubbranchesToConnect > numberOfBranches2):
			print("addPredictiveSequenceToNeuronSyntacticalBranchCP error: (numberOfSubbranchesToConnect > numberOfBranches2): numberOfSubbranchesToConnect = ", numberOfSubbranchesToConnect)
			exit()
		for DPdependentNodeIndex in range(numberOfSubbranchesToConnect):
			#print("DPdependentNodeIndex = ", DPdependentNodeIndex)
			#print("len(dendriticBranch.subbranches) = ", len(dendriticBranch.subbranches))
			DPdependentNode = DPgovernorNode.DPdependentList[DPdependentNodeIndex]
			dendriticBranchSub = dendriticBranch.subbranches[DPdependentNodeIndex]

			previousContextConceptNode = sentenceConceptNodeList[DPdependentNode.w]
			currentSequentialSegmentIndex = 0	#SyntacticalBranchDP/SyntacticalBranchSP useAlgorithmDendriticSANIbiologicalSimulation implementation does not use local sequential segments (encode sequentiality in branch structure only)
			currentSequentialSegment = dendriticBranchSub.sequentialSegments[currentSequentialSegmentIndex]
			createNewConnection, existingSequentialSegmentInput = HFNLPpy_DendriticSANIGenerate.verifyCreateNewConnection(currentSequentialSegment, previousContextConceptNode)
			if(createNewConnection):
				if(performSummationOfSequentialSegmentInputsAcrossBranch):
					weight = sequentialSegmentMinActivationLevel * (numberOfHorizontalSubBranchesRequiredForActivation/numberOfSubbranchesToConnect)
					#print("previousContextConceptNode = ", previousContextConceptNode.nodeName, ", conceptNeuron = ", conceptNeuron.nodeName, ", weight = ", weight)
				else:
					weight = sequentialSegmentMinActivationLevel
				newSequentialSegmentSegmentInputIndex = HFNLPpy_DendriticSANIGenerate.calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
				currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex, previousContextConceptNode)
				#currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
				if(preventGenerationOfDuplicateConnections):
					currentSequentialSegment.inputs[previousContextConceptNode.nodeName] = currentSequentialSegmentInput			
				else:
					currentSequentialSegment.inputs[newSequentialSegmentSegmentInputIndex] = currentSequentialSegmentInput
				HFNLPpy_DendriticSANIGenerate.addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, useAlgorithmDendriticSANIbiologicalPrototype=False, weight=weight, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, useAlgorithmDendriticSANIbiologicalSimulation=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)
			else:
				currentSequentialSegmentInput = existingSequentialSegmentInput

			expectFurtherSubbranches = True
			if(len(DPdependentNode.DPdependentList) == 0):
				expectFurtherSubbranches = False
			if(len(dendriticBranchSub.subbranches) == 0):
				expectFurtherSubbranches = False
			if(not expectFurtherSubbranches):
				currentSequentialSegmentInput.firstInputInSequence = True

			addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNeuron, sentenceIndex, sentenceConceptNodeList, DPdependentNode, dendriticBranchSub, currentBranchIndex1+1)
			currentBranchIndex2 += 1



#biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinear:

def calculateNeuronActivationSyntacticalBranchDPlinear(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchSourceNode, DPbranchTargetNode, activationTime, connectionTargetNeuronSet, contextConceptNodesList):
	#note use of activationTime parameters requires allowNegativeActivationTimes
	somaActivationFound = False
	activationTime = len(contextConceptNodesList)	#set activationTime with respect to contextConceptNodesList (the order in which source node is activated when parsing tree)
	contextConceptNodesList.append(DPbranchSourceNode)	#contextConceptNodesList is currently only used to calculate activation time only (based on tree parse reverse order)
	#identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPbranchSourceNode, contextConceptNodesList)
	#print("len(contextConceptNodesList) = ", len(contextConceptNodesList))
	wTarget = DPbranchTargetNode.w
	conceptNeuronTarget = sentenceConceptNodeList[wTarget]
	#print("wTarget = ", wTarget)
	#print("conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
	if(biologicalSimulationForward):
		#somaActivationFound = HFNLPpy_DendriticSANI.simulateBiologicalHFnetworkSequenceNodePropagateForwardFull(networkConceptNodeDict, sentenceIndex, contextConceptNodesList, wTarget, conceptNeuronTarget)
		wSource = DPbranchSourceNode.w
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		somaActivationFound = HFNLPpy_DendriticSANI.simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)	#assumes simulateBiologicalHFnetworkSequenceNodePropagateForward was executed for contiguous wSource)
	else:
		print("calculateNeuronActivationSyntacticalBranchDPlinear requires biologicalSimulationForward")
		exit()
		#somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup(networkConceptNodeDict, sentenceIndex, contextConceptNodesList, wTarget, conceptNeuronTarget)	
	return somaActivationFound
	
def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPgovernorNode, contextConceptNodesList):
	#adds the nodes in reverse order of tree crawl to a linear contextConceptNodesList
	for DPdependentNode in DPgovernorNode.DPdependentList:
		identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulation(sentenceConceptNodeList, DPdependentNode, contextConceptNodesList)
		wSource = DPdependentNode.w
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		contextConceptNodesList.append(conceptNeuronSource)	

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulationHierarchical(sentenceConceptNodeList, DPgovernorNode, contextConceptNodesList):
	#adds the most distant nodes to the start of a linear contextConceptNodesList
	maximumBranchDepth = calculateMaximumBranchDepth(DPgovernorNode, 0)
	#print("maximumBranchDepth = ", maximumBranchDepth)
	for level in reversed(range(maximumBranchDepth+1)):
		identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulationHierarchicalRecurse(sentenceConceptNodeList, DPgovernorNode, contextConceptNodesList, level, 0)

def calculateMaximumBranchDepth(DPgovernorNode, level):
	maxDepth = level
	for DPdependentNode in DPgovernorNode.DPdependentList:
		depth = calculateMaximumBranchDepth(DPdependentNode, level+1)
		if(depth > maxDepth):
			maxDepth = depth
	return maxDepth

def identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulationHierarchicalRecurse(sentenceConceptNodeList, DPgovernorNode, contextConceptNodesList, levelToAdd, level):
	for DPdependentNode in DPgovernorNode.DPdependentList:
		identifyHopfieldGraphNodeSyntacticalBranchDPbiologicalSimulationHierarchicalRecurse(sentenceConceptNodeList, DPdependentNode, contextConceptNodesList, levelToAdd, level+1)
		if(level+1 == levelToAdd):
			wSource = DPdependentNode.w
			conceptNeuronSource = sentenceConceptNodeList[wSource]
			contextConceptNodesList.append(conceptNeuronSource)


#Constituency Parser (INCOMPLETE);

def simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPtargetNode, activationTime, connectionTargetNeuronSet):
	somaActivationFound = False
	if(len(SPtargetNode.CPgraphNodeSourceList) > 0):
		for CPsourceNode in SPtargetNode.CPgraphNodeSourceList:
			simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPsourceNode, activationTime-1, connectionTargetNeuronSet)
			if(simulateBiologicalHFnetworkSequenceSyntacticalBranchCPPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPsourceNode, CPtargetNode, activationTime, connectionTargetNeuronSet)):
				somaActivationFound = True
	else:
		somaActivationFound = True	#SPtargetNode is leaf node (do not add predictive sequence)
	if(somaActivationFound):
		#if(printVerbose):
		print("somaActivationFound, DPbranchHeadNode = ", CPtargetNode.word)
	else:
		#if(printVerbose):
		print("!somaActivationFound: simulateBiologicalHFnetworkSequenceSyntacticalBranchCPAdd, DPbranchHeadNode = ", CPtargetNode.word)	
		simulateBiologicalHFnetworkSequenceSyntacticalBranchCPAdd(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPtargetNode, DPparentNode, activationTime, connectionTargetNeuronSet)

def simulateBiologicalHFnetworkSequenceSyntacticalBranchCPPropagate(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPbranchSourceNode, CPbranchTargetNode, activationTime, connectionTargetNeuronSet):
	somaActivationFound = False
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect):
		if(calculateNeuronActivationSyntacticalBranchCP(sentenceConceptNodeList, CPbranchSourceNode, CPbranchTargetNode, activationTime, connectionTargetNeuronSet)):
			somaActivationFound = True  
	else:
		print("simulateBiologicalHFnetworkSequenceTrainSyntacticalBranchCP error: requires biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect")
		exit()
	return somaActivationFound

def simulateBiologicalHFnetworkSequenceSyntacticalBranchCPAdd(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPbranchHeadNode, activationTime, connectionTargetNeuronSet):
	addPredictiveSequenceToNeuronSyntacticalBranchCP(sentenceIndex, sentenceConceptNodeList, CPbranchHeadNode)
	if(debugDrawAfterAddPredictiveSequence):
		HFNLPpy_DendriticSANIDraw.drawDendriticSANIStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)

def addPredictiveSequenceToNeuronSyntacticalBranchCP(conceptNeuron, sentenceIndex, sentenceConceptNodeList, CPtargetNode, dendriticBranch, currentBranchIndex1):

	if(expectFirstBranchSequentialSegmentConnection):
		print("addPredictiveSequenceToNeuronSyntacticalBranchCP error: currently requires !expectFirstBranchSequentialSegmentConnection")
		exit()
		
	#headNode in DP = current conceptNode (so not encoded in dendritic tree)

	activationTime = 0	#unactivated
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)

	currentBranchIndex2 = 0
	if(len(dendriticBranch.subbranches) > 0): 	#ensure dendriticTree has a sufficient number of branches to store the SPgraph
		numberOfSubbranchesToConnect = len(CPtargetNode.CPgraphNodeSourceList)
		if(numberOfSubbranchesToConnect > numberOfBranches2):
			print("addPredictiveSequenceToNeuronSyntacticalBranchCP error: (numberOfSubbranchesToConnect > numberOfBranches2): numberOfSubbranchesToConnect = ", numberOfSubbranchesToConnect)
			exit()
		for CPsourceNodeIndex in range(numberOfSubbranchesToConnect):
			#print("CPsourceNodeIndex = ", CPsourceNodeIndex)
			#print("len(dendriticBranch.subbranches) = ", len(dendriticBranch.subbranches))
			CPsourceNode = CPtargetNode.CPgraphNodeSourceList[CPsourceNodeIndex]
			dendriticBranchSub = dendriticBranch.subbranches[CPsourceNodeIndex]

			previousContextConceptNode = sentenceConceptNodeList[CPsourceNode.w]
			currentSequentialSegmentIndex = 0	#SyntacticalBranchDP/SyntacticalBranchSP useAlgorithmDendriticSANIbiologicalSimulation implementation does not use local sequential segments (encode sequentiality in branch structure only)
			currentSequentialSegment = dendriticBranchSub.sequentialSegments[currentSequentialSegmentIndex]
			createNewConnection, existingSequentialSegmentInput = HFNLPpy_DendriticSANIGenerate.verifyCreateNewConnection(currentSequentialSegment, previousContextConceptNode)
			if(createNewConnection):
				if(performSummationOfSequentialSegmentInputsAcrossBranch):
					weight = sequentialSegmentMinActivationLevel * (numberOfHorizontalSubBranchesRequiredForActivation/numberOfSubbranchesToConnect)
					#print("weight = ", weight)
				else:
					weight = sequentialSegmentMinActivationLevel
				newSequentialSegmentSegmentInputIndex = HFNLPpy_DendriticSANIGenerate.calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
				currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex, previousContextConceptNode)
				#currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
				if(preventGenerationOfDuplicateConnections):
					currentSequentialSegment.inputs[previousContextConceptNode.nodeName] = currentSequentialSegmentInput			
				else:
					currentSequentialSegment.inputs[newSequentialSegmentSegmentInputIndex] = currentSequentialSegmentInput
				HFNLPpy_DendriticSANIGenerate.addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, useAlgorithmDendriticSANIbiologicalPrototype=False, weight=weight, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, useAlgorithmDendriticSANIbiologicalSimulation=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)
			else:
				currentSequentialSegmentInput = existingSequentialSegmentInput

			expectFurtherSubbranches = True
			if(len(CPsourceNode.CPgraphNodeSourceList) == 0):
				expectFurtherSubbranches = False
			if(len(dendriticBranchSub.subbranches) == 0):
				expectFurtherSubbranches = False
			if(not expectFurtherSubbranches):
				currentSequentialSegmentInput.firstInputInSequence = True
				
			addPredictiveSequenceToNeuronSyntacticalBranchDP(conceptNeuron, sentenceIndex, sentenceConceptNodeList, CPsourceNode, dendriticBranchSub, currentBranchIndex1+1)
			currentBranchIndex2 += 1
			
			
