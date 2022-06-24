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

alwaysAddPredictionInputFromPreviousConcept = False
if(vectoriseComputation):
	alwaysAddPredictionInputFromPreviousConcept = True #ensures that simulateBiologicalHFnetworkSequenceNodeTrainParallel:conceptNeuronBatchIndexFound

biologicalSimulationForward = True	#default mode	#required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic
if(not vectoriseComputation):
	biologicalSimulationForward = False	#orig implementation; simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup

drawBiologicalSimulation = True	#default: True
if(drawBiologicalSimulation):
	drawBiologicalSimulationDendriticTreeSentence = True	#default: True	#draw graph for sentence neurons and their dendritic tree
	if(drawBiologicalSimulationDendriticTreeSentence):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentence
	drawBiologicalSimulationDendriticTreeNetwork = True	#default: True	#draw graph for entire network (not just sentence)
	if(drawBiologicalSimulationDendriticTreeNetwork):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetwork
else:
	drawBiologicalSimulationDendriticTreeSentence = False
	drawBiologicalSimulationDendriticTreeNetwork = False

#if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):

#def simulateBiologicalHFnetworkSequenceSyntacticalBranchCPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, CPbranchHeadNode):
#	activationTime = calculateActivationTime(sentenceIndex)
#	somaActivationFound = False
#	if(calculateNeuronActivationSyntacticalBranchCP(sentenceConceptNodeList, CPbranchHeadNode, CPbranchHeadNode, activationTime)):
#		somaActivationFound = True	
#	conceptNode = sentenceConceptNodeList[SPbranchHeadNode.w]
#	resetDendriticTreeActivation(conceptNode)
#	if(not somaActivationFound):
#		addPredictiveSequenceToNeuronSyntacticalBranchCP(sentenceConceptNodeList, CPbranchHeadNode, sentenceIndex)
		
def simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchHeadNode):
	activationTime = calculateActivationTime(sentenceIndex)
	somaActivationFound = False
	if(calculateNeuronActivationSyntacticalBranchDP(sentenceConceptNodeList, DPbranchHeadNode, DPbranchHeadNode, activationTime)):
		somaActivationFound = True	
	conceptNode = sentenceConceptNodeList[DPbranchHeadNode.w]
	resetDendriticTreeActivation(conceptNode)
	if(not somaActivationFound):
		addPredictiveSequenceToNeuronSyntacticalBranchDP(sentenceConceptNodeList, DPbranchHeadNode, sentenceIndex, conceptNode, conceptNode.dendriticTree)

def calculateNeuronActivationSyntacticalBranchDP(sentenceIndex, sentenceConceptNodeList, DPgovernorNode, DPbranchHeadNode, activationTime):
	somaActivationFound = False
	conceptNode = sentenceConceptNodeList[DPbranchHeadNode.w]
	for DPdependentNode in DPgovernorNode.DPdependentList:
		previousContextConceptNode = sentenceConceptNodeList[DPdependentNode.w]
		calculateNeuronActivationSyntacticalBranchDP(sentenceIndex, sentenceConceptNodeList, DPdependentNode, DPbranchHeadNode, activationTime)
		
		if(conceptNeuron.nodeName in previousContextConceptNode.targetConnectionDict):
			connectionList = previousContextConceptNode.targetConnectionDict[conceptNeuron.nodeName]
			for connection in connectionList:
				targetNeuron = connection.nodeTarget	#targetNeuron will be the same for all connection in connectionList (if targetConnectionConceptName == conceptNeuron)
				if(targetNeuron != conceptNeuron):
					print("calculateNeuronActivationSyntacticalBranchDP error: (targetNeuron != conceptNeuron)")
					exit()

				if(calculateNeuronActivation(connection, 0, targetNeuron.dendriticTree, activationTime)):
					somaActivationFound = True
					
	return somaActivationFound
			
def addPredictiveSequenceToNeuronSyntacticalBranchDP(sentenceIndex, sentenceConceptNodeList, DPgovernorNode, currentBranchIndex1, conceptNeuron, dendriticBranch):

	activationTime = calculateActivationTime(sentenceIndex)
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
	
	#headNode in DP = current conceptNode (so not encoded in dendritic tree)
	currentBranchIndex2 = 0
	for DPdependentNodeIndex in range(len(DPdependentNode.DPdependentList)):
		DPdependentNode = DPdependentNode.DPdependentList[DPdependentNodeIndex]
		dendriticBranchSub = dendriticBranch.subbranches[DPdependentNodeIndex]
		
		previousContextConceptNode = sentenceConceptNodeList[DPdependentNode.w]
		currentSequentialSegmentIndex = 0	#SyntacticalBranchDP/SyntacticalBranchSP biologicalSimulation implementation does not use local sequential segments (encode sequentiality in branch structure only)
		currentSequentialSegment = dendriticBranchSub.sequentialSegments[currentSequentialSegmentIndex]
		newSequentialSegmentSegmentInputIndex = calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
		currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex)
		currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
		addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)

		addPredictiveSequenceToNeuronSyntacticalBranchDP(sentenceIndex, sentenceConceptNodeList, DPdependentNode, currentBranchIndex1+1, dendriticBranchSub)
		currentBranchIndex2 += 1

#else (!biologicalSimulationEncodeSyntaxInDendriticBranchStructure):

def simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):

	#cannot clear now as HFNLPpy_biologicalSimulationDrawSentence/HFNLPpy_biologicalSimulationDrawNetwork memory structure is not independent (diagnose reason for this);
	#if(drawBiologicalSimulationDendriticTreeSentence):
	#	HFNLPpy_biologicalSimulationDrawSentence.clearHopfieldGraph()
	#if(drawBiologicalSimulationDendriticTreeNetwork):
	#	HFNLPpy_biologicalSimulationDrawNetwork.clearHopfieldGraph()
	
	sentenceLength = len(sentenceConceptNodeList)
					
	for w in range(sentenceLength):
		searchForPrediction = False
		somaActivationFound = False
		if(biologicalSimulationForward):
			if(w < sentenceLength-1):	#do not propagate signal from last neuron in sentence
				searchForPrediction = True
				wSource = w
				wTarget = w+1
				conceptNeuronSource = sentenceConceptNodeList[wSource]
				conceptNeuronTarget = sentenceConceptNodeList[wTarget]
				if(vectoriseComputationCurrentDendriticInput):
					somaActivationFound = HFNLPpy_biologicalSimulationVectorised.simulateBiologicalHFnetworkSequenceNodeTrainParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget)
				else:
					somaActivationFound = HFNLPpy_biologicalSimulationStandard.simulateBiologicalHFnetworkSequenceNodeTrainStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget)					
		else:
			searchForPrediction = True
			wTarget = w
			conceptNeuronTarget = sentenceConceptNodeList[wTarget]
			somaActivationFound = HFNLPpy_biologicalSimulationStandard.simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup(sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)

		if(searchForPrediction):
			if(somaActivationFound):
				#if(printVerbose):
				print("somaActivationFound")
			else:
				#if(printVerbose):
				print("!somaActivationFound: addPredictiveSequenceToNeuron")
				addPredictiveSequenceToNeuron(conceptNeuronTarget, w, sentenceConceptNodeList, sentenceIndex, conceptNeuronTarget.dendriticTree, wTarget, 0)
		
	for w, conceptNeuron in enumerate(sentenceConceptNodeList):
		if(vectoriseComputationCurrentDendriticInput):
			resetDendriticTreeActivation(conceptNeuron)

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

						
		
def addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex, dendriticBranch, dendriticBranchMaxW, branchIndex1, expectFurtherSubbranches=True):
	
	activationTime = calculateActivationTime(sentenceIndex)
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
	numberOfWordsInSequence = len(sentenceConceptNodeList)
	
	#no prediction found for previous sequence; generate prediction for conceptNeuron (encode subsequences in dendrite)
	#print("addPredictiveSequenceToNeuron:")
	
	if(branchIndex1 > 0):
		#do not create (recursive) connection from conceptNode to conceptNode branchIndex1=0
		previousContextConceptNode = sentenceConceptNodeList[dendriticBranchMaxW]
		currentSequentialSegmentIndex = 0	#biologicalSimulation implementation does not currently use local sequential segments (encode sequentiality in branch structure only)
		currentSequentialSegment = dendriticBranch.sequentialSegments[currentSequentialSegmentIndex]
		newSequentialSegmentSegmentInputIndex = calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
		#print("addPredictiveSynapseToNeuron ", conceptNeuron.nodeName, " branchIndex1 = ", branchIndex1)
		currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex)
		currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
		addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)

	if(expectFurtherSubbranches):
		for subbranchIndex, subbranch in enumerate(dendriticBranch.subbranches):
			#lengthOfSubsequenceScale = random.uniform(0, 1)
			#dendriticSubBranchMaxW = int(lengthOfSubsequenceScale*dendriticBranchMaxW)	
			lengthOfSubsequence = int(np.random.exponential()*subsequenceLengthCalibration)	#the more proximal the previous context, the more likely to form a synapse
			#lengthOfSubsequence = max(1, lengthOfSubsequence)
			lengthOfSubsequence = lengthOfSubsequence + 1	#ensure >= 1
			if(alwaysAddPredictionInputFromPreviousConcept):
				if(branchIndex1 == 0):
					if(subbranchIndex == 0):
						lengthOfSubsequence = 1	#ensures lengthOfSubsequence=1 for at least one subbranch
			#print("lengthOfSubsequence = ", lengthOfSubsequence)
			dendriticSubBranchMaxW = dendriticBranchMaxW-lengthOfSubsequence
			#print("dendriticSubBranchMaxW = ", dendriticSubBranchMaxW)
			dendriticSubBranchMaxW = max(dendriticSubBranchMaxW, 0)	#ensure >=0 (if zero then subbranch.seqentialSegment.firstInputInSequence will be set to true)	
			if(dendriticSubBranchMaxW == 0):
				expectFurtherSubbranches = False			
			if(len(subbranch.subbranches) == 0):
				expectFurtherSubbranches = False
				#print("no further subbranches")
			addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex, subbranch, dendriticSubBranchMaxW, branchIndex1+1, expectFurtherSubbranches)
	else:
		if(branchIndex1 > 0):
			currentSequentialSegmentInput.firstInputInSequence = True
			#print("setting currentSequentialSegmentInput.firstInputInSequence, branchIndex1 = ", branchIndex1)

#adds predictive synapse such that subsequences occur in order
def addPredictiveSynapseToNeuron(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=False, biologicalSynapse=False, nodeTargetSequentialSegmentInput=None):
	HFNLPpy_hopfieldOperations.addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalPrototype=biologicalPrototype, weight=weight, subsequenceConnection=subsequenceConnection, contextConnection=contextConnection, contextConnectionSANIindex=contextConnectionSANIindex, biologicalSimulation=biologicalSimulation, biologicalSynapse=biologicalSynapse, nodeTargetSequentialSegmentInput=nodeTargetSequentialSegmentInput)

																								
def calculateNewSequentialSegmentInputIndex(currentSequentialSegment):
	newSequentialSegmentSegmentInputIndex = len(currentSequentialSegment.inputs)
	newSequentialSegmentSegmentInputIndex =+ 1	
	#print("newSequentialSegmentSegmentInputIndex = ", newSequentialSegmentSegmentInputIndex)
	return newSequentialSegmentSegmentInputIndex	

