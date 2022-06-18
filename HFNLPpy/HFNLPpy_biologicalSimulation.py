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
		for every sequential synapse in segment (1+, sequentially dependent):
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


# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
vectoriseComputation = False	#FUTURE: parallel processing for optimisation
if(vectoriseComputation):
	import tensorflow as tf
#import random

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations

drawBiologicalSimulationDendriticTreeSentence = True	#draw graph for sentence neurons and their dendritic tree
if(drawBiologicalSimulationDendriticTreeSentence):
	import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentence
drawBiologicalSimulationDendriticTreeNetwork = True	#draw graph for entire network (not just sentence)
if(drawBiologicalSimulationDendriticTreeNetwork):
	import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetwork
		
printVerbose = True

probabilityOfSubsequenceThreshold = 0.01	#FUTURE: calibrate depending on number of branches/sequentialSegments etc



numberOfHorizontalSubBranchesRequiredForActivation = 2	#calibrate
activationRepolarisationTime = 1	#calibrate

resetSequentialSegments = True



#if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):

#def simulateBiologicalHFnetworkSequenceSyntacticalBranchCPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, SPbranchHeadNode):
#	activationTime = calculateActivationTime(sentenceIndex)
#	somaActivationFound = False
#	if(calculateNeuronActivationSyntacticalBranchSP(sentenceConceptNodeList, SPbranchHeadNode, SPbranchHeadNode, activationTime)):
#		somaActivationFound = True	
#	conceptNode = sentenceConceptNodeList[SPbranchHeadNode.w]
#	resetBranchActivation(conceptNode.dendriticTree)
#	if(not somaActivationFound):
#		addPredictiveSequenceToNeuronSyntacticalBranchSP(sentenceConceptNodeList, SPbranchHeadNode, sentenceIndex)
		
def simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, DPbranchHeadNode):
	activationTime = calculateActivationTime(sentenceIndex)
	somaActivationFound = False
	if(calculateNeuronActivationSyntacticalBranchDP(sentenceConceptNodeList, DPbranchHeadNode, DPbranchHeadNode, activationTime)):
		somaActivationFound = True	
	conceptNode = sentenceConceptNodeList[DPbranchHeadNode.w]
	resetBranchActivation(conceptNode.dendriticTree)
	if(not somaActivationFound):
		addPredictiveSequenceToNeuronSyntacticalBranchDP(sentenceConceptNodeList, DPbranchHeadNode, sentenceIndex, conceptNode, conceptNode.dendriticTree)

def calculateNeuronActivationSyntacticalBranchDP(sentenceIndex, sentenceConceptNodeList, DPgovernorNode, DPbranchHeadNode, activationTime):
	somaActivationFound = False
	conceptNode = sentenceConceptNodeList[DPbranchHeadNode.w]
	for DPdependentNode in DPgovernorNode.DPdependentList:
		previousContextConceptNode = sentenceConceptNodeList[DPdependentNode.w]
		calculateNeuronActivationSyntacticalBranchDP(sentenceIndex, sentenceConceptNodeList, DPdependentNode, DPbranchHeadNode, activationTime)
		
		for targetConnectionConceptName, connectionList in previousContextConceptNode.targetConnectionDict.items():
			print("targetConnectionConceptName = ", targetConnectionConceptName)
			if(targetConnectionConceptName == conceptNode.nodeName):
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
					
	for w, conceptNeuron in enumerate(sentenceConceptNodeList):
		simulateBiologicalHFnetworkSequenceNodeTrain(sentenceIndex, sentenceConceptNodeList, w, conceptNeuron)

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
					
def simulateBiologicalHFnetworkSequenceNodeTrain(sentenceIndex, sentenceConceptNodeList, w, conceptNeuron):

	if(printVerbose):
		print("simulateBiologicalHFnetworkSequenceNodeTrain: w = ", w, ", conceptNeuron = ", conceptNeuron.nodeName)

	activationTime = calculateActivationTime(sentenceIndex)
	
	somaActivationFound = False	#is conceptNeuron activated by its prior context?
	for w2 in range(0, w):
		previousConceptNeuron = sentenceConceptNodeList[w2]	#source neuron
		for targetConnectionConceptName, connectionList in previousConceptNeuron.targetConnectionDict.items():
			if(targetConnectionConceptName == conceptNeuron.nodeName):
				for connection in connectionList:
					targetNeuron = connection.nodeTarget	#targetNeuron will be the same for all connection in connectionList (if targetConnectionConceptName == conceptNeuron)
					if(targetNeuron != conceptNeuron):
						print("simulateBiologicalHFnetworkSequenceNodeTrain error: (targetNeuron != conceptNeuron)")
						exit()

					#FUTURE: vectoriseComputation: perform parallel processing (add target concept synapse/sequentialSegment/branch to tensor)
					#print("calculateNeuronActivation")
					if(calculateNeuronActivation(connection, 0, targetNeuron.dendriticTree, activationTime)):
						somaActivationFound = True
						#if(printVerbose):
						print("somaActivationFound")

	resetBranchActivation(conceptNeuron.dendriticTree)
	
	if(not somaActivationFound):
		addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex, conceptNeuron.dendriticTree, w, 0)
					
def addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex, dendriticBranch, dendriticBranchMaxW, level):

	activationTime = calculateActivationTime(sentenceIndex)
	spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)
	numberOfWordsInSequence = len(sentenceConceptNodeList)
	
	#no prediction found for previous sequence; generate prediction for conceptNeuron (encode subsequences in dendrite)
	#print("addPredictiveSequenceToNeuron:")
	
	if(level > 0):
		#do not create (recursive) connection from conceptNode to conceptNode branchIndex1=0
		previousContextConceptNode = sentenceConceptNodeList[dendriticBranchMaxW]
		currentSequentialSegmentIndex = 0	#biologicalSimulation implementation does not currently use local sequential segments (encode sequentiality in branch structure only)
		currentSequentialSegment = dendriticBranch.sequentialSegments[currentSequentialSegmentIndex]
		newSequentialSegmentSegmentInputIndex = calculateNewSequentialSegmentInputIndex(currentSequentialSegment)
		currentSequentialSegmentInput = SequentialSegmentInput(conceptNeuron, currentSequentialSegment, newSequentialSegmentSegmentInputIndex)
		currentSequentialSegment.inputs.append(currentSequentialSegmentInput)
		addPredictiveSynapseToNeuron(previousContextConceptNode, conceptNeuron, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetSequentialSegmentInput=currentSequentialSegmentInput)

	if(dendriticBranchMaxW > 0):
		for subbranch in dendriticBranch.subbranches:
			#lengthOfSubsequenceScale = random.uniform(0, 1)
			#dendriticSubBranchMaxW = int(lengthOfSubsequenceScale*dendriticBranchMaxW)	
			lengthOfSubsequence = int(np.random.exponential())	#the more proximal the previous context, the more likely to form a synapse
			#lengthOfSubsequence = max(1, lengthOfSubsequence)
			lengthOfSubsequence = lengthOfSubsequence + 1	#ensure >= 1
			#print("lengthOfSubsequence = ", lengthOfSubsequence)
			dendriticSubBranchMaxW = dendriticBranchMaxW-lengthOfSubsequence
			if(dendriticSubBranchMaxW >= 0):
				#print("dendriticSubBranchMaxW = ", dendriticSubBranchMaxW)
				addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex, subbranch, dendriticSubBranchMaxW, level+1)
	else:
		if(level > 0):
			currentSequentialSegmentInput.firstInputInSequence = True

#adds predictive synapse such that subsequences occur in order
def addPredictiveSynapseToNeuron(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=False, biologicalSynapse=False, nodeTargetSequentialSegmentInput=None):

	HFNLPpy_hopfieldOperations.addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalPrototype=biologicalPrototype, weight=weight, subsequenceConnection=subsequenceConnection, contextConnection=contextConnection, contextConnectionSANIindex=contextConnectionSANIindex, biologicalSimulation=biologicalSimulation, biologicalSynapse=biologicalSynapse, nodeTargetSequentialSegmentInput=nodeTargetSequentialSegmentInput)

	
																				
def filledList(lst):
	result = False
	if(len(lst) > 0):
		result = True
	return result
		
def calculateNeuronActivation(connection, currentBranchIndex1, currentBranch, activationTime):
	
	activationFound = False
	targetConceptNeuron = connection.nodeTarget
		
	#calculate subbranch activations:
	subbranchesActive = False
	if(len(currentBranch.subbranches) > 0):
		numberOfBranch2active = 0
		for subbranch in currentBranch.subbranches:	
			subbranchActive = calculateNeuronActivation(connection, currentBranchIndex1+1, subbranch, activationTime)
			if(subbranchActive):
				numberOfBranch2active += 1
		if(numberOfBranch2active >= numberOfHorizontalSubBranchesRequiredForActivation):
			subbranchesActive = True
	else:
	 	subbranchesActive = True
		
	#calculate branch segment activations:
	for currentSequentialSegmentIndex, currentSequentialSegment in enumerate(currentBranch.sequentialSegments):
		sequentialSegmentActivationLevel = currentSequentialSegment.activationLevel
		sequentialSegmentActivationTime = currentSequentialSegment.activationTime
		if(currentSequentialSegmentIndex == 0):
			sequentialSegmentActivationLevel = True	#no sequential requirement @index0
		sequentialSegmentActivationLevelNew = False

		for currentSequentialSegmentInputIndex, currentSequentialSegmentInput in enumerate(currentSequentialSegment.inputs):
			if(connection.nodeTargetSequentialSegmentInput == currentSequentialSegmentInput):
				if(printVerbose):
					printIndentation(currentBranchIndex1+1)
					print("activate currentSequentialSegmentInput, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
				passSegmentActivationTimeTests = False
				if(currentSequentialSegmentInput.firstInputInSequence):
					passSegmentActivationTimeTests = True	#if input corresponds to first in sequence, then enforce no previous dendritic activation requirements
					subbranchesActive = True
				else:
					if(sequentialSegmentActivationLevel):	#previous sequential segment was activated
						if((currentSequentialSegmentIndex == 0) or (activationTime > sequentialSegmentActivationTime+activationRepolarisationTime)):	#ensure that the segment isnt in a repolarisation state (ie it can be activated)
							#if(activationTime > previousVerticalBranchActivationTime):	#guaranteed
							if(subbranchesActive):
								passSegmentActivationTimeTests = True	#previous (ie more distal) branch was active
				if(passSegmentActivationTimeTests):
					sequentialSegmentActivationLevelNew = True
					sequentialSegmentActivationTimeNew = activationTime

		if(sequentialSegmentActivationLevelNew):
			if(printVerbose):
				printIndentation(currentBranchIndex1+1)
				print("activate currentSequentialSegment, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
			if(resetSequentialSegments):
				if(currentSequentialSegmentIndex == 0):
					resetBranchActivation(currentBranch)
					numberOfSequentialSegmentsActive = 0
			numberOfSequentialSegmentsActive += 1	#CHECKTHIS
			sequentialSegmentActivationLevel = True
			sequentialSegmentActivationTime = activationTime
			currentSequentialSegment.activationLevel = sequentialSegmentActivationLevel
			currentSequentialSegment.activationTime = sequentialSegmentActivationTime

	sequentialSegmentActivationLevelLast = sequentialSegmentActivationLevel
	sequentialSegmentActivationTimeLast = sequentialSegmentActivationTime
	#sequentialSegmentActivationLevelLastNew = sequentialSegmentActivationLevelLast
	sequentialSegmentsActive = False
	if(sequentialSegmentActivationLevelLast):
		if(printVerbose):
			printIndentation(currentBranchIndex1+1)
			print("activate currentBranch, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
		branch2ActivationLevel = sequentialSegmentActivationLevelLast	#activate branch2	#activate whole currentSequentialSegment
		branch2ActivationTime = sequentialSegmentActivationTimeLast
		currentBranch.activationLevel = branch2ActivationLevel
		currentBranch.activationTime = branch2ActivationTime
		sequentialSegmentsActive = True	

	if(subbranchesActive and sequentialSegmentsActive):
		if(printVerbose):
			printIndentation(currentBranchIndex1+1)
			print("activationFound")
		activationFound = True
			
	return activationFound							
	
def resetBranchActivation(currentBranch):

	currentBranch.activationLevel = 0
	for sequentialSegment in currentBranch.sequentialSegments:
		sequentialSegment.activationLevel = 0
		
	for subbranch in currentBranch.subbranches:	
		resetBranchActivation(subbranch)
	
def printIndentation(level):
	for indentation in range(level):
		print('\t', end='')

def calculateNewSequentialSegmentInputIndex(currentSequentialSegment):
	newSequentialSegmentSegmentInputIndex = len(currentSequentialSegment.inputs)
	newSequentialSegmentSegmentInputIndex =+ 1	
	#print("newSequentialSegmentSegmentInputIndex = ", newSequentialSegmentSegmentInputIndex)
	return newSequentialSegmentSegmentInputIndex
	
