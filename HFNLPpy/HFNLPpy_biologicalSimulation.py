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

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations


debugCalculateNeuronActivationParallel = False
if(debugCalculateNeuronActivationParallel):
	wSourceDebug = 5
	wTargetDebug = wSourceDebug+1

alwaysAddPredictionInputFromPreviousConcept = False
if(vectoriseComputation):
	alwaysAddPredictionInputFromPreviousConcept = True #ensures that simulateBiologicalHFnetworkSequenceNodeTrainParallel:conceptNeuronBatchIndexFound

drawBiologicalSimulationDendriticTreeSentence = True	#draw graph for sentence neurons and their dendritic tree
if(drawBiologicalSimulationDendriticTreeSentence):
	import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentence
drawBiologicalSimulationDendriticTreeNetwork = True	#draw graph for entire network (not just sentence)
if(drawBiologicalSimulationDendriticTreeNetwork):
	import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetwork
		
printVerbose = False

probabilityOfSubsequenceThreshold = 0.01	#FUTURE: calibrate depending on number of branches/sequentialSegments etc

subsequenceLengthCalibration = 1.0

numberOfHorizontalSubBranchesRequiredForActivation = 2	#calibrate
activationRepolarisationTime = 1	#calibrate

resetSequentialSegments = True




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
		if(vectoriseComputationCurrentDendriticInput):
			if(w < sentenceLength-1):	#do not propagate signal from last neuron in sentence
				wSource = w
				wTarget = w+1
				conceptNeuronSource = sentenceConceptNodeList[wSource]
				conceptNeuronTarget = sentenceConceptNodeList[wTarget]
				simulateBiologicalHFnetworkSequenceNodeTrainParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget)
		else:
			wTarget = w
			conceptNeuronTarget = sentenceConceptNodeList[wTarget]
			simulateBiologicalHFnetworkSequenceNodeTrain(sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)

	for w, conceptNeuron in enumerate(sentenceConceptNodeList):
		if(vectoriseComputationCurrentDendriticInput):
			resetDendriticTreeActivation(conceptNeuron)

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

def simulateBiologicalHFnetworkSequenceNodeTrainParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, w, conceptNeuron):
	
	#construct batch dendritic tree templates for parallel processing;
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
	vectorisedBranchActivationLevelBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationTimeBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationLevelBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationTimeBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
			
	#if(printVerbose):
	print("simulateBiologicalHFnetworkSequenceNodeTrainParallel: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName)

	#vectorisedBranchInputBatchList = []	#list of tensors for every branchIndex1	- filled by createDendriticTreeVectorised; each element is of shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]

	activationTime = calculateActivationTime(sentenceIndex)
	
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
	
	batchIndex = 0	#batchSampleIndex
	conceptNeuronBatchIndex = None
	conceptNeuronBatchIndexFound = False
	for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():
		#add target neuron to batch processing tensor
		#if(vectoriseComputationIndependentBranches):	#only coded algorithm
		conceptNeuronTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
		for connection in connectionList:
			#trigger all target synaptic inputs before parallel processing
			currentSequentialSegmentInput = connection.nodeTargetSequentialSegmentInput
			setVectorisedBranchActivation(conceptNeuronTarget, currentSequentialSegmentInput, activationTime)

			#if(debugCalculateNeuronActivationParallel):		
			#	if(wSource==wSourceDebug and w==wTargetDebug):
			#		for branchIndex1 in range(numberOfVerticalBranches):
			#			print("\t(wSource==wSourceDebug and wTarget==wTargetDebug): branchIndex1 = ", branchIndex1)
			#			print("\tconceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1] = ", conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1])

							
		for branchIndex1 in range(numberOfVerticalBranches):
			vectorisedBranchActivationLevelBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1])
			vectorisedBranchActivationTimeBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationTimeList[branchIndex1])
		if(targetConnectionConceptName == conceptNeuron.nodeName):
			conceptNeuronBatchIndex = batchIndex
			conceptNeuronBatchIndexFound = True
			#print("conceptNeuron.nodeName = ", conceptNeuron.nodeName)
			#print("conceptNeuronBatchIndex = ", conceptNeuronBatchIndex)
		batchIndex += 1
			
	for branchIndex1 in range(numberOfVerticalBranches):
		vectorisedBranchActivationLevelBatchList[branchIndex1] = tf.stack(vectorisedBranchActivationLevelBatchListList[branchIndex1])
		vectorisedBranchActivationTimeBatchList[branchIndex1] = tf.stack(vectorisedBranchActivationTimeBatchListList[branchIndex1])	

	if(debugCalculateNeuronActivationParallel):	
		if(wSource==wSourceDebug and w==wTargetDebug):
			for branchIndex1 in range(numberOfVerticalBranches):
				print("\t(wSource==wSourceDebug and wTarget==wTargetDebug): branchIndex1 = ", branchIndex1)
				print("\tvectorisedBranchActivationLevelBatchList[branchIndex1] = ", vectorisedBranchActivationLevelBatchList[branchIndex1])


	if(conceptNeuronBatchIndexFound):	#optimsation; only execute calculateNeuronActivationParallel if conceptNeuron input(s) are activated by conceptNeuronSource
		if(calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, activationTime, w, conceptNeuron, conceptNeuronBatchIndex, wSource)):
			somaActivationFound = True
	else:
		print("warning !conceptNeuronBatchIndexFound")
	
	if(somaActivationFound):
		#if(printVerbose):
		print("somaActivationFound")
	else:
		#if(printVerbose):
		print("!somaActivationFound: addPredictiveSequenceToNeuron")	
		addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex, conceptNeuron.dendriticTree, w, 0)
	
def setVectorisedBranchActivation(conceptNeuronTarget, currentSequentialSegmentInput, activationTime):
	
	currentSequentialSegment = currentSequentialSegmentInput.sequentialSegment
	currentBranch = currentSequentialSegment.branch

	branchIndex1 = currentBranch.branchIndex1
	branchIndex2 = currentBranch.branchIndex2 	#local horizontalBranchIndex (wrt horizontalBranchWidth)
	horizontalBranchIndex = currentBranch.horizontalBranchIndex	#absolute horizontalBranchIndex	#required by vectoriseComputationCurrentDendriticInput only
	currentSequentialSegmentIndex = currentSequentialSegment.sequentialSegmentIndex
	if(useSequentialSegmentInputActivationLevels):
		currentSequentialSegmentInputIndex = currentSequentialSegmentInput.sequentialSegmentInputIndex
			
	activationValue = generateSequentialSegmentInputActivationValue(currentSequentialSegmentInput)
	#print("activationValue = ", activationValue)
	
	if(useSequentialSegmentInputActivationLevels):
		if(verifyRepolarised(currentSequentialSegmentIndex, activationTime, currentSequentialSegmentInput.activationTime)):
			currentSequentialSegmentInput.activationLevel = True
			currentSequentialSegmentInput.activationTime = activationTime	
			vectorisedBranchActivationLevel = conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1]
			vectorisedBranchActivationTime = conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1]
			conceptNeuronTarget.vectorisedBranchActivationLevel[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, sequentialSegmentInputIndex].assign(activationValue)
			conceptNeuronTarget.vectorisedBranchActivationTime[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, sequentialSegmentInputIndex].assign(activationTime)	
	else:
		if(verifyRepolarised(currentSequentialSegmentIndex, activationTime, currentSequentialSegment.activationTime)):
			#print("setting currentSequentialSegment.activationLevel")
			currentSequentialSegment.activationLevel = True
			currentSequentialSegment.activationTime = activationTime
			conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationValue)
			conceptNeuronTarget.vectorisedBranchActivationTimeList[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationTime)

def generateSequentialSegmentInputActivationValue(currentSequentialSegmentInput):
	if(currentSequentialSegmentInput.firstInputInSequence):
		activationValue = 2
	else:
		activationValue = 1
	return activationValue
						
def simulateBiologicalHFnetworkSequenceNodeTrain(sentenceIndex, sentenceConceptNodeList, w, conceptNeuron):

	#if(printVerbose):
	print("simulateBiologicalHFnetworkSequenceNodeTrain: w = ", w, ", conceptNeuron = ", conceptNeuron.nodeName)

	activationTime = calculateActivationTime(sentenceIndex)
	
	somaActivationFound = False	#is conceptNeuron activated by its prior context?
	for w2 in range(0, w):
		previousConceptNeuron = sentenceConceptNodeList[w2]	#source neuron
		if(conceptNeuron.nodeName in previousConceptNeuron.targetConnectionDict):
			connectionList = previousConceptNeuron.targetConnectionDict[conceptNeuron.nodeName]
			for connection in connectionList:
				targetNeuron = connection.nodeTarget	#targetNeuron will be the same for all connection in connectionList (if targetConnectionConceptName == conceptNeuron)
				if(targetNeuron != conceptNeuron):
					print("simulateBiologicalHFnetworkSequenceNodeTrain error: (targetNeuron != conceptNeuron)")
					exit()

				#FUTURE: vectoriseComputationCurrentDendriticInput: perform parallel processing (add target concept synapse/sequentialSegment/branch to tensor)
				#print("calculateNeuronActivation")
				if(calculateNeuronActivation(connection, 0, targetNeuron.dendriticTree, activationTime)):
					somaActivationFound = True
					#if(printVerbose):
					#print("somaActivationFound")

	resetDendriticTreeActivation(conceptNeuron)
	
	if(somaActivationFound):
		#if(printVerbose):
		print("somaActivationFound")
	else:
		#if(printVerbose):
		print("!somaActivationFound: addPredictiveSequenceToNeuron")	
		addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex, conceptNeuron.dendriticTree, w, 0)
					
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

																					
def filledList(lst):
	result = False
	if(len(lst) > 0):
		result = True
	return result
		
def calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, activationTime, wTarget, conceptNeuronTarget, conceptNeuronBatchIndex, wSource=None):
	print("calculateNeuronActivationParallel:")
	
	#vectorisedBranchActivationLevelBatchList/vectorisedBranchActivationTimeBatchList: list of tensors for every branchIndex1 - each element is of shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments], each batch sample refers to a unique target concept
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 	#len(vectorisedBranchActivationLevelBatchList)
	
	vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious = (None, None)
	for branchIndex1 in reversed(range(numberOfVerticalBranches)):
			
		vectorisedBranchActivationLevelBatch = vectorisedBranchActivationLevelBatchList[branchIndex1]
		vectorisedBranchActivationTimeBatch = vectorisedBranchActivationTimeBatchList[branchIndex1]

		#print("\tbranchIndex1 = ", branchIndex1)
		#print("\tvectorisedBranchActivationLevelBatch = ", vectorisedBranchActivationLevelBatch)
		#print("\tvectorisedBranchActivationTimeBatch = ", vectorisedBranchActivationTimeBatch)
				
		#initialise sequential segments activation (shape: [batchSize, numberOfHorizontalBranches, horizontalBranchWidth]);
		vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationTimeBatchSequentialSegments = calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatch.shape, vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious)

		#if(debugCalculateNeuronActivationParallel):
		#	if(wSource==wSourceDebug and wTarget==wTargetDebug):
		#		print("\t\t(wSource==wSourceDebug and wTarget==wTargetDebug): branchIndex1 = ", branchIndex1)
		#		print("\t\tinitial vectorisedBranchActivationLevelBatchSequentialSegments[conceptNeuronBatchIndex] = ", vectorisedBranchActivationLevelBatchSequentialSegments[conceptNeuronBatchIndex])

		for sequentialSegmentIndex in range(numberOfBranchSequentialSegments):
			vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatch[:, :, :, sequentialSegmentIndex]
			vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatch[:, :, :, sequentialSegmentIndex]
			#print("vectorisedBranchActivationLevelBatchSequentialSegment.shape = ", vectorisedBranchActivationLevelBatchSequentialSegment.shape)
			#print("vectorisedBranchActivationLevelBatchSequentialSegments.shape = ", vectorisedBranchActivationLevelBatchSequentialSegments.shape)
			vectorisedBranchActivationLevelBatchSequentialSegments = tf.add(vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationLevelBatchSequentialSegment)	#note if firstSequentialSegmentInSequence (ie sequentialSegmentActivationLevel=2), and higher branch input is zero, then sequential segment will still activate
			vectorisedBranchActivationTimeBatchSequentialSegments = tf.add(vectorisedBranchActivationTimeBatchSequentialSegments, vectorisedBranchActivationTimeBatchSequentialSegment)
			vectorisedBranchActivationTimeBatchSequentialSegments = tf.cast(tf.greater(vectorisedBranchActivationLevelBatchSequentialSegments, 1), tf.float32)	#or greater_equal(vectorisedBranchActivationLevelBatchSequentialSegments, 2)
			vectorisedBranchActivationTimeBatchSequentialSegments = tf.cast(tf.greater(vectorisedBranchActivationTimeBatchSequentialSegments, 1), tf.float32)	#or greater_equal(vectorisedBranchActivationLevelBatchSequentialSegments, 2)
		
		if(debugCalculateNeuronActivationParallel):	
			if(wSource==wSourceDebug and wTarget==wTargetDebug):
				print("\t\t(wSource==wSourceDebug and wTarget==wTargetDebug): branchIndex1 = ", branchIndex1)
				print("\t\tvectorisedBranchActivationLevelBatchSequentialSegments[conceptNeuronBatchIndex] = ", vectorisedBranchActivationLevelBatchSequentialSegments[conceptNeuronBatchIndex])
								
		vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious = vectorisedBranchActivationLevelBatchSequentialSegments
		vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious = vectorisedBranchActivationTimeBatchSequentialSegments
		
	vectorisedSomaActivationLevel = tf.squeeze(vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, axis=[1,2])	#size:batchSize
	vectorisedSomaActivationTime = tf.squeeze(vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious, axis=[1,2]) #size:batchSize
	
	#print("vectorisedSomaActivationLevel.shape = ", vectorisedSomaActivationLevel.shape)
	#print("vectorisedSomaActivationLevel = ", vectorisedSomaActivationLevel)
	
	somaActivationFound = vectorisedSomaActivationLevel[conceptNeuronBatchIndex].numpy()
	#print("somaActivationFound = ", somaActivationFound)
	somaActivationFound = bool(somaActivationFound)
	
	return somaActivationFound


def calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatchShape, vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious):
	#initialise sequential segments activation;
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 
	if(branchIndex1 == numberOfVerticalBranches-1):
		#highest branch in dendritic tree (initialise activation to true)
		vectorisedBranchActivationLevelBatchSequentialSegments = tf.ones((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2]))	#use every dimension except sequentialSegmentIndex 
		vectorisedBranchActivationTimeBatchSequentialSegments = tf.ones((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2])) 	#use every dimension except sequentialSegmentIndex
	else:
		#intermediary branch in dendritic tree (initialise activation to that of higher branch)
		vectorisedBranchActivationLevelBatchSequentialSegmentsPreviousSummed = tf.reduce_sum(vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, axis=2)
		vectorisedBranchActivationTimeBatchSequentialSegmentsPreviousSummed = tf.reduce_sum(vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious, axis=2)
		vectorisedBranchActivationLevelBatchSequentialSegments = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentsPreviousSummed, numberOfHorizontalSubBranchesRequiredForActivation)
		vectorisedBranchActivationTimeBatchSequentialSegments = tf.greater_equal(vectorisedBranchActivationTimeBatchSequentialSegmentsPreviousSummed, numberOfHorizontalSubBranchesRequiredForActivation)
		vectorisedBranchActivationLevelBatchSequentialSegments = tf.cast(vectorisedBranchActivationLevelBatchSequentialSegments, tf.float32)
		vectorisedBranchActivationTimeBatchSequentialSegments = tf.cast(vectorisedBranchActivationLevelBatchSequentialSegments, tf.float32)
		numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(branchIndex1, numberOfBranches2)
		#print("vectorisedBranchActivationLevelBatchSequentialSegments.shape = ", vectorisedBranchActivationLevelBatchSequentialSegments.shape)
		#print("numberOfHorizontalBranches = ", numberOfHorizontalBranches)
		#print("horizontalBranchWidth = ", horizontalBranchWidth)
		vectorisedBranchActivationLevelBatchSequentialSegments = sliceReshapeExpandDims(vectorisedBranchActivationLevelBatchSequentialSegments, horizontalBranchWidth, axis=-1)	#OLD: tf.reshape(vectorisedBranchActivationLevelBatchSequentialSegments, [vectorisedBranchActivationLevelBatchSequentialSegments.shape[0], numberOfHorizontalBranches, horizontalBranchWidth])
		vectorisedBranchActivationTimeBatchSequentialSegments = sliceReshapeExpandDims(vectorisedBranchActivationTimeBatchSequentialSegments, horizontalBranchWidth, axis=-1)	#OLD: tf.reshape(vectorisedBranchActivationTimeBatchSequentialSegments, [vectorisedBranchActivationTimeBatchSequentialSegments.shape[0], numberOfHorizontalBranches, horizontalBranchWidth])
	return vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationTimeBatchSequentialSegments

def sliceReshapeExpandDims(t, numberOfSlices, axis):
	sliceList = []
	#print("t.shape = ", t.shape)
	for sliceIndex in range(numberOfSlices):
		indices = tf.range(sliceIndex, t.shape[axis], delta=numberOfSlices)
		#print("indices = ", indices)
		#print("indices.shape = ", indices.shape)
		tSlice = tf.gather(t, indices, axis=axis)
		#print("tSlice.shape = ", tSlice.shape)
		sliceList.append(tSlice)
	tModified = tf.stack(sliceList, axis=-1)
	#print("tModified.shape = ", tModified.shape)
	return tModified
	
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
		if(numberOfBranch2active >= numberOfHorizontalSubBranchesRequiredForActivation):	#must conform with branch merge method *
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
						if(verifyRepolarised(currentSequentialSegmentIndex, activationTime, sequentialSegmentActivationTime)):	#ensure that the segment isnt in a repolarisation state (ie it can be activated)
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
	
def resetDendriticTreeActivation(conceptNeuron):
	resetBranchActivation(conceptNeuron.dendriticTree)
	if(vectoriseComputationCurrentDendriticInput):
		conceptNeuron.vectorisedBranchActivationLevelList, conceptNeuron.vectorisedBranchActivationTimeList = createBatchDendriticTreeVectorised(batched=False)	#rezero tensors by regenerating them 
	
def resetBranchActivation(currentBranch):

	currentBranch.activationLevel = False
	for sequentialSegment in currentBranch.sequentialSegments:
		sequentialSegment.activationLevel = False
		if(useSequentialSegmentInputActivationLevels):
			for sequentialSegmentInput in sequentialSegment.inputs:
				sequentialSegmentInput.activationLevel = False
									
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
	
def verifyRepolarised(currentSequentialSegmentIndex, activationTime, sequentialSegmentActivationTimePrevious):
	repolarised = False
	if(currentSequentialSegmentIndex == 0):
		repolarised = True	#CHECKTHIS: do not require repolarisation time for first sequential segment in branch
	else:
		if(activationTime > sequentialSegmentActivationTimePrevious+activationRepolarisationTime):
			repolarised = True
	return repolarised
	
