"""HFNLPpy_biologicalSimulationPropagateVectorised.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Propagate Vectorised

"""



import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_biologicalSimulationGlobalDefs import *
from HFNLPpy_biologicalSimulationNode import *


#if(biologicalSimulationForward):	#required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic
if(vectoriseComputation):		#dynamic draw should use vectoriseComputation, as this activates all target neuron synapses of wSource simultaneously 
	if(updateNeuronObjectActivationLevels):
		drawBiologicalSimulationDynamic = False	#draw dynamic activation levels of biological simulation	#optional
		if(drawBiologicalSimulationDynamic):
			drawBiologicalSimulationDynamicPlot = True	#default: False
			drawBiologicalSimulationDynamicSave = False	#default: True	#save to file
			drawBiologicalSimulationDendriticTreeSentenceDynamic = True	#default: True	#draw graph for sentence neurons and their dendritic tree
			if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
				import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentenceDynamic
			drawBiologicalSimulationDendriticTreeNetworkDynamic = False	#default: True	#draw graph for entire network (not just sentence)
			if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
				import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetworkDynamic
	else:
		print("HFNLPpy_biologicalSimulationPropagateVectorised warning: updateNeuronObjectActivationLevels is required for vectoriseComputation:drawBiologicalSimulationDynamic (if drawBiologicalSimulationDynamic is required; either enable updateNeuronObjectActivationLevels or disable vectoriseComputation)")
		drawBiologicalSimulationDynamic = False	#mandatory: False

printVerbose = False
printConnectionTargetActivations = False

debugCalculateNeuronActivationParallel = False	#requires !drawBiologicalSimulationDynamicHighlightNewActivations
if(debugCalculateNeuronActivationParallel):
	sentenceIndexDebug = 208	#1	#10	#397
	wSourceDebug = 3	#14
	wTargetDebug = 4	#8
	batchIndexOfWTargetDebug = None
else:
	wTargetDebug = None

#parameters only used for drawBiologicalSimulationDynamic: sentenceIndex, sentenceConceptNodeList
def simulateBiologicalHFnetworkSequenceNodePropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet):
	conceptNeuronSourceList = []
	conceptNeuronSourceList.append(conceptNeuronSource)
	return simulateBiologicalHFnetworkSequenceNodesPropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)
	
#parameters only used for drawBiologicalSimulationDynamic: wSource, sentenceIndex, sentenceConceptNodeList
def simulateBiologicalHFnetworkSequenceNodesPropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet):
	
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?

	#construct batch dendritic tree templates for parallel processing;
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
	vectorisedBranchActivationLevelBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationTimeBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationFlagBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationLevelBatchListListBuffer = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationTimeBatchListListBuffer = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing	
	vectorisedBranchActivationFlagBatchListListBuffer = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing	
	if(recordVectorisedBranchObjectList):
		vectorisedBranchObjectBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationLevelBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationTimeBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationFlagBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationLevelBatchListBuffer = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationTimeBatchListBuffer = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationFlagBatchListBuffer = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	if(recordVectorisedBranchObjectList):
		vectorisedBranchObjectBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	else:
		vectorisedBranchObjectBatchList = None
		
	batchNeuronsList = []	#preserve insertion order	#alternatively in recordVectorisedBranchObjectList; can lookup batchNeurons from vectorisedBranchObjectBatchList instead
	batchIndex = 0	#batchSampleIndex
	conceptNeuronBatchIndex = None
	conceptNeuronBatchIndexFound = False
	targetConnectionFound = False
	
	for conceptNeuronSource in conceptNeuronSourceList:

		if(printVerbose):
			print("simulateBiologicalHFnetworkSequenceNodesPropagateParallel: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
	
		if(updateNeuronObjectActivationLevels):
			conceptNeuronSource.activationLevel = objectAreaActivationLevelOn

		for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():

			#add target neuron to batch processing tensor
			#if(vectoriseComputationIndependentBranches):	#only coded algorithm
			conceptNeuronConnectionTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
			if(conceptNeuronConnectionTarget not in batchNeuronsList):
			
				connectionTargetNeuronSet.add(conceptNeuronConnectionTarget)
				batchNeuronsList.append(conceptNeuronConnectionTarget)

				if(debugCalculateNeuronActivationParallel):
					if(sentenceIndex == sentenceIndexDebug and wSource == wSourceDebug):
						if(conceptNeuronConnectionTarget.w == wTargetDebug):
							global batchIndexOfWTargetDebug
							batchIndexOfWTargetDebug = batchIndex
							print("batchIndex of wTargetDebug = ", batchIndex)

				targetConnectionFound = True
				if(targetConnectionConceptName == conceptNeuronTarget.nodeName):
					conceptNeuronBatchIndex = batchIndex
					conceptNeuronBatchIndexFound = True
					#print("conceptNeuronTarget.nodeName = ", conceptNeuronTarget.nodeName)
					#print("conceptNeuronBatchIndex = ", conceptNeuronBatchIndex)
				batchIndex += 1
				
				#create temporary vectorised buffers for conceptNeuronSource connection target input sequentialSegment candidate application;
				conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer, conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer, conceptNeuronConnectionTarget.vectorisedBranchActivationFlagListBuffer = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=False, storeSequentialSegmentInputActivationLevels=vectoriseComputionUseSequentialSegmentInputActivationLevels)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, {numberOfSequentialSegmentInputs}]
				
			#trigger all target synaptic inputs before parallel processing	
			for connection in connectionList:
				if(updateNeuronObjectActivationLevels):
					connection.activationLevel = objectAreaActivationLevelOn
				setVectorisedBranchActivation(conceptNeuronConnectionTarget, connection, activationTime)

	batchNeuronsList2 = []
	for conceptNeuronSource in conceptNeuronSourceList:
		for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():
			conceptNeuronConnectionTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
			if(conceptNeuronConnectionTarget not in batchNeuronsList2):
				batchNeuronsList2.append(conceptNeuronConnectionTarget)
				for branchIndex1 in range(numberOfVerticalBranches):
					vectorisedBranchActivationLevelBatchListList[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationLevelList[branchIndex1])
					vectorisedBranchActivationTimeBatchListList[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationTimeList[branchIndex1])
					vectorisedBranchActivationFlagBatchListList[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationFlagList[branchIndex1])
					vectorisedBranchActivationLevelBatchListListBuffer[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1])
					vectorisedBranchActivationTimeBatchListListBuffer[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1])
					vectorisedBranchActivationFlagBatchListListBuffer[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationFlagListBuffer[branchIndex1])
					if(recordVectorisedBranchObjectList):
						vectorisedBranchObjectBatchListList[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchObjectList[branchIndex1])			

	for branchIndex1 in range(numberOfVerticalBranches):
		vectorisedBranchActivationLevelBatchList[branchIndex1] = tf.Variable(tf.stack(vectorisedBranchActivationLevelBatchListList[branchIndex1]))
		vectorisedBranchActivationTimeBatchList[branchIndex1] = tf.Variable(tf.stack(vectorisedBranchActivationTimeBatchListList[branchIndex1]))	
		vectorisedBranchActivationFlagBatchList[branchIndex1] = tf.Variable(tf.stack(vectorisedBranchActivationFlagBatchListList[branchIndex1]))	
		vectorisedBranchActivationLevelBatchListBuffer[branchIndex1] = tf.stack(vectorisedBranchActivationLevelBatchListListBuffer[branchIndex1])
		vectorisedBranchActivationTimeBatchListBuffer[branchIndex1] = tf.stack(vectorisedBranchActivationTimeBatchListListBuffer[branchIndex1])		
		vectorisedBranchActivationFlagBatchListBuffer[branchIndex1] = tf.stack(vectorisedBranchActivationFlagBatchListListBuffer[branchIndex1])				
		#print("vectorisedBranchActivationLevelBatchListListBuffer[branchIndex1] = ", vectorisedBranchActivationLevelBatchListListBuffer[branchIndex1])
		if(recordVectorisedBranchObjectList):
			if(not emptyList(vectorisedBranchObjectBatchListList[branchIndex1])):
				vectorisedBranchObjectBatchList[branchIndex1] = np.stack(vectorisedBranchObjectBatchListList[branchIndex1])
				#print("vectorisedBranchObjectBatchList[branchIndex1] = ", vectorisedBranchObjectBatchList[branchIndex1])	
		
	#if(debugCalculateNeuronActivationParallel):	
	#	if(wSource==wSourceDebug and wTarget==wTargetDebug):
	#		for branchIndex1 in range(numberOfVerticalBranches):
	#			print("\t(wSource==wSourceDebug and wTarget==wTargetDebug): branchIndex1 = ", branchIndex1)
	#			print("\tvectorisedBranchActivationLevelBatchList[branchIndex1] = ", vectorisedBranchActivationLevelBatchList[branchIndex1])

	if(targetConnectionFound):
		if(conceptNeuronBatchIndexFound or not onlyPropagateIfConceptNeuronTargetActivatedByConceptNeuronSourceVectorised):	#orig optimisation; only execute calculateNeuronActivationParallel if conceptNeuronTarget input(s) are activated by conceptNeuronSource
			if(calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationFlagBatchList, vectorisedBranchActivationLevelBatchListBuffer, vectorisedBranchActivationTimeBatchListBuffer, vectorisedBranchActivationFlagBatchListBuffer, vectorisedBranchObjectBatchList, activationTime, wTarget, conceptNeuronTarget, conceptNeuronBatchIndex, batchNeuronsList, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)):
				somaActivationFound = True
		else:
			print("warning !conceptNeuronBatchIndexFound")
	#else:
	#	print("warning !targetConnectionFound")
	
	#save updated activations (ideally these should be able to be dynamically updated by calculateNeuronActivationParallel; store tensors (memory/reference) in a bulk/stacked tensor, write to the bulk tensor and have the individual tensors updated)
	for batchIndex, batchNeuron in enumerate(batchNeuronsList):
		for branchIndex1 in range(numberOfVerticalBranches):
			#iterating over batchSize to save tensors is slow and may require optimisation
			batchNeuron.vectorisedBranchActivationLevelList[branchIndex1] = tf.Variable(vectorisedBranchActivationLevelBatchList[branchIndex1][batchIndex])
			batchNeuron.vectorisedBranchActivationTimeList[branchIndex1] = tf.Variable(vectorisedBranchActivationTimeBatchList[branchIndex1][batchIndex])
			batchNeuron.vectorisedBranchActivationFlagList[branchIndex1] = tf.Variable(vectorisedBranchActivationFlagBatchList[branchIndex1][batchIndex])
			
	for conceptNeuronSource in conceptNeuronSourceList:
		resetSourceNeuronAfterActivation(conceptNeuronSource)

	return somaActivationFound
	
def emptyList(lst):
	result = False
	if(len(lst) == 0):
		result = True
	return result
		
def setVectorisedBranchActivation(conceptNeuronConnectionTarget, connection, activationTime):
	
	currentSequentialSegmentInput = connection.nodeTargetSequentialSegmentInput
	currentSequentialSegment = currentSequentialSegmentInput.sequentialSegment
	currentBranch = currentSequentialSegment.branch

	branchIndex1 = currentBranch.branchIndex1
	branchIndex2 = currentBranch.branchIndex2 	#local horizontalBranchIndex (wrt horizontalBranchWidth)
	horizontalBranchIndex = currentBranch.horizontalBranchIndex	#absolute horizontalBranchIndex	#required by vectoriseComputationCurrentDendriticInput only
	currentSequentialSegmentIndex = currentSequentialSegment.sequentialSegmentIndex
	if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
		currentSequentialSegmentInputIndex = currentSequentialSegmentInput.sequentialSegmentInputIndex
			
	activationValue = calculateVectorisedSequentialSegmentInputActivation(connection)
	#print("activationValue = ", activationValue)
	#print("activationTime = ", activationTime)
	#print("currentSequentialSegment.nodeName = ", currentSequentialSegment.nodeName)
	
	if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
		#if(verifyRepolarised(currentSequentialSegmentIndex, activationTime, currentSequentialSegmentInput.activationTime)):
		if(updateNeuronObjectActivationLevels):
			if(weightedSequentialSegmentInputs):
				currentSequentialSegmentInput.activationLevel = activationValue	#CHECKTHIS: or connection.weight (disregarding firstInputInSequence activation level modifier)
			else:
				currentSequentialSegmentInput.activationLevel = objectLocalActivationLevelOn	#True
			if(drawBiologicalSimulationDynamicHighlightNewActivations):
				currentSequentialSegmentInput.activationStateNew = True
			currentSequentialSegmentInput.activationTime = activationTime	
		activationFlags = vectorisedActivationTimeFlagDefault
		if(currentSequentialSegmentInput.firstInputInSequence):
			activationFlags = vectorisedActivationTimeFlagFirstInputInSequence
		conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationValue)
		conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationTime)	#not used (all inputs should have same activation time)
		conceptNeuronConnectionTarget.vectorisedBranchActivationFlagListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationFlags)	
	else:
		if(updateNeuronObjectActivationLevels):
			if(weightedSequentialSegmentInputs):
				currentSequentialSegmentInput.activationLevel = activationValue		#CHECKTHIS: or connection.weight (disregarding firstInputInSequence activation level modifier)
			else:
				currentSequentialSegmentInput.activationLevel = objectLocalActivationLevelOn	#True
			if(drawBiologicalSimulationDynamicHighlightNewActivations):
				currentSequentialSegmentInput.activationStateNew = True
			currentSequentialSegmentInput.activationTime = activationTime
		activationFlags = vectorisedActivationTimeFlagDefault
		if(performSummationOfSequentialSegmentInputs):
			summationOfSequentialSegmentInputs = conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].numpy()
			summationOfSequentialSegmentInputs = summationOfSequentialSegmentInputs + activationValue
			activationValue = summationOfSequentialSegmentInputs
			firstInputInSequenceExisting = conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].numpy()
			if(bool(firstInputInSequenceExisting)):
				activationFlags = vectorisedActivationTimeFlagFirstInputInSequence
		if(currentSequentialSegmentInput.firstInputInSequence):
			activationFlags = vectorisedActivationTimeFlagFirstInputInSequence
		#if(verifyRepolarised(currentSequentialSegment, activationTime)):	#do not perform this test for buffer (all inputs should have same activation time)
		#print("activationValue = ", activationValue)
		conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationValue)
		conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationTime)	#not used (all inputs should have same activation time)
		conceptNeuronConnectionTarget.vectorisedBranchActivationFlagListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationFlags)

def calculateVectorisedSequentialSegmentInputActivation(connection):
	activationValue = calculateInputActivationLevelVectorised(connection)	
	return activationValue

def calculateInputActivationLevelVectorised(connection):
	inputActivationLevel = objectLocalActivationLevelOff
	if(weightedSequentialSegmentInputs):
		inputActivationLevel = connection.weight
	else:
		inputActivationLevel = vectorisedActivationLevelOn
	return inputActivationLevel
						
								
def generateVectorisedSequentialSegmentInputActivationBias(currentSequentialSegmentInput):
	#artificially increase activation value of first inputs in sequence
	if(currentSequentialSegmentInput.firstInputInSequence):
		activationValue = vectorisedActivationLevelOnFirstInputInSequence
	else:
		activationValue = vectorisedActivationLevelOn
	return activationValue
						

#parameters only used for drawBiologicalSimulationDynamic: wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList
#does not currently support vectoriseComputionUseSequentialSegmentInputActivationLevels;
def calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationFlagBatchList, vectorisedBranchActivationLevelBatchListBuffer, vectorisedBranchActivationTimeBatchListBuffer, vectorisedBranchActivationFlagBatchListBuffer, vectorisedBranchObjectBatchList, activationTime, wTarget, conceptNeuronTarget, conceptNeuronBatchIndex, batchNeuronsList, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):
	
	somaActivationFound = False
	
	#print("calculateNeuronActivationParallel:")
		
	#vectorisedBranchActivationLevelBatchList/vectorisedBranchActivationTimeBatchList: list of tensors for every branchIndex1 - each element is of shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments], each batch sample refers to a unique target concept
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 	#len(vectorisedBranchActivationLevelBatchList)
	
	if(reversePropagationOrder):
		vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentPrevious, vectorisedBranchActivationFlagBatchSequentialSegmentPrevious = (None, None, None)
		
	#if(resetConnectionTargetNeuronDendriteAfterSequence):
	vectorisedBranchActivationStateBatchSequentialSegmentFinalNew = None

	branchSequence = range(numberOfVerticalBranches)
	if(reversePropagationOrder):
		branchSequence = reversed(branchSequence)

	for branchIndex1 in branchSequence:

		sequentialSegmentSequence = range(numberOfBranchSequentialSegments)	
		if(reversePropagationOrder):
			sequentialSegmentSequence = reversed(sequentialSegmentSequence)
					
		vectorisedBranchActivationLevelBatchBuffer = vectorisedBranchActivationLevelBatchListBuffer[branchIndex1]
		vectorisedBranchActivationTimeBatchBuffer = vectorisedBranchActivationTimeBatchListBuffer[branchIndex1]
		vectorisedBranchActivationFlagBatchBuffer = vectorisedBranchActivationFlagBatchListBuffer[branchIndex1]
		vectorisedBranchActivationLevelBatch = vectorisedBranchActivationLevelBatchList[branchIndex1]
		vectorisedBranchActivationTimeBatch = vectorisedBranchActivationTimeBatchList[branchIndex1]
		vectorisedBranchActivationFlagBatch = vectorisedBranchActivationFlagBatchList[branchIndex1]
		if(recordVectorisedBranchObjectList):
			vectorisedBranchObjectBatch = vectorisedBranchObjectBatchList[branchIndex1]
				
		#print("\tbranchIndex1 = ", branchIndex1)
						
		#initialise sequential segments activation (shape: [batchSize, numberOfHorizontalBranches, horizontalBranchWidth]);
		
		if(reversePropagationOrder):
			vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed = calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatch.shape, vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentPrevious)
		
		for sequentialSegmentIndex in sequentialSegmentSequence:

			if(not reversePropagationOrder):
				if(sequentialSegmentIndex < numberOfBranchSequentialSegments-1):
					vectorisedBranchActivationLevelBatchSequentialSegmentPrevious = vectorisedBranchActivationLevelBatchList[branchIndex1][:, :, :, sequentialSegmentIndex+1]
					vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = vectorisedBranchActivationLevelBatchList[branchIndex1][:, :, :, sequentialSegmentIndex+1]
					vectorisedBranchActivationFlagBatchSequentialSegmentPrevious = vectorisedBranchActivationLevelBatchList[branchIndex1][:, :, :, sequentialSegmentIndex+1]
				else:
					if(branchIndex1 < numberOfVerticalBranches-1):
						vectorisedBranchActivationLevelBatchSequentialSegmentPrevious = vectorisedBranchActivationLevelBatchList[branchIndex1+1][:, :, :, sequentialSegmentIndexMostProximal]
						vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = vectorisedBranchActivationTimeBatchList[branchIndex1+1][:, :, :, sequentialSegmentIndexMostProximal]
					else:
						vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = (None, None)
					vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed = calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatch.shape, vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentPrevious)
						
			if((branchIndex1 > 0) or expectFirstBranchSequentialSegmentConnection):
				vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatch[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatch[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationFlagBatchSequentialSegment = vectorisedBranchActivationFlagBatch[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationLevelBatchSequentialSegmentBuffer = vectorisedBranchActivationLevelBatchBuffer[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationTimeBatchSequentialSegmentBuffer = vectorisedBranchActivationTimeBatchBuffer[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationFlagBatchSequentialSegmentBuffer = vectorisedBranchActivationFlagBatchBuffer[:, :, :, sequentialSegmentIndex]

				vectorisedBranchActivationFlagBatchSequentialSegmentCurrent = vectorisedBranchActivationFlagBatchSequentialSegment	#vectorisedBranchActivationFlagBatchSequentialSegmentCurrent is derived from memory, not from previous branch/sequential segment
				
				if(deactivateSequentialSegmentsIfAllConnectionInputsOff):
					vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatchSequentialSegmentBuffer	#overwrite current sequential segment with buffer
					vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatchSequentialSegmentBuffer	#overwrite current sequential segment with buffer
					vectorisedBranchActivationFlagBatchSequentialSegment = vectorisedBranchActivationFlagBatchSequentialSegmentBuffer	#overwrite current sequential segment with buffer
					vectorisedBranchActivationStateBatchSequentialSegment = calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer)
				else:
					vectorisedBranchActivationStateBatchSequentialSegment = calculateSequentialSegmentActivationStateVectorisedMemory(vectorisedBranchActivationLevelBatchSequentialSegment)
					
					vectorisedBranchActivationNewBatchSequentialSegmentMask = calculateVectorisedBranchActivationNewBatchSequentialSegmentMask(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedBranchActivationStateBatchSequentialSegment, vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedBranchActivationFlagBatchSequentialSegment, activationTime)
					vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat = tf.cast(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.float32)	
					vectorisedBranchActivationExistingBatchSequentialSegmentMask = tf.logical_not(vectorisedBranchActivationNewBatchSequentialSegmentMask)
					vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat = tf.cast(vectorisedBranchActivationExistingBatchSequentialSegmentMask, tf.float32)
					
					vectorisedBranchActivationStateBatchSequentialSegmentNew = vectorisedBranchActivationNewBatchSequentialSegmentMask	#implied
					vectorisedBranchActivationLevelBatchSequentialSegmentNew = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)
					vectorisedBranchActivationTimeBatchSequentialSegmentNew = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegmentBuffer, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)
					vectorisedBranchActivationFlagBatchSequentialSegmentNew = tf.multiply(vectorisedBranchActivationFlagBatchSequentialSegmentBuffer, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)
					
					vectorisedBranchActivationStateBatchSequentialSegmentExisting = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMask)	#existing activations will be reapplied later
					vectorisedBranchActivationLevelBatchSequentialSegmentExisting = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat)	#existing activations will be reapplied later
					vectorisedBranchActivationTimeBatchSequentialSegmentExisting = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat)	#existing activations will be reapplied later
					vectorisedBranchActivationFlagBatchSequentialSegmentExisting = tf.multiply(vectorisedBranchActivationFlagBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat)	#existing activations will be reapplied later
					
					vectorisedBranchActivationStateBatchSequentialSegment = calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegmentNew)	#perform sequential segment calculations with new activations from buffer	#or calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegment)				
					vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatchSequentialSegmentNew	#perform sequential segment calculations with new activations from buffer
					vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatchSequentialSegmentNew		#perform sequential segment calculations with new activations from buffer	
					vectorisedBranchActivationFlagBatchSequentialSegment = vectorisedBranchActivationFlagBatchSequentialSegmentNew		#perform sequential segment calculations with new activations from buffer	

					vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationNewBatchSequentialSegmentMask)	#perform sequential segment calculations with new activations from buffer	#existing activations will be reapplied later
					vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)	#perform sequential segment calculations with new activations from buffer	#existing activations will be reapplied later		
					vectorisedBranchActivationFlagBatchSequentialSegmentCurrent = tf.multiply(vectorisedBranchActivationFlagBatchSequentialSegmentCurrent, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)	#perform sequential segment calculations with new activations from buffer	#existing activations will be reapplied later		
		
				vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence = getFirstInputInSequenceFlagFromVectorisedBranchActivationFlagBatchSequentialSegmentBuffer(vectorisedBranchActivationFlagBatchSequentialSegment)	#note vectorisedBranchActivationFlagBatchSequentialSegment contents is from buffer
			
				#apply previous sequentialSegment/subbranch activation level tests;
				#note if firstSequentialSegmentInSequence and higher branch input is zero, then sequential segment will still activate
				vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_and(tf.logical_or(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence), vectorisedBranchActivationStateBatchSequentialSegment)	
				
				#apply previous sequentialSegment/subbranch activation time tests;
				#note if firstSequentialSegmentInSequence and previous sequential segments/subbranch time tests fail, then sequential segment will still activate
				vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests = verifySequentialActivationTimeVectorised(activationTime, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent)

				vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests = tf.logical_or(vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests, vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence)
				vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests)
				
				if(performSummationOfSequentialSegmentInputsAcrossBranch):
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.where(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, x=vectorisedBranchActivationLevelBatchSequentialSegment, y=vectorisedActivationLevelOff)	#filter sequential segment activation level based on previous sequentialSegment/subbranch activation level/time tests
				else:
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.cast(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, tf.float32)
					#or vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = activationTime (as activationTimes will be ignored for unactivated sequential segments) 
				vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.where(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, x=activationTime, y=minimumActivationTime)	#filter sequential segment activation time based on previous sequentialSegment/subbranch activation level/time tests			
				vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.cast(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, tf.float32)
		
				if(overwriteSequentialSegmentsAfterPropagatingSignal):
					if(not ((branchIndex1 == branchIndex1MostProximal) and (sequentialSegmentIndex == sequentialSegmentIndexMostProximal))):	#never freeze most proximal sequential segment in tree
						#freeze all newly activated sequential segment states
						vectorisedBranchActivationFlagBatchSequentialSegmentFrozen = getFrozenFlagFromVectorisedBranchActivationFlagBatchSequentialSegment(vectorisedBranchActivationFlagBatchSequentialSegmentCurrent)
						vectorisedBranchActivationFlagBatchSequentialSegmentCurrent = tf.cast(tf.logical_or(vectorisedBranchActivationFlagBatchSequentialSegmentFrozen, vectorisedBranchActivationStateBatchSequentialSegmentCurrent), tf.float32)
					unfreezePreviousSequentialSegmentOrSubbranchVectorised(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, branchIndex1, sequentialSegmentIndex, vectorisedBranchActivationFlagBatchList, vectorisedBranchActivationFlagBatchSequentialSegmentPrevious)
						
				#if(resetConnectionTargetNeuronDendriteAfterSequence):
				vectorisedBranchActivationStateBatchSequentialSegmentNew = vectorisedBranchActivationStateBatchSequentialSegmentCurrent

				if(not deactivateSequentialSegmentsIfAllConnectionInputsOff):
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.add(vectorisedBranchActivationLevelBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentExisting)		#merge contents of mutually exclusive indices together
					vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.add(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, vectorisedBranchActivationTimeBatchSequentialSegmentExisting)	#merge contents of mutually exclusive indices together
					vectorisedBranchActivationFlagBatchSequentialSegmentCurrent = tf.add(vectorisedBranchActivationFlagBatchSequentialSegmentCurrent, vectorisedBranchActivationFlagBatchSequentialSegmentExisting)	#merge contents of mutually exclusive indices together					
					vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_or(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationStateBatchSequentialSegmentExisting)
					
			else:
				vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.cast(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, tf.float32)
				
				vectorisedBranchActivationStateBatchSequentialSegmentNew = tf.zeros(shape=vectorisedBranchActivationStateBatchSequentialSegmentCurrent.shape)	#invalid (used for draw compatibility)
			
			if((branchIndex1 == branchIndex1MostProximal) and (sequentialSegmentIndex == sequentialSegmentIndexMostProximal)):
				vectorisedBranchActivationStateBatchSequentialSegmentFinalNew = vectorisedBranchActivationStateBatchSequentialSegmentNew
			
			vectorisedBranchActivationLevelBatchList[branchIndex1][:, :, :, sequentialSegmentIndex].assign(vectorisedBranchActivationLevelBatchSequentialSegmentCurrent)
			vectorisedBranchActivationTimeBatchList[branchIndex1][:, :, :, sequentialSegmentIndex].assign(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent)
			vectorisedBranchActivationFlagBatchList[branchIndex1][:, :, :, sequentialSegmentIndex].assign(vectorisedBranchActivationFlagBatchSequentialSegmentCurrent)
			#note conceptNode.vectorisedBranchActivationLevelList/vectorisedBranchActivationTimeList will be updated at end of simulateBiologicalHFnetworkSequenceNodesPropagateParallel based on vectorisedBranchActivationLevelBatchList/vectorisedBranchActivationTimeBatchList	

			if(resetConnectionTargetNeuronDendriteDuringActivation):
				deactivatePreviousSequentialSegmentOrSubbranchVectorised(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, branchIndex1, sequentialSegmentIndex, vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentPrevious)
																
			if(updateNeuronObjectActivationLevels):
				vectorisedBranchObjectBatchSequentialSegment = vectorisedBranchObjectBatch[:, :, :, sequentialSegmentIndex]	#requires recordVectorisedBranchObjectList
				for batchIndex in range(vectorisedBranchObjectBatchSequentialSegment.shape[0]):
					batchNeuron = batchNeuronsList[batchIndex]
					for horizontalBranchIndex in range(vectorisedBranchObjectBatchSequentialSegment.shape[1]):
						for branchIndex2 in range(vectorisedBranchObjectBatchSequentialSegment.shape[2]):
							sequentialSegment = vectorisedBranchObjectBatchSequentialSegment[batchIndex, horizontalBranchIndex, branchIndex2]
							activationState = vectorisedBranchActivationStateBatchSequentialSegmentCurrent[batchIndex, horizontalBranchIndex, branchIndex2].numpy()
							activationLevel = vectorisedBranchActivationLevelBatchSequentialSegmentCurrent[batchIndex, horizontalBranchIndex, branchIndex2].numpy()
							activationTimeSeg = vectorisedBranchActivationTimeBatchSequentialSegmentCurrent[batchIndex, horizontalBranchIndex, branchIndex2].numpy()
							activationStateNew = vectorisedBranchActivationStateBatchSequentialSegmentNew[batchIndex, horizontalBranchIndex, branchIndex2].numpy()
							sequentialSegment.activationLevel = activationLevel
							if(activationState):
								sequentialSegment.activationTime = activationTimeSeg
								#print("activate sequential segment: batchNeuron = ", batchNeuron.nodeName, ", branchIndex1 = ", branchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex, ", branchIndex2 = ", branchIndex2, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
								if(resetConnectionTargetNeuronDendriteDuringActivation):
									deactivatePreviousSequentialSegmentOrSubbranch(sequentialSegment)
							if(activationStateNew):
								if(drawBiologicalSimulationDynamicHighlightNewActivations):
									sequentialSegment.activationStateNew = activationStateNew
								if(overwriteSequentialSegmentsAfterPropagatingSignal):
									if(not ((branchIndex1 == branchIndex1MostProximal) and (sequentialSegmentIndex == sequentialSegmentIndexMostProximal))):	#never freeze most proximal sequential segment in tree
										sequentialSegment.frozen = True
									for subbranch in sequentialSegment.branch.subbranches:	
										previousSequentialSegment = subbranch.sequentialSegments[sequentialSegmentIndexMostProximal]
										previousSequentialSegment.frozen = False	
							if(sequentialSegmentIndex == sequentialSegmentIndexMostProximal):
								#update branch object parameters;
								if(storeBranchActivationState):
									sequentialSegment.branch.activationLevel = activationState
								else:
									sequentialSegment.branch.activationLevel = activationLevel
								if(drawBiologicalSimulationDynamicHighlightNewActivations):
									sequentialSegment.branch.activationStateNew = activationStateNew

							#if(activationState):
							#	print("activate branch: batchNeuron = ", batchNeuron.nodeName, ", branchIndex1 = ", branchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex, ", branchIndex2 = ", branchIndex2, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
												
			drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex, activationTime, wTarget=wTarget)			

			if(reversePropagationOrder):
				vectorisedBranchActivationLevelBatchSequentialSegmentPrevious = vectorisedBranchActivationLevelBatchSequentialSegmentCurrent
				vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = vectorisedBranchActivationTimeBatchSequentialSegmentCurrent
				vectorisedBranchActivationFlagBatchSequentialSegmentPrevious = vectorisedBranchActivationFlagBatchSequentialSegmentCurrent
		
		if(requireSubbranchOrSequentialSegmentForActivation):
			vectorisedBranchActivationLevelBatchSequentialSegmentPreviousFull =	tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed, numberOfHorizontalSubBranchesOrSequentialSegmentsRequiredForActivation)
			vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.cast(tf.logical_or(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentPreviousFull), tf.float32)
			#resetConnectionTargetNeuronDendriteAfterSequence:vectorisedBranchActivationStateBatchSequentialSegmentFinalNew not supported (most proximal sequential segment in dendritic tree must be active)
	
	if(reversePropagationOrder):
		vectorisedBranchActivationLevelBatchSequentialSegmentFinal = vectorisedBranchActivationLevelBatchSequentialSegmentPrevious
		vectorisedBranchActivationTimeBatchSequentialSegmentFinal = vectorisedBranchActivationTimeBatchSequentialSegmentPrevious
	else:	
		vectorisedBranchActivationLevelBatchSequentialSegmentFinal = vectorisedBranchActivationLevelBatchList[branchIndex1MostProximal][:, :, :, sequentialSegmentIndexMostProximal]
		vectorisedBranchActivationTimeBatchSequentialSegmentFinal = vectorisedBranchActivationTimeBatchList[branchIndex1MostProximal][:, :, :, sequentialSegmentIndexMostProximal]
		
	if(resetConnectionTargetNeuronDendriteAfterSequence):
		vectorisedSomaActivationLevel = tf.squeeze(vectorisedBranchActivationStateBatchSequentialSegmentFinalNew, axis=[1,2])	#size:batchSize
	else:
		vectorisedSomaActivationLevel = tf.squeeze(vectorisedBranchActivationLevelBatchSequentialSegmentFinal, axis=[1,2])	#size:batchSize
	vectorisedSomaActivationTime = tf.squeeze(vectorisedBranchActivationTimeBatchSequentialSegmentFinal, axis=[1,2]) #size:batchSize
	
	batchSize = vectorisedBranchActivationLevelBatchSequentialSegmentFinal.shape[0]	#or #batchSize = vectorisedBranchObjectBatchList[0].shape[0]
	for batchIndex in range(batchSize):
		if(updateNeuronObjectActivationLevels):
			vectorisedBranchObjectBatchSequentialSegment = vectorisedBranchObjectBatchList[0][batchIndex, 0, 0, 0]	#branchIndex1Arbitrary = 0	#indexArbitrary = 0	#get any (first) sequential segment object in batchIndex neuron
			batchNeuron = vectorisedBranchObjectBatchSequentialSegment.conceptNode
		else:
			batchNeuron = batchNeuronsList[batchIndex]
		somaActivationLevel = vectorisedSomaActivationLevel[batchIndex].numpy()
		somaActivationLevel = bool(somaActivationLevel)
		
		if(printConnectionTargetActivations):
			print("calculateNeuronActivationParallel: conceptNeuronConnectionTarget = ", batchNeuron.nodeName, ", somaActivationLevel = ", somaActivationLevel)
		
		if(applySomaActivation(batchNeuron, conceptNeuronTarget, somaActivationLevel)):
			somaActivationFound = True

	#print("somaActivationFound = ", somaActivationFound)
	
	drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=wTarget)
						
	return somaActivationFound

def calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer):
	#sync with calculateSequentialSegmentActivationState(activationLevel, vectorised=True)
	if(weightedSequentialSegmentInputs):
		if(performSummationOfSequentialSegmentInputs):
			vectorisedBranchActivationStateBatchSequentialSegmentBuffer = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, sequentialSegmentMinActivationLevel)	#or tf.greater	#vectorisedBranchActivationLevelListBuffer stores numeric values
		else:
			vectorisedBranchActivationStateBatchSequentialSegmentBuffer = tf.greater(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedActivationLevelOff)
	else:
		vectorisedBranchActivationStateBatchSequentialSegmentBuffer = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedActivationLevelOn)	#or tf.equal
	return vectorisedBranchActivationStateBatchSequentialSegmentBuffer

def calculateSequentialSegmentActivationStateVectorisedMemory(vectorisedBranchActivationLevelBatchSequentialSegment):
	if(performSummationOfSequentialSegmentInputsAcrossBranch):
		vectorisedBranchActivationStateBatchSequentialSegment = tf.greater(vectorisedBranchActivationLevelBatchSequentialSegment, vectorisedActivationLevelOff)
	else:	
		vectorisedBranchActivationStateBatchSequentialSegment = tf.equal(vectorisedBranchActivationLevelBatchSequentialSegment, vectorisedActivationLevelOn)	#greater_equal test not required as vectorisedBranchActivationLevelList always stores effective boolean 1/0 values (only vectorisedBranchActivationLevelListBuffer stores numeric values)
	return vectorisedBranchActivationStateBatchSequentialSegment
										
def calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatchShape, vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch, vectorisedBranchActivationTimeBatchSequentialSegmentPreviousBranch):
	#initialise sequential segments activation;
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 
	if(branchIndex1 == numberOfVerticalBranches-1):
		#highest branch in dendritic tree (initialise activation to true)
		vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.cast(tf.ones((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2])), tf.bool)	#use every dimension except sequentialSegmentIndex 	#shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth]
		vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.multiply(tf.ones((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2])), minimumActivationTime) 	#use every dimension except sequentialSegmentIndex	#shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth]	#initial activation time of dendritic leaf nodes set artificially low such that passSegmentActivationTimeTests automatically pass
		if(requireSubbranchOrSequentialSegmentForActivation):
			vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed = tf.ones((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2]))	#set to 1 for requireSubbranchOrSequentialSegmentForActivation
		else:
			vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed = None
			#vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed = tf.zeros((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2]))	#not used		
	else:
		#intermediary branch in dendritic tree (initialise activation to that of higher branch)
		#print("vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch.shape = ", vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch.shape) 
		vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed = tf.reduce_sum(vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch, axis=2)
		#print("vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed.shape = ", vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed.shape) 
		if(performSummationOfSequentialSegmentInputsAcrossBranch):
			vectorisedBranchActivationStateBatchSequentialSegmentPreviousActive = tf.greater(vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch, 0)
			vectorisedBranchActivationTimeBatchSequentialSegmentPreviousFilteredActive = tf.where(vectorisedBranchActivationStateBatchSequentialSegmentPreviousActive, vectorisedBranchActivationTimeBatchSequentialSegmentPreviousBranch, minimumActivationTime)
			vectorisedBranchActivationTimeBatchSequentialSegmentPreviousMax = tf.reduce_max(vectorisedBranchActivationTimeBatchSequentialSegmentPreviousFilteredActive, axis=2)
			vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed, numberOfHorizontalSubBranchesRequiredForActivation)	
		else:
			vectorisedBranchActivationTimeBatchSequentialSegmentPreviousMax = tf.reduce_max(vectorisedBranchActivationTimeBatchSequentialSegmentPreviousBranch, axis=2)
			vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed, numberOfHorizontalSubBranchesRequiredForActivation)
		vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = vectorisedBranchActivationTimeBatchSequentialSegmentPreviousMax
		numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(branchIndex1, numberOfBranches2)
		vectorisedBranchActivationStateBatchSequentialSegmentCurrent = sliceReshapeExpandDims(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, horizontalBranchWidth, axis=-1)
		vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = sliceReshapeExpandDims(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, horizontalBranchWidth, axis=-1)
		if(requireSubbranchOrSequentialSegmentForActivation):
			vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed = sliceReshapeExpandDims(vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed, horizontalBranchWidth, axis=-1)
	return vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentPreviousSummed

def sliceReshapeExpandDims(t, numberOfSlices, axis):
	sliceList = []
	for sliceIndex in range(numberOfSlices):
		indices = tf.range(sliceIndex, t.shape[axis], delta=numberOfSlices)
		tSlice = tf.gather(t, indices, axis=axis)
		sliceList.append(tSlice)
	tModified = tf.stack(sliceList, axis=-1)
	return tModified

def calculateVectorisedBranchActivationNewBatchSequentialSegmentMask(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedBranchActivationStateBatchSequentialSegment, vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedBranchActivationFlagBatchSequentialSegment, activationTime):
	vectorisedBranchActivationNewBatchSequentialSegmentMask = calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer)
	#print("vectorisedBranchActivationNewBatchSequentialSegmentMask = ", vectorisedBranchActivationNewBatchSequentialSegmentMask)
	if(overwriteSequentialSegments):
		if(overwriteSequentialSegmentsAfterPropagatingSignal):
			#sync with calculateNeuronActivationStandard:overwriteSequentialSegmentsAfterPropagatingSignal
			vectorisedBranchActivationStateBatchSequentialSegmentOverwrite = tf.logical_and(vectorisedBranchActivationNewBatchSequentialSegmentMask, vectorisedBranchActivationStateBatchSequentialSegment)
			vectorisedBranchActivationStateBatchSequentialSegmentNonOverwrite = tf.logical_and(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegment))
			vectorisedBranchActivationFlagBatchSequentialSegmentOverwrite = tf.multiply(vectorisedBranchActivationFlagBatchSequentialSegment, tf.cast(vectorisedBranchActivationStateBatchSequentialSegmentOverwrite, tf.float32))
			vectorisedBranchActivationFlagBatchSequentialSegmentOverwriteFrozen = getFrozenFlagFromVectorisedBranchActivationFlagBatchSequentialSegment(vectorisedBranchActivationFlagBatchSequentialSegmentOverwrite)
			vectorisedBranchActivationStateBatchSequentialSegmentOverwriteFrozenTests = tf.logical_not(vectorisedBranchActivationFlagBatchSequentialSegmentOverwriteFrozen)
			vectorisedBranchActivationStateBatchSequentialSegmentOverwriteFrozenTests = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegmentOverwriteFrozenTests, vectorisedBranchActivationStateBatchSequentialSegmentOverwrite)	#required because frozen flag tests is only valid for valid times (ie states=On)
			vectorisedBranchActivationNewBatchSequentialSegmentMask = tf.logical_or(vectorisedBranchActivationStateBatchSequentialSegmentOverwriteFrozenTests, vectorisedBranchActivationStateBatchSequentialSegmentNonOverwrite)
		if(verifyActivationTime):
			vectorisedBranchActivationStateBatchSequentialSegmentOverwrite = tf.logical_and(vectorisedBranchActivationNewBatchSequentialSegmentMask, vectorisedBranchActivationStateBatchSequentialSegment)
			vectorisedBranchActivationStateBatchSequentialSegmentNonOverwrite = tf.logical_and(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegment))
			vectorisedBranchActivationTimeBatchSequentialSegmentOverwrite = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegment, tf.cast(vectorisedBranchActivationStateBatchSequentialSegmentOverwrite, tf.float32))
			vectorisedBranchActivationStateBatchSequentialSegmentOverwriteTimeTests = verifyReactivationTimeVectorised(vectorisedBranchActivationTimeBatchSequentialSegmentOverwrite, activationTime)
			vectorisedBranchActivationStateBatchSequentialSegmentOverwriteTimeTests = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegmentOverwriteTimeTests, vectorisedBranchActivationStateBatchSequentialSegmentOverwrite)	#required because verifyRepolarisedVectorised is only valid for valid times (ie states=On)
			vectorisedBranchActivationNewBatchSequentialSegmentMask = tf.logical_or(vectorisedBranchActivationStateBatchSequentialSegmentOverwriteTimeTests, vectorisedBranchActivationStateBatchSequentialSegmentNonOverwrite)
	else:
		vectorisedBranchActivationNewBatchSequentialSegmentMask = tf.logical_and(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegment))
	#print("2 vectorisedBranchActivationNewBatchSequentialSegmentMask = ", vectorisedBranchActivationNewBatchSequentialSegmentMask)
	return vectorisedBranchActivationNewBatchSequentialSegmentMask

def getFrozenFlagFromVectorisedBranchActivationFlagBatchSequentialSegment(vectorisedBranchActivationFlagBatchSequentialSegment):
	vectorisedBranchActivationFlagBatchSequentialSegmentFrozen = tf.cast(vectorisedBranchActivationFlagBatchSequentialSegment, tf.bool)
	return vectorisedBranchActivationFlagBatchSequentialSegmentFrozen

def getFirstInputInSequenceFlagFromVectorisedBranchActivationFlagBatchSequentialSegmentBuffer(vectorisedBranchActivationFlagBatchSequentialSegmentBuffer):
	vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence = tf.equal(vectorisedBranchActivationFlagBatchSequentialSegmentBuffer, vectorisedActivationTimeFlagFirstInputInSequence)
	return vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence
	

def verifyReactivationTimeVectorised(vectorisedBranchActivationTimeBatchSequentialSegmentOverwrite, activationTime):	
	#sync with verifyReactivationTime
	if(verifyRepolarisationTime):
		repolarised = tf.greater_equal(activationTime, tf.add(vectorisedBranchActivationTimeBatchSequentialSegmentOverwrite, activationRepolarisationTime))
	else:
		repolarised = tf.cast(tf.ones(vectorisedBranchActivationTimeBatchSequentialSegmentOverwrite.shape), tf.bool)
	return repolarised

def verifySequentialActivationTimeVectorised(activationTime, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent):	
	#sync with verifySequentialActivationTime
	if(algorithmTimingWorkaround1):
		sequentiality = tf.greater_equal(activationTime, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent)	#ensure activationTime of sequentialSegment is greater than that of previous sequential segments/subbranch - equivalent to verifySequentialActivationTime			
	else:
		sequentiality = tf.greater(activationTime, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent)	#ensure activationTime of sequentialSegment is greater than that of previous sequential segments/subbranch - equivalent to verifySequentialActivationTime

	if(verifyPropagationTime):
		propagate = tf.less_equal(activationTime, tf.add(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, activationPropagationTimeMax))
	else:
		propagate = tf.cast(tf.ones(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent.shape), tf.bool)
	
	vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests = tf.logical_and(sequentiality, propagate)
	
	return vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests
						
def deactivatePreviousSequentialSegmentOrSubbranchVectorised(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, branchIndex1, sequentialSegmentIndex, vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentPrevious):
	#sync with deactivatePreviousSequentialSegmentOrSubbranch
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
	if(isMostDistalSequentialSegmentInBranch(sequentialSegmentIndex)):
		if(branchIndex1 < numberOfVerticalBranches-1):
			#update previous branch last sequential segment activations
			previousBranchIndex1 = branchIndex1+1
			numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(previousBranchIndex1, numberOfBranches2)
			#print("vectorisedBranchActivationStateBatchSequentialSegmentCurrent = ", vectorisedBranchActivationStateBatchSequentialSegmentCurrent)
			vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = tf.reshape(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, shape=(vectorisedBranchActivationStateBatchSequentialSegmentCurrent.shape[0],vectorisedBranchActivationStateBatchSequentialSegmentCurrent.shape[1]*vectorisedBranchActivationStateBatchSequentialSegmentCurrent.shape[2])) 	#shape [batchSize, numberOfHorizontalBranches*horizontalBranchWidth]
			#print("vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = ", vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch)
			vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = tf.expand_dims(vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch, axis=2)
			#print("vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = ", vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch)
			multiples = [1, 1, horizontalBranchWidth]
			vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = tf.tile(vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch, multiples)
			#print("vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = ", vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch)
			vectorisedBranchActivationLevelBatchSequentialSegmentPrevious = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, tf.cast(tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch), tf.float32))
			vectorisedBranchActivationLevelBatchList[previousBranchIndex1][:, :, :, sequentialSegmentIndexMostProximal].assign(vectorisedBranchActivationLevelBatchSequentialSegmentPrevious)
			#vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = vectorisedBranchActivationTimeBatchSequentialSegmentPrevious	#no change to last activation times
			#vectorisedBranchActivationTimeBatchList[branchIndex1+1][:, :, :, sequentialSegmentIndexMostProximal].assign(vectorisedBranchActivationTimeBatchSequentialSegmentPrevious)
	else:
		#update previous sequential segment activations
		previousSequentialSegmentIndex = sequentialSegmentIndex+1
		vectorisedBranchActivationLevelBatchSequentialSegmentPrevious = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, tf.cast(tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegmentCurrent), tf.float32))
		vectorisedBranchActivationLevelBatchList[branchIndex1][:, :, :, previousSequentialSegmentIndex].assign(vectorisedBranchActivationLevelBatchSequentialSegmentPrevious)
		#vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = vectorisedBranchActivationTimeBatchSequentialSegmentPrevious	#no change to last activation times
		#vectorisedBranchActivationTimeBatchList[branchIndex1][:, :, :, previousSequentialSegmentIndex].assign(vectorisedBranchActivationTimeBatchSequentialSegmentPrevious)	

def unfreezePreviousSequentialSegmentOrSubbranchVectorised(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, branchIndex1, sequentialSegmentIndex, vectorisedBranchActivationFlagBatchList, vectorisedBranchActivationFlagBatchSequentialSegmentPrevious):
	#sync with calculateNeuronActivationStandard:overwriteSequentialSegmentsAfterPropagatingSignal 
	#most code extracted from deactivatePreviousSequentialSegmentOrSubbranchVectorised
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
	if(isMostDistalSequentialSegmentInBranch(sequentialSegmentIndex)):
		if(branchIndex1 < numberOfVerticalBranches-1):
			#update previous branch last sequential segment activations
			previousBranchIndex1 = branchIndex1+1
			numberOfHorizontalBranches, horizontalBranchWidth = calculateNumberOfHorizontalBranches(previousBranchIndex1, numberOfBranches2)
			vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = tf.reshape(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, shape=(vectorisedBranchActivationStateBatchSequentialSegmentCurrent.shape[0],vectorisedBranchActivationStateBatchSequentialSegmentCurrent.shape[1]*vectorisedBranchActivationStateBatchSequentialSegmentCurrent.shape[2])) 	#shape [batchSize, numberOfHorizontalBranches*horizontalBranchWidth]
			vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = tf.expand_dims(vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch, axis=2)
			multiples = [1, 1, horizontalBranchWidth]
			vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch = tf.tile(vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch, multiples)
			vectorisedBranchActivationFlagBatchSequentialSegmentPrevious = tf.multiply(vectorisedBranchActivationFlagBatchSequentialSegmentPrevious, tf.cast(tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegmentCurrentReshapedForPrevBranch), tf.float32))
			vectorisedBranchActivationFlagBatchList[previousBranchIndex1][:, :, :, sequentialSegmentIndexMostProximal].assign(vectorisedBranchActivationFlagBatchSequentialSegmentPrevious)
	else:
		#update previous sequential segment activations
		previousSequentialSegmentIndex = sequentialSegmentIndex+1
		vectorisedBranchActivationFlagBatchSequentialSegmentPrevious = tf.multiply(vectorisedBranchActivationFlagBatchSequentialSegmentPrevious, tf.cast(tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegmentCurrent), tf.float32))
		vectorisedBranchActivationFlagBatchList[branchIndex1][:, :, :, previousSequentialSegmentIndex].assign(vectorisedBranchActivationFlagBatchSequentialSegmentPrevious)

def drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex, activationTime, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivationParallel or (sentenceIndex == sentenceIndexDebug and wTarget == wSource+1)):
		#if(not debugCalculateNeuronActivationParallel or (sentenceIndex == sentenceIndexDebug and wSource >= wSourceDebug)):
			print("branchIndex1 = ", branchIndex1, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
			if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
				fileName = generateBiologicalSimulationDynamicFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.drawHopfieldGraphSentence(sentenceConceptNodeList, activationTime=activationTime, wTarget=wTargetDebug)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
				fileName = generateBiologicalSimulationDynamicFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.drawHopfieldGraphNetwork(networkConceptNodeDict, activationTime=activationTime, wTarget=wTargetDebug)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)				

def drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivationParallel or (sentenceIndex == sentenceIndexDebug and wSource >= wSourceDebug)):
			if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
				fileName = generateBiologicalSimulationFileName(True, wSource, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.drawHopfieldGraphSentence(sentenceConceptNodeList, activationTime=activationTime, wTarget=wTargetDebug)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
				fileName = generateBiologicalSimulationFileName(False, wSource, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.drawHopfieldGraphNetwork(networkConceptNodeDict, activationTime=activationTime, wTarget=wTargetDebug)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)		

