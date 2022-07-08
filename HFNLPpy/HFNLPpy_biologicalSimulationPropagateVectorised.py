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

debugCalculateNeuronActivationParallel = False
if(debugCalculateNeuronActivationParallel):
	sentenceIndexDebug = 23
	wSourceDebug = 8
	branchIndex1Debug = 2
	batchIndexOfWTargetDebug = None
	wTargetDebug = 8
else:
	wTargetDebug = None

#parameters only used for drawBiologicalSimulationDynamic: sentenceIndex, sentenceConceptNodeList
def simulateBiologicalHFnetworkSequenceNodePropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet):
	conceptNeuronSourceList = []
	conceptNeuronSourceList.append(conceptNeuronSource)
	return simulateBiologicalHFnetworkSequenceNodesPropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)
	
#parameters only used for drawBiologicalSimulationDynamic: wSource, sentenceIndex, sentenceConceptNodeList
def simulateBiologicalHFnetworkSequenceNodesPropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet):
	
	#construct batch dendritic tree templates for parallel processing;
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
	vectorisedBranchActivationLevelBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationTimeBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationLevelBatchListListBuffer = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationTimeBatchListListBuffer = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing	
	if(recordVectorisedBranchObjectList):
		vectorisedBranchObjectBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationLevelBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationTimeBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationLevelBatchListBuffer = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationTimeBatchListBuffer = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	if(recordVectorisedBranchObjectList):
		vectorisedBranchObjectBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	else:
		vectorisedBranchObjectBatchList = None
		
	batchNeuronsList = []	#preserve insertion order	#alternatively in recordVectorisedBranchObjectList; can lookup batchNeurons from vectorisedBranchObjectBatchList instead
	batchIndex = 0	#batchSampleIndex
	conceptNeuronBatchIndex = None
	conceptNeuronBatchIndexFound = False
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
		
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

				if(targetConnectionConceptName == conceptNeuronTarget.nodeName):
					conceptNeuronBatchIndex = batchIndex
					conceptNeuronBatchIndexFound = True
					#print("conceptNeuronTarget.nodeName = ", conceptNeuronTarget.nodeName)
					#print("conceptNeuronBatchIndex = ", conceptNeuronBatchIndex)
				batchIndex += 1
				
				#create temporary vectorised buffers for conceptNeuronSource connection target input sequentialSegment candidate application;
				conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer, conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=False, storeSequentialSegmentInputActivationLevels=vectoriseComputionUseSequentialSegmentInputActivationLevels)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, {numberOfSequentialSegmentInputs}]
				
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
					vectorisedBranchActivationLevelBatchListListBuffer[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1])
					vectorisedBranchActivationTimeBatchListListBuffer[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1])
					if(recordVectorisedBranchObjectList):
						vectorisedBranchObjectBatchListList[branchIndex1].append(conceptNeuronConnectionTarget.vectorisedBranchObjectList[branchIndex1])			

	for branchIndex1 in range(numberOfVerticalBranches):
		vectorisedBranchActivationLevelBatchList[branchIndex1] = tf.Variable(tf.stack(vectorisedBranchActivationLevelBatchListList[branchIndex1]))
		vectorisedBranchActivationTimeBatchList[branchIndex1] = tf.Variable(tf.stack(vectorisedBranchActivationTimeBatchListList[branchIndex1]))	
		vectorisedBranchActivationLevelBatchListBuffer[branchIndex1] = tf.stack(vectorisedBranchActivationLevelBatchListListBuffer[branchIndex1])
		vectorisedBranchActivationTimeBatchListBuffer[branchIndex1] = tf.stack(vectorisedBranchActivationTimeBatchListListBuffer[branchIndex1])			
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

	if(conceptNeuronBatchIndexFound):	#optimsation; only execute calculateNeuronActivationParallel if conceptNeuronTarget input(s) are activated by conceptNeuronSource
		if(calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationLevelBatchListBuffer, vectorisedBranchActivationTimeBatchListBuffer, vectorisedBranchObjectBatchList, activationTime, wTarget, conceptNeuronTarget, conceptNeuronBatchIndex, batchNeuronsList, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)):
			somaActivationFound = True
	else:
		print("warning !conceptNeuronBatchIndexFound")
	
	#save updated activations (ideally these should be able to be dynamically updated by calculateNeuronActivationParallel; store tensors (memory/reference) in a bulk/stacked tensor, write to the bulk tensor and have the individual tensors updated)
	for batchIndex, batchNeuron in enumerate(batchNeuronsList):
		for branchIndex1 in range(numberOfVerticalBranches):
			#iterating over batchSize to save tensors is slow and may require optimisation
			batchNeuron.vectorisedBranchActivationLevelList[branchIndex1] = tf.Variable(vectorisedBranchActivationLevelBatchList[branchIndex1][batchIndex])
			batchNeuron.vectorisedBranchActivationTimeList[branchIndex1] = tf.Variable(vectorisedBranchActivationTimeBatchList[branchIndex1][batchIndex])
			
	for conceptNeuronSource in conceptNeuronSourceList:
		if(updateNeuronObjectActivationLevels):
			resetSourceNeuronAfterActivation(conceptNeuronSource)
		else:
			if(resetSourceNeuronDendriteAfterActivation):
				resetDendriticTreeActivationVectorised(conceptNeuronSource)

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
			currentSequentialSegmentInput.activationTime = activationTime	
		activationTimeFlags = vectorisedActivationTimeFlagDefault
		if(currentSequentialSegmentInput.firstInputInSequence):
			activationTimeFlags = vectorisedActivationTimeFlagFirstInputInSequence
		conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationValue)
		conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationTimeFlags)	#orig: activationTime
	else:
		if(updateNeuronObjectActivationLevels):
			if(weightedSequentialSegmentInputs):
				currentSequentialSegmentInput.activationLevel = activationValue		#CHECKTHIS: or connection.weight (disregarding firstInputInSequence activation level modifier)
			else:
				currentSequentialSegmentInput.activationLevel = objectLocalActivationLevelOn	#True
			currentSequentialSegmentInput.activationTime = activationTime
		activationTimeFlags = vectorisedActivationTimeFlagDefault
		if(performSummationOfSequentialSegmentInputs):
			summationOfSequentialSegmentInputs = conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].numpy()
			summationOfSequentialSegmentInputs = summationOfSequentialSegmentInputs + activationValue
			activationValue = summationOfSequentialSegmentInputs
			firstInputInSequenceExisting = conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].numpy()
			if(bool(firstInputInSequenceExisting)):
				activationTimeFlags = vectorisedActivationTimeFlagFirstInputInSequence
		if(currentSequentialSegmentInput.firstInputInSequence):
			activationTimeFlags = vectorisedActivationTimeFlagFirstInputInSequence
		#if(verifyRepolarised(currentSequentialSegment, activationTime)):
		#print("activationValue = ", activationValue)
		conceptNeuronConnectionTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationValue)
		conceptNeuronConnectionTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationTimeFlags)	#orig: activationTime (all inputs should have same activation time)

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
def calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationLevelBatchListBuffer, vectorisedBranchActivationTimeBatchListBuffer, vectorisedBranchObjectBatchList, activationTime, wTarget, conceptNeuronTarget, conceptNeuronBatchIndex, batchNeuronsList, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):
	
	somaActivationFound = False
	
	#print("calculateNeuronActivationParallel:")
		
	#vectorisedBranchActivationLevelBatchList/vectorisedBranchActivationTimeBatchList: list of tensors for every branchIndex1 - each element is of shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments], each batch sample refers to a unique target concept
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 	#len(vectorisedBranchActivationLevelBatchList)
	
	vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch, vectorisedBranchActivationTimeBatchSequentialSegmentPreviousBranch = (None, None)
	if(resetConnectionTargetNeuronDendriteDuringActivation):
		#note vectorisedBranchActivationLevelBatchSequentialSegmentPrevious is same as vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch but updated more regularly than vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch (for every sequential segment); could therefore replace PreviousBranch tensor with Previous tensor 
		vectorisedBranchActivationLevelBatchSequentialSegmentPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = (None, None)
	
	for branchIndex1 in reversed(range(numberOfVerticalBranches)):
			
		vectorisedBranchActivationLevelBatchBuffer = vectorisedBranchActivationLevelBatchListBuffer[branchIndex1]
		vectorisedBranchActivationTimeBatchBuffer = vectorisedBranchActivationTimeBatchListBuffer[branchIndex1]
		vectorisedBranchActivationLevelBatch = vectorisedBranchActivationLevelBatchList[branchIndex1]
		vectorisedBranchActivationTimeBatch = vectorisedBranchActivationTimeBatchList[branchIndex1]
		if(recordVectorisedBranchObjectList):
			vectorisedBranchObjectBatch = vectorisedBranchObjectBatchList[branchIndex1]

		#print("\tbranchIndex1 = ", branchIndex1)
						
		#initialise sequential segments activation (shape: [batchSize, numberOfHorizontalBranches, horizontalBranchWidth]);
		vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatch.shape, vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch, vectorisedBranchActivationTimeBatchSequentialSegmentPreviousBranch)
		
		for sequentialSegmentIndex in reversed(range(numberOfBranchSequentialSegments)):
			if((branchIndex1 > 0) or expectFirstBranchSequentialSegmentConnection):
				vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatch[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatch[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationLevelBatchSequentialSegmentBuffer = vectorisedBranchActivationLevelBatchBuffer[:, :, :, sequentialSegmentIndex]
				vectorisedBranchActivationTimeBatchSequentialSegmentBuffer = vectorisedBranchActivationTimeBatchBuffer[:, :, :, sequentialSegmentIndex]
				
				if(deactivateSequentialSegmentsIfAllConnectionInputsOff):
					vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatchSequentialSegmentBuffer	#overwrite current sequential segment with buffer
					vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatchSequentialSegmentBuffer	#overwrite current sequential segment with buffer
					vectorisedBranchActivationStateBatchSequentialSegment = calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer)
				else:
					vectorisedBranchActivationStateBatchSequentialSegment = calculateSequentialSegmentActivationStateVectorisedMemory(vectorisedBranchActivationLevelBatchSequentialSegment)
					vectorisedBranchActivationNewBatchSequentialSegmentMask = calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer)
					if(not overwriteSequentialSegments):
						vectorisedBranchActivationNewBatchSequentialSegmentMask = tf.logical_and(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.logical_not(vectorisedBranchActivationStateBatchSequentialSegment))
					vectorisedBranchActivationExistingBatchSequentialSegmentMask = tf.logical_not(vectorisedBranchActivationNewBatchSequentialSegmentMask)
					vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat = tf.cast(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.float32)
					vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat = tf.cast(vectorisedBranchActivationExistingBatchSequentialSegmentMask, tf.float32)
					vectorisedBranchActivationLevelBatchSequentialSegmentNew = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)
					vectorisedBranchActivationTimeBatchSequentialSegmentNew = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegmentBuffer, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)
					vectorisedBranchActivationStateBatchSequentialSegmentExisting = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMask)	#existing activations will be reapplied later
					vectorisedBranchActivationLevelBatchSequentialSegmentExisting = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat)	#existing activations will be reapplied later
					vectorisedBranchActivationTimeBatchSequentialSegmentExisting = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat)	#existing activations will be reapplied later
					vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatchSequentialSegmentNew	#perform sequential segment calculations with new activations from buffer
					vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatchSequentialSegmentNew		#perform sequential segment calculations with new activations from buffer	
					vectorisedBranchActivationStateBatchSequentialSegment = calculateSequentialSegmentActivationStateVectorisedBuffer(vectorisedBranchActivationLevelBatchSequentialSegment)	#perform sequential segment calculations with new activations from buffer

					vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationNewBatchSequentialSegmentMask)	#perform sequential segment calculations with new activations from buffer	#existing activations will be reapplied later
					vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)	#perform sequential segment calculations with new activations from buffer	#existing activations will be reapplied later		
			
				vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence = tf.equal(vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedActivationTimeFlagFirstInputInSequence)
			
				#apply previous sequentialSegment/subbranch activation level tests;
				#note if firstSequentialSegmentInSequence and higher branch input is zero, then sequential segment will still activate
				vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_and(tf.logical_or(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence), vectorisedBranchActivationStateBatchSequentialSegment)	
				
				#apply previous sequentialSegment/subbranch activation time tests;
				#note if firstSequentialSegmentInSequence and previous sequential segments/subbranch time tests fail, then sequential segment will still activate
				if(algorithmTimingWorkaround1):
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests = tf.greater_equal(activationTime, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent)	#ensure activationTime of sequentialSegment is greater than that of previous sequential segments/subbranch - equivalent to verifySequentialActivationTime			
				else:
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests = tf.greater(activationTime, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent)	#ensure activationTime of sequentialSegment is greater than that of previous sequential segments/subbranch - equivalent to verifySequentialActivationTime
				vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests = tf.logical_or(vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests, vectorisedBranchActivationFlagBatchSequentialSegmentFirstInputInSequence)
				vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_and(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentCurrentTimeTests)
				
				if(performSummationOfSequentialSegmentInputsAcrossBranch):
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.where(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, x=vectorisedBranchActivationLevelBatchSequentialSegment, y=vectorisedActivationLevelOff)	#filter sequential segment activation level based on previous sequentialSegment/subbranch activation level/time tests
				else:
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.cast(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, tf.float32)
					#or vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = activationTime (as activationTimes will be ignored for unactivated sequential segments) 
				vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.where(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, x=activationTime, y=minimumActivationTime)	#filter sequential segment activation time based on previous sequentialSegment/subbranch activation level/time tests			
				vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.cast(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, tf.float32)
				
				if(not deactivateSequentialSegmentsIfAllConnectionInputsOff):
					vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.add(vectorisedBranchActivationLevelBatchSequentialSegmentCurrent, vectorisedBranchActivationLevelBatchSequentialSegmentExisting)		#merge contents of mutually exclusive indices together
					vectorisedBranchActivationTimeBatchSequentialSegmentCurrent = tf.add(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent, vectorisedBranchActivationTimeBatchSequentialSegmentExisting)	#merge contents of mutually exclusive indices together
					vectorisedBranchActivationStateBatchSequentialSegmentCurrent = tf.logical_or(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationStateBatchSequentialSegmentExisting)
			else:
				vectorisedBranchActivationLevelBatchSequentialSegmentCurrent = tf.cast(vectorisedBranchActivationStateBatchSequentialSegmentCurrent, tf.float32)

			vectorisedBranchActivationLevelBatchList[branchIndex1][:, :, :, sequentialSegmentIndex].assign(vectorisedBranchActivationLevelBatchSequentialSegmentCurrent)
			vectorisedBranchActivationTimeBatchList[branchIndex1][:, :, :, sequentialSegmentIndex].assign(vectorisedBranchActivationTimeBatchSequentialSegmentCurrent)
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
							sequentialSegment.activationLevel = activationLevel
							if(activationState):
								sequentialSegment.activationTime = activationTimeSeg
								#print("activate sequential segment: batchNeuron = ", batchNeuron.nodeName, ", branchIndex1 = ", branchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex, ", branchIndex2 = ", branchIndex2, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
								if(resetConnectionTargetNeuronDendriteDuringActivation):
									deactivatePreviousSequentialSegmentOrSubbranch(sequentialSegment)
							if(sequentialSegmentIndex == 0):
								sequentialSegment.branch.activationLevel = sequentialSegment.activationLevel
								#if(activationState):
								#	print("activate branch: batchNeuron = ", batchNeuron.nodeName, ", branchIndex1 = ", branchIndex1, ", horizontalBranchIndex = ", horizontalBranchIndex, ", branchIndex2 = ", branchIndex2, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
	
			drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex, wTarget)			

			if(resetConnectionTargetNeuronDendriteDuringActivation):
				vectorisedBranchActivationLevelBatchSequentialSegmentPrevious = vectorisedBranchActivationLevelBatchSequentialSegmentCurrent
				vectorisedBranchActivationTimeBatchSequentialSegmentPrevious = vectorisedBranchActivationTimeBatchSequentialSegmentCurrent
						
		vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch = vectorisedBranchActivationLevelBatchSequentialSegmentCurrent
		vectorisedBranchActivationTimeBatchSequentialSegmentPreviousBranch = vectorisedBranchActivationTimeBatchSequentialSegmentCurrent
		
	vectorisedSomaActivationLevel = tf.squeeze(vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch, axis=[1,2])	#size:batchSize
	vectorisedSomaActivationTime = tf.squeeze(vectorisedBranchActivationTimeBatchSequentialSegmentPreviousBranch, axis=[1,2]) #size:batchSize
	
	batchSize = vectorisedBranchActivationLevelBatchSequentialSegmentPreviousBranch.shape[0]	#or #batchSize = vectorisedBranchObjectBatchList[0].shape[0]
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
	
	drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget)
						
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
	return vectorisedBranchActivationStateBatchSequentialSegmentCurrent, vectorisedBranchActivationTimeBatchSequentialSegmentCurrent

def sliceReshapeExpandDims(t, numberOfSlices, axis):
	sliceList = []
	for sliceIndex in range(numberOfSlices):
		indices = tf.range(sliceIndex, t.shape[axis], delta=numberOfSlices)
		tSlice = tf.gather(t, indices, axis=axis)
		sliceList.append(tSlice)
	tModified = tf.stack(sliceList, axis=-1)
	return tModified

def drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivationParallel or (sentenceIndex == sentenceIndexDebug and wSource == wSourceDebug)):
			print("branchIndex1 = ", branchIndex1, ", sequentialSegmentIndex = ", sequentialSegmentIndex)
			if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
				fileName = generateBiologicalSimulationDynamicFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.drawHopfieldGraphSentence(sentenceConceptNodeList, wTargetDebug)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
				fileName = generateBiologicalSimulationDynamicFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.drawHopfieldGraphNetwork(networkConceptNodeDict, wTargetDebug)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)				

def drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget=None):
	if(drawBiologicalSimulationDynamic):
		if(not debugCalculateNeuronActivationParallel or (sentenceIndex == sentenceIndexDebug and wSource == wSourceDebug)):
			if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
				fileName = generateBiologicalSimulationFileName(True, wSource, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.drawHopfieldGraphSentence(sentenceConceptNodeList, wTargetDebug)
				HFNLPpy_biologicalSimulationDrawSentenceDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
			if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
				fileName = generateBiologicalSimulationFileName(False, wSource, sentenceIndex)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.clearHopfieldGraph()
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.drawHopfieldGraphNetwork(networkConceptNodeDict, wTargetDebug)
				HFNLPpy_biologicalSimulationDrawNetworkDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)		

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

