"""HFNLPpy_biologicalSimulationPropagateVectorised.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

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
		drawBiologicalSimulationDynamic = False	#mandatory: False

	
printVerbose = False

debugCalculateNeuronActivationParallel = False
if(debugCalculateNeuronActivationParallel):
	wSourceDebug = 5
	wTargetDebug = wSourceDebug+1

#parameters only used for drawBiologicalSimulationDynamic: sentenceIndex, sentenceConceptNodeList
def simulateBiologicalHFnetworkSequenceNodePropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, w, conceptNeuron, connectionTargetNeuronSet):
	conceptNeuronSourceList = []
	conceptNeuronSourceList.append(conceptNeuronSource)
	return simulateBiologicalHFnetworkSequenceNodesPropagateParallel(networkConceptNodeDict, conceptNeuronSourceList, activationTime, w, conceptNeuron, connectionTargetNeuronSet, wSource, sentenceIndex, sentenceConceptNodeList)
	
#parameters only used for drawBiologicalSimulationDynamic: wSource, sentenceIndex, sentenceConceptNodeList
def simulateBiologicalHFnetworkSequenceNodesPropagateParallel(networkConceptNodeDict, conceptNeuronSourceList, activationTime, w, conceptNeuron, connectionTargetNeuronSet, wSource=None, sentenceIndex=None, sentenceConceptNodeList=None):
	
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

		#if(printVerbose):
		print("simulateBiologicalHFnetworkSequenceNodePropagateParallel: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName)
	
		if(updateNeuronObjectActivationLevels):
			conceptNeuronSource.activationLevel = objectAreaActivationLevelOn

		for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():

			#add target neuron to batch processing tensor
			#if(vectoriseComputationIndependentBranches):	#only coded algorithm
			conceptNeuronTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
			if(conceptNeuronTarget not in batchNeuronsList):
			
				connectionTargetNeuronSet.add(conceptNeuronTarget)
				batchNeuronsList.append(conceptNeuronTarget)

				if(targetConnectionConceptName == conceptNeuron.nodeName):
					conceptNeuronBatchIndex = batchIndex
					conceptNeuronBatchIndexFound = True
					#print("conceptNeuron.nodeName = ", conceptNeuron.nodeName)
					#print("conceptNeuronBatchIndex = ", conceptNeuronBatchIndex)
				batchIndex += 1
				
				#create temporary vectorised buffers for conceptNeuronSource connection target input sequentialSegment candidate application;
				conceptNeuronTarget.vectorisedBranchActivationLevelListBuffer, conceptNeuronTarget.vectorisedBranchActivationTimeListBuffer = createDendriticTreeVectorised(batched=False, createVectorisedBranchObjectList=False, storeSequentialSegmentInputActivationLevels=vectoriseComputionUseSequentialSegmentInputActivationLevels)	#shape [numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments, {numberOfSequentialSegmentInputs}]
				
			#trigger all target synaptic inputs before parallel processing	
			for connection in connectionList:
				if(updateNeuronObjectActivationLevels):
					connection.activationLevel = objectAreaActivationLevelOn
				setVectorisedBranchActivation(conceptNeuronTarget, connection, activationTime)

	batchNeuronsList2 = []
	for conceptNeuronSource in conceptNeuronSourceList:
		for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():
			conceptNeuronTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
			if(conceptNeuronTarget not in batchNeuronsList2):
				batchNeuronsList2.append(conceptNeuronTarget)
				for branchIndex1 in range(numberOfVerticalBranches):
					vectorisedBranchActivationLevelBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1])
					vectorisedBranchActivationTimeBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationTimeList[branchIndex1])
					vectorisedBranchActivationLevelBatchListListBuffer[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1])
					vectorisedBranchActivationTimeBatchListListBuffer[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1])
					if(recordVectorisedBranchObjectList):
						vectorisedBranchObjectBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchObjectList[branchIndex1])			

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
		
	if(debugCalculateNeuronActivationParallel):	
		if(wSource==wSourceDebug and w==wTargetDebug):
			for branchIndex1 in range(numberOfVerticalBranches):
				print("\t(wSource==wSourceDebug and wTarget==wTargetDebug): branchIndex1 = ", branchIndex1)
				print("\tvectorisedBranchActivationLevelBatchList[branchIndex1] = ", vectorisedBranchActivationLevelBatchList[branchIndex1])

	if(conceptNeuronBatchIndexFound):	#optimsation; only execute calculateNeuronActivationParallel if conceptNeuron input(s) are activated by conceptNeuronSource
		if(calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationLevelBatchListBuffer, vectorisedBranchActivationTimeBatchListBuffer, vectorisedBranchObjectBatchList, activationTime, w, conceptNeuron, conceptNeuronBatchIndex, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)):
			somaActivationFound = True
	else:
		print("warning !conceptNeuronBatchIndexFound")
	
	#save updated activations (ideally these should be able to be dynamically updated by calculateNeuronActivationParallel; store tensors (memory/reference) in a bulk/stacked tensor, write to the bulk tensor and have the individual tensors updated)
	for batchIndex, batchNeuron in enumerate(batchNeuronsList):
		for branchIndex1 in range(numberOfVerticalBranches):
			#iterating over batchSize to save tensors is slow and may require optimisation
			batchNeuron.vectorisedBranchActivationLevelList[branchIndex1] = vectorisedBranchActivationLevelBatchList[branchIndex1][batchIndex]
			batchNeuron.vectorisedBranchActivationTimeList[branchIndex1] = vectorisedBranchActivationTimeBatchList[branchIndex1][batchIndex]

	if(updateNeuronObjectActivationLevels):
		for conceptNeuronSource in conceptNeuronSourceList:
			resetAxonsActivation(conceptNeuronSource)
			if(resetWsourceNeuronDendriteAfterActivation):
				resetDendriticTreeActivation(conceptNeuronSource)
				
	return somaActivationFound
	
def emptyList(lst):
	result = False
	if(len(lst) == 0):
		result = True
	return result
		
def setVectorisedBranchActivation(conceptNeuronTarget, connection, activationTime):
	
	currentSequentialSegmentInput = connection.nodeTargetSequentialSegmentInput
	currentSequentialSegment = currentSequentialSegmentInput.sequentialSegment
	currentBranch = currentSequentialSegment.branch

	branchIndex1 = currentBranch.branchIndex1
	branchIndex2 = currentBranch.branchIndex2 	#local horizontalBranchIndex (wrt horizontalBranchWidth)
	horizontalBranchIndex = currentBranch.horizontalBranchIndex	#absolute horizontalBranchIndex	#required by vectoriseComputationCurrentDendriticInput only
	currentSequentialSegmentIndex = currentSequentialSegment.sequentialSegmentIndex
	if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
		currentSequentialSegmentInputIndex = currentSequentialSegmentInput.sequentialSegmentInputIndex
			
	activationValue = generateVectorisedSequentialSegmentInputActivationBias(currentSequentialSegmentInput)	#artificially increase activation value of first inputs in sequence
	#print("activationValue = ", activationValue)
	#print("activationTime = ", activationTime)
	#print("currentSequentialSegment.nodeName = ", currentSequentialSegment.nodeName)
	
	if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
		#if(verifyRepolarised(currentSequentialSegmentIndex, activationTime, currentSequentialSegmentInput.activationTime)):
		if(performSummationOfSequentialSegmentInputs):
			activationValue = activationValue * calculateInputActivationLevel(connection)
		if(updateNeuronObjectActivationLevels):
			if(performSummationOfSequentialSegmentInputs):
				currentSequentialSegmentInput.activationLevel = activationValue	#CHECKTHIS: or connection.weight (disregarding firstInputInSequence activation level modifier)
			else:
				currentSequentialSegmentInput.activationLevel = objectLocalActivationLevelOn	#True
			currentSequentialSegmentInput.activationTime = activationTime	
		conceptNeuronTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationValue)
		conceptNeuronTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationTime)	
	else:
		if(performSummationOfSequentialSegmentInputs):
			activationValue = activationValue * calculateInputActivationLevel(connection)
		if(updateNeuronObjectActivationLevels):
			if(performSummationOfSequentialSegmentInputs):
				currentSequentialSegmentInput.activationLevel = activationValue		#CHECKTHIS: or connection.weight (disregarding firstInputInSequence activation level modifier)
			else:
				currentSequentialSegmentInput.activationLevel = objectLocalActivationLevelOn	#True
			currentSequentialSegmentInput.activationTime = activationTime
		if(performSummationOfSequentialSegmentInputs):
			summationOfSequentialSegmentInputs = conceptNeuronTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].numpy()
			summationOfSequentialSegmentInputs = summationOfSequentialSegmentInputs + activationValue
			activationValue = summationOfSequentialSegmentInputs	
			#overwrite activationTime for input (all inputs should have same activation time)
		#if(verifyRepolarised(currentSequentialSegment, activationTime)):
		conceptNeuronTarget.vectorisedBranchActivationLevelListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationValue)
		conceptNeuronTarget.vectorisedBranchActivationTimeListBuffer[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex].assign(activationTime)
			
def generateVectorisedSequentialSegmentInputActivationBias(currentSequentialSegmentInput):
	#artificially increase activation value of first inputs in sequence
	if(currentSequentialSegmentInput.firstInputInSequence):
		activationValue = vectorisedActivationLevelOnFirstInputInSequence
	else:
		activationValue = vectorisedActivationLevelOn
	return activationValue
						

#parameters only used for drawBiologicalSimulationDynamic: wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList
#does not currently support vectoriseComputionUseSequentialSegmentInputActivationLevels;
def calculateNeuronActivationParallel(vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchActivationLevelBatchListBuffer, vectorisedBranchActivationTimeBatchListBuffer, vectorisedBranchObjectBatchList, activationTime, wTarget, conceptNeuronTarget, conceptNeuronBatchIndex, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):
	print("calculateNeuronActivationParallel:")
	
	#vectorisedBranchActivationLevelBatchList/vectorisedBranchActivationTimeBatchList: list of tensors for every branchIndex1 - each element is of shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments], each batch sample refers to a unique target concept
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 	#len(vectorisedBranchActivationLevelBatchList)
	
	vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious = (None, None)
	for branchIndex1 in reversed(range(numberOfVerticalBranches)):
			
		vectorisedBranchActivationLevelBatchBuffer = vectorisedBranchActivationLevelBatchListBuffer[branchIndex1]
		vectorisedBranchActivationTimeBatchBuffer = vectorisedBranchActivationTimeBatchListBuffer[branchIndex1]
		vectorisedBranchActivationLevelBatch = vectorisedBranchActivationLevelBatchList[branchIndex1]
		vectorisedBranchActivationTimeBatch = vectorisedBranchActivationTimeBatchList[branchIndex1]

		#print("vectorisedBranchActivationLevelBatchBuffer = ", vectorisedBranchActivationLevelBatchBuffer)
		if(recordVectorisedBranchObjectList):
			vectorisedBranchObjectBatch = vectorisedBranchObjectBatchList[branchIndex1]

		#print("\tbranchIndex1 = ", branchIndex1)
			#print("\tvectorisedBranchActivationLevelBatch = ", vectorisedBranchActivationLevelBatch)
			#print("\tvectorisedBranchActivationTimeBatch = ", vectorisedBranchActivationTimeBatch)
				
		#initialise sequential segments activation (shape: [batchSize, numberOfHorizontalBranches, horizontalBranchWidth]);
		vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationTimeBatchSequentialSegments = calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatch.shape, vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious)
		#print("vectorisedBranchActivationLevelBatchSequentialSegments = ", vectorisedBranchActivationLevelBatchSequentialSegments)
		
		#print("vectorisedBranchActivationLevelBatchSequentialSegments.shape = ", vectorisedBranchActivationLevelBatchSequentialSegments.shape)
		
		for sequentialSegmentIndex in reversed(range(numberOfBranchSequentialSegments)):
			vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatch[:, :, :, sequentialSegmentIndex]
			vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatch[:, :, :, sequentialSegmentIndex]
			vectorisedBranchActivationLevelBatchSequentialSegmentBuffer = vectorisedBranchActivationLevelBatchBuffer[:, :, :, sequentialSegmentIndex]
			vectorisedBranchActivationTimeBatchSequentialSegmentBuffer = vectorisedBranchActivationTimeBatchBuffer[:, :, :, sequentialSegmentIndex]
			if(preventReactivationOfSequentialSegments):
				if(performSummationOfSequentialSegmentInputs):
					vectorisedBranchActivationNewBatchSequentialSegmentMask = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, sequentialSegmentMinActivationLevel)	#vectorisedBranchActivationLevelListBuffer stores numeric values
				else:
					vectorisedBranchActivationNewBatchSequentialSegmentMask = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedActivationLevelOn)		#vectorisedBranchActivationLevelListBuffer stores numeric values
				vectorisedBranchActivationNewBatchSequentialSegmentMask = tf.logical_and(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.logical_not(tf.equal(vectorisedBranchActivationLevelBatchSequentialSegment, vectorisedActivationLevelOn)))	#greater_equal test not required as vectorisedBranchActivationLevelList always stores effective boolean 1/0 values (only vectorisedBranchActivationLevelListBuffer stores numeric values)
				vectorisedBranchActivationExistingBatchSequentialSegmentMask = tf.logical_not(vectorisedBranchActivationNewBatchSequentialSegmentMask)
				vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat = tf.cast(vectorisedBranchActivationNewBatchSequentialSegmentMask, tf.float32)
				vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat = tf.cast(vectorisedBranchActivationExistingBatchSequentialSegmentMask, tf.float32)
				vectorisedBranchActivationLevelBatchSequentialSegmentNew = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegmentBuffer, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)
				vectorisedBranchActivationTimeBatchSequentialSegmentNew = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegmentBuffer, vectorisedBranchActivationNewBatchSequentialSegmentMaskFloat)
				vectorisedBranchActivationLevelBatchSequentialSegmentExisting = tf.multiply(vectorisedBranchActivationLevelBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat)
				vectorisedBranchActivationTimeBatchSequentialSegmentExisting = tf.multiply(vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedBranchActivationExistingBatchSequentialSegmentMaskFloat)
				vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatchSequentialSegmentNew	#perform sequential segment calculations with new activations from buffer
				vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatchSequentialSegmentNew		#perform sequential segment calculations with new activations from buffer							
			else:
				vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatchSequentialSegmentBuffer	#overwrite current sequential segment with buffer
				vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatchSequentialSegmentBuffer	#overwrite current sequential segment with buffer
						
			vectorisedBranchActivationLevelBatchSequentialSegments = tf.add(vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationLevelBatchSequentialSegment)	#note if firstSequentialSegmentInSequence (ie sequentialSegmentActivationLevel=2), and higher branch input is zero, then sequential segment will still activate
			vectorisedBranchActivationLevelBatchSequentialSegments = tf.greater(vectorisedBranchActivationLevelBatchSequentialSegments, 1)	#or greater_equal(vectorisedBranchActivationLevelBatchSequentialSegments, 2)
			vectorisedBranchActivationLevelBatchSequentialSegmentFirstInputInSequence = tf.equal(vectorisedBranchActivationLevelBatchSequentialSegment, vectorisedActivationLevelOnFirstInputInSequence)
			if(algorithmTimingWorkaround1):
				vectorisedBranchActivationLevelBatchSequentialSegmentsTimeTests = tf.greater_equal(vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedBranchActivationTimeBatchSequentialSegments)	#ensure activationTime of sequentialSegment is greater than that of previous sequential segments/subbranch - equivalent to verifySequentialActivationTime			
			else:
				vectorisedBranchActivationLevelBatchSequentialSegmentsTimeTests = tf.greater(vectorisedBranchActivationTimeBatchSequentialSegment, vectorisedBranchActivationTimeBatchSequentialSegments)	#ensure activationTime of sequentialSegment is greater than that of previous sequential segments/subbranch - equivalent to verifySequentialActivationTime
			vectorisedBranchActivationLevelBatchSequentialSegmentsTimeTests = tf.logical_or(vectorisedBranchActivationLevelBatchSequentialSegmentsTimeTests, vectorisedBranchActivationLevelBatchSequentialSegmentFirstInputInSequence)
			vectorisedBranchActivationLevelBatchSequentialSegments = tf.logical_and(vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationLevelBatchSequentialSegmentsTimeTests)
			vectorisedBranchActivationLevelBatchSequentialSegments = tf.cast(vectorisedBranchActivationLevelBatchSequentialSegments, tf.float32)
			vectorisedBranchActivationTimeBatchSequentialSegments = vectorisedBranchActivationTimeBatchSequentialSegment

			if(preventReactivationOfSequentialSegments):
				vectorisedBranchActivationLevelBatchSequentialSegments = tf.add(vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationLevelBatchSequentialSegmentExisting)
				vectorisedBranchActivationTimeBatchSequentialSegments = tf.add(vectorisedBranchActivationTimeBatchSequentialSegments, vectorisedBranchActivationTimeBatchSequentialSegmentExisting)
			vectorisedBranchActivationLevelBatchList[branchIndex1][:, :, :, sequentialSegmentIndex].assign(vectorisedBranchActivationLevelBatchSequentialSegments)
			vectorisedBranchActivationTimeBatchList[branchIndex1][:, :, :, sequentialSegmentIndex].assign(vectorisedBranchActivationTimeBatchSequentialSegments)
			#note conceptNode.vectorisedBranchActivationLevelList/vectorisedBranchActivationTimeList will be updated at end of simulateBiologicalHFnetworkSequenceNodesPropagateParallel based on vectorisedBranchActivationLevelBatchList/vectorisedBranchActivationTimeBatchList
			
			if(updateNeuronObjectActivationLevels):	#OLD: if(drawBiologicalSimulationDynamic):
				#if(sentenceIndex == 3):	#temp debug requirement	#and wTarget==15
				vectorisedBranchObjectBatchSequentialSegment = vectorisedBranchObjectBatch[:, :, :, sequentialSegmentIndex]	#requires recordVectorisedBranchObjectList
				for batchIndex in range(vectorisedBranchObjectBatchSequentialSegment.shape[0]):
					for horizontalBranchIndex in range(vectorisedBranchObjectBatchSequentialSegment.shape[1]):
						for branchIndex2 in range(vectorisedBranchObjectBatchSequentialSegment.shape[2]):
							sequentialSegment = vectorisedBranchObjectBatchSequentialSegment[batchIndex, horizontalBranchIndex, branchIndex2]
							activationLevel = vectorisedBranchActivationLevelBatchSequentialSegments[batchIndex, horizontalBranchIndex, branchIndex2].numpy()
							activationTimeSeg = vectorisedBranchActivationTimeBatchSequentialSegments[batchIndex, horizontalBranchIndex, branchIndex2].numpy()
							activationLevel = bool(activationLevel)
							#print("activationLevel = ", activationLevel)
							#print("sequentialSegment.branch.branchIndex1 = ", sequentialSegment.branch.branchIndex1)
							#print("sequentialSegment.nodeName = ", sequentialSegment.nodeName)
							#print("\t\tactivationLevel = ", activationLevel)
							sequentialSegment.activationLevel = bool(activationLevel)
							if(activationLevel):
								sequentialSegment.activationTime = activationTimeSeg
								#print("activationTimeSeg = ", activationTimeSeg)
							if(sequentialSegmentIndex == 0):
								sequentialSegment.branch.activationLevel = sequentialSegment.activationLevel
	
			drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex)			
				
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

	if(updateNeuronObjectActivationLevels):	#OLD: if(drawBiologicalSimulationDynamic):
		#if(recordVectorisedBranchObjectList):
		for batchIndex in range(vectorisedBranchObjectBatchList[0].shape[0]):		#or vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious.shape[0]
			vectorisedBranchObjectBatchSequentialSegment = vectorisedBranchObjectBatchList[0][batchIndex, 0, 0, 0]	#get any (first) sequential segment object in batchIndex neuron
			batchNeuron = vectorisedBranchObjectBatchSequentialSegment.conceptNode
			somaActivationLevel = vectorisedSomaActivationLevel[batchIndex].numpy()
			somaActivationLevel = bool(somaActivationLevel)
			batchNeuron.activationLevel = somaActivationLevel
	
	drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	

	somaActivationFound = vectorisedSomaActivationLevel[conceptNeuronBatchIndex].numpy()	#is target neuron activated?
	#print("somaActivationFound = ", somaActivationFound)
	somaActivationFound = bool(somaActivationFound)

						
	return somaActivationFound

def calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatchShape, vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious):
	#initialise sequential segments activation;
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 
	if(branchIndex1 == numberOfVerticalBranches-1):
		#highest branch in dendritic tree (initialise activation to true)
		vectorisedBranchActivationLevelBatchSequentialSegments = tf.ones((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2]))	#use every dimension except sequentialSegmentIndex 
		vectorisedBranchActivationTimeBatchSequentialSegments = tf.negative(tf.ones((vectorisedBranchActivationLevelBatchShape[0], vectorisedBranchActivationLevelBatchShape[1], vectorisedBranchActivationLevelBatchShape[2]))) 	#use every dimension except sequentialSegmentIndex	#initial activation time of dendritic leaf nodes set artificially low such that passSegmentActivationTimeTests automatically pass
	else:
		#intermediary branch in dendritic tree (initialise activation to that of higher branch)
		vectorisedBranchActivationLevelBatchSequentialSegmentsPreviousSummed = tf.reduce_sum(vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, axis=2)
		vectorisedBranchActivationTimeBatchSequentialSegmentsPreviousMax = tf.reduce_max(vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious, axis=2)
		vectorisedBranchActivationLevelBatchSequentialSegments = tf.greater_equal(vectorisedBranchActivationLevelBatchSequentialSegmentsPreviousSummed, numberOfHorizontalSubBranchesRequiredForActivation)
		vectorisedBranchActivationTimeBatchSequentialSegments = vectorisedBranchActivationTimeBatchSequentialSegmentsPreviousMax
		vectorisedBranchActivationLevelBatchSequentialSegments = tf.cast(vectorisedBranchActivationLevelBatchSequentialSegments, tf.float32)
		vectorisedBranchActivationTimeBatchSequentialSegments = vectorisedBranchActivationTimeBatchSequentialSegments
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

def drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1, sequentialSegmentIndex):
	if(drawBiologicalSimulationDynamic):
		#if(sentenceIndex == 3):	#temp debug requirement	#and wTarget==15
		if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
			fileName = generateBiologicalSimulationDynamicFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
			HFNLPpy_biologicalSimulationDrawSentenceDynamic.clearHopfieldGraph()
			HFNLPpy_biologicalSimulationDrawSentenceDynamic.drawHopfieldGraphSentence(sentenceConceptNodeList)
			HFNLPpy_biologicalSimulationDrawSentenceDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
		if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
			fileName = generateBiologicalSimulationDynamicFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
			HFNLPpy_biologicalSimulationDrawNetworkDynamic.clearHopfieldGraph()
			HFNLPpy_biologicalSimulationDrawNetworkDynamic.drawHopfieldGraphNetwork(networkConceptNodeDict)
			HFNLPpy_biologicalSimulationDrawNetworkDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)				

def drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):
	if(drawBiologicalSimulationDynamic):
		#if(sentenceIndex == 3):	#temp debug requirement	#and wTarget==15
		if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
			fileName = generateBiologicalSimulationFileName(True, wSource, sentenceIndex)
			HFNLPpy_biologicalSimulationDrawSentenceDynamic.clearHopfieldGraph()
			HFNLPpy_biologicalSimulationDrawSentenceDynamic.drawHopfieldGraphSentence(sentenceConceptNodeList)
			HFNLPpy_biologicalSimulationDrawSentenceDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
		if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
			generateBiologicalSimulationFileName(False, wSource, sentenceIndex)
			HFNLPpy_biologicalSimulationDrawNetworkDynamic.clearHopfieldGraph()
			HFNLPpy_biologicalSimulationDrawNetworkDynamic.drawHopfieldGraphNetwork(networkConceptNodeDict)
			HFNLPpy_biologicalSimulationDrawNetworkDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)		

