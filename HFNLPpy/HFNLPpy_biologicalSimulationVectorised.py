"""HFNLPpy_biologicalSimulationVectorised.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Vectorised

"""



import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *

	

#if(biologicalSimulationForward):	#required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic
if(vectoriseComputation):		#dynamic draw should use vectoriseComputation, as this activates all target neuron synapses of wSource simultaneously 
	drawBiologicalSimulationDynamic = True	#draw dynamic activation levels of biological simulation (save to file)
	if(drawBiologicalSimulationDynamic):
		drawBiologicalSimulationDynamicPlot = True	#default: False
		drawBiologicalSimulationDynamicSave = True	#default: True	#save to file
		drawBiologicalSimulationDendriticTreeSentenceDynamic = True	#default: True	#draw graph for sentence neurons and their dendritic tree
		if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
			import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentenceDynamic
		drawBiologicalSimulationDendriticTreeNetworkDynamic = False	#default: True	#draw graph for entire network (not just sentence)
		if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
			import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetworkDynamic


def generateBiologicalSimulationFileName(sentenceOrNetwork, wSource, sentenceIndex=None):
	fileName = "biologicalSimulationDynamic"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + str(sentenceIndex)
	fileName = fileName + "Wsource" + str(wSource)
	return fileName

def generateBiologicalSimulationDynamicFileName(sentenceOrNetwork, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex=None):
	fileName = "biologicalSimulationDynamic"
	if(sentenceOrNetwork):
		fileName = fileName + "Sentence"
	else:
		fileName = fileName + "Network"
		fileName = fileName + "sentenceIndex" + str(sentenceIndex)
	fileName = fileName + "Wsource" + str(wSource)
	fileName = fileName + "branchIndex1" + str(branchIndex1)
	fileName = fileName + "sequentialSegmentIndex" + str(sequentialSegmentIndex)
	return fileName
	
			
printVerbose = False

debugCalculateNeuronActivationParallel = False
if(debugCalculateNeuronActivationParallel):
	wSourceDebug = 5
	wTargetDebug = wSourceDebug+1

def simulateBiologicalHFnetworkSequenceNodeTrainParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, w, conceptNeuron):
	
	#construct batch dendritic tree templates for parallel processing;
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)
	vectorisedBranchActivationLevelBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationTimeBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	if(recordVectorisedBranchObjectList):
		vectorisedBranchObjectBatchListList = [[] for _ in range(numberOfVerticalBranches)]	#temporary list before being coverted to tensor for parallel processing
	vectorisedBranchActivationLevelBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	vectorisedBranchActivationTimeBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	if(recordVectorisedBranchObjectList):
		vectorisedBranchObjectBatchList = [None for _ in range(numberOfVerticalBranches)]	#[]*(numberOfVerticalBranches)
	else:
		vectorisedBranchObjectBatchList = None
		
	#if(printVerbose):
	print("simulateBiologicalHFnetworkSequenceNodeTrainParallel: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName)

	#vectorisedBranchInputBatchList = []	#list of tensors for every branchIndex1	- filled by createDendriticTreeVectorised; each element is of shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments]

	activationTime = calculateActivationTime(sentenceIndex)
	conceptNeuronSource.activationLevel = True
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
	
	batchNeuronsList = []
	
	batchIndex = 0	#batchSampleIndex
	conceptNeuronBatchIndex = None
	conceptNeuronBatchIndexFound = False
	for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():
		#add target neuron to batch processing tensor
		#if(vectoriseComputationIndependentBranches):	#only coded algorithm
		conceptNeuronTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
		batchNeuronsList.append(conceptNeuronTarget)
		for connection in connectionList:
			connection.activationLevel = True
			#trigger all target synaptic inputs before parallel processing
			currentSequentialSegmentInput = connection.nodeTargetSequentialSegmentInput
			setVectorisedBranchActivation(conceptNeuronTarget, currentSequentialSegmentInput, activationTime)
				
		for branchIndex1 in range(numberOfVerticalBranches):
			vectorisedBranchActivationLevelBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1])
			vectorisedBranchActivationTimeBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchActivationTimeList[branchIndex1])
			if(recordVectorisedBranchObjectList):
				obj = conceptNeuronTarget.vectorisedBranchObjectList[branchIndex1]
				#print("conceptNeuronTarget.nodeName = ", conceptNeuronTarget.nodeName)
				#print("obj = ", obj)
				vectorisedBranchObjectBatchListList[branchIndex1].append(conceptNeuronTarget.vectorisedBranchObjectList[branchIndex1])			
		if(targetConnectionConceptName == conceptNeuron.nodeName):
			conceptNeuronBatchIndex = batchIndex
			conceptNeuronBatchIndexFound = True
			#print("conceptNeuron.nodeName = ", conceptNeuron.nodeName)
			#print("conceptNeuronBatchIndex = ", conceptNeuronBatchIndex)
		batchIndex += 1
			
	for branchIndex1 in range(numberOfVerticalBranches):
		vectorisedBranchActivationLevelBatchList[branchIndex1] = tf.stack(vectorisedBranchActivationLevelBatchListList[branchIndex1])
		vectorisedBranchActivationTimeBatchList[branchIndex1] = tf.stack(vectorisedBranchActivationTimeBatchListList[branchIndex1])	
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
		if(calculateNeuronActivationParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchObjectBatchList, activationTime, w, conceptNeuron, conceptNeuronBatchIndex, wSource, batchNeuronsList)):
			somaActivationFound = True
	else:
		print("warning !conceptNeuronBatchIndexFound")
	
	resetAxonsActivation(conceptNeuronSource)

	return somaActivationFound
	
def emptyList(lst):
	result = False
	if(len(lst) == 0):
		result = True
	return result
		
def setVectorisedBranchActivation(conceptNeuronTarget, currentSequentialSegmentInput, activationTime):
	
	currentSequentialSegment = currentSequentialSegmentInput.sequentialSegment
	currentBranch = currentSequentialSegment.branch

	branchIndex1 = currentBranch.branchIndex1
	branchIndex2 = currentBranch.branchIndex2 	#local horizontalBranchIndex (wrt horizontalBranchWidth)
	horizontalBranchIndex = currentBranch.horizontalBranchIndex	#absolute horizontalBranchIndex	#required by vectoriseComputationCurrentDendriticInput only
	currentSequentialSegmentIndex = currentSequentialSegment.sequentialSegmentIndex
	if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
		currentSequentialSegmentInputIndex = currentSequentialSegmentInput.sequentialSegmentInputIndex
			
	activationValue = generateSequentialSegmentInputActivationValue(currentSequentialSegmentInput)
	#print("activationValue = ", activationValue)
	
	if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
		#if(verifyRepolarised(currentSequentialSegmentIndex, activationTime, currentSequentialSegmentInput.activationTime)):
		currentSequentialSegmentInput.activationLevel = True
		currentSequentialSegmentInput.activationTime = activationTime	
		vectorisedBranchActivationLevel = conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1]
		vectorisedBranchActivationTime = conceptNeuronTarget.vectorisedBranchActivationLevelList[branchIndex1]
		conceptNeuronTarget.vectorisedBranchActivationLevel[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationValue)
		conceptNeuronTarget.vectorisedBranchActivationTime[branchIndex1][horizontalBranchIndex, branchIndex2, currentSequentialSegmentIndex, currentSequentialSegmentInputIndex].assign(activationTime)	
	else:
		if(recordSequentialSegmentInputActivationLevels):
			currentSequentialSegmentInput.activationLevel = True
			currentSequentialSegmentInput.activationTime = activationTime
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
						

#does not currently support vectoriseComputionUseSequentialSegmentInputActivationLevels;
def calculateNeuronActivationParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, vectorisedBranchActivationLevelBatchList, vectorisedBranchActivationTimeBatchList, vectorisedBranchObjectBatchList, activationTime, wTarget, conceptNeuronTarget, conceptNeuronBatchIndex, wSource, batchNeuronsList):
	print("calculateNeuronActivationParallel:")
	
	#vectorisedBranchActivationLevelBatchList/vectorisedBranchActivationTimeBatchList: list of tensors for every branchIndex1 - each element is of shape [batchSize, numberOfHorizontalBranches, horizontalBranchWidth, numberOfBranchSequentialSegments], each batch sample refers to a unique target concept
	numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1) 	#len(vectorisedBranchActivationLevelBatchList)
	
	vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious = (None, None)
	for branchIndex1 in reversed(range(numberOfVerticalBranches)):
			
		vectorisedBranchActivationLevelBatch = vectorisedBranchActivationLevelBatchList[branchIndex1]
		vectorisedBranchActivationTimeBatch = vectorisedBranchActivationTimeBatchList[branchIndex1]
		if(recordVectorisedBranchObjectList):
			vectorisedBranchObjectBatch = vectorisedBranchObjectBatchList[branchIndex1]

		print("\tbranchIndex1 = ", branchIndex1)
		#print("\tvectorisedBranchActivationLevelBatch = ", vectorisedBranchActivationLevelBatch)
		#print("\tvectorisedBranchActivationTimeBatch = ", vectorisedBranchActivationTimeBatch)
				
		#initialise sequential segments activation (shape: [batchSize, numberOfHorizontalBranches, horizontalBranchWidth]);
		vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationTimeBatchSequentialSegments = calculateSequentialSegmentsInitialActivationFromHigherBranchParallel(branchIndex1, vectorisedBranchActivationLevelBatch.shape, vectorisedBranchActivationLevelBatchSequentialSegmentsPrevious, vectorisedBranchActivationTimeBatchSequentialSegmentsPrevious)
		#print("vectorisedBranchActivationLevelBatchSequentialSegments = ", vectorisedBranchActivationLevelBatchSequentialSegments)
		
		for sequentialSegmentIndex in reversed(range(numberOfBranchSequentialSegments)):
			vectorisedBranchActivationLevelBatchSequentialSegment = vectorisedBranchActivationLevelBatch[:, :, :, sequentialSegmentIndex]
			vectorisedBranchActivationTimeBatchSequentialSegment = vectorisedBranchActivationTimeBatch[:, :, :, sequentialSegmentIndex]
			print("vectorisedBranchActivationLevelBatchSequentialSegment = ", vectorisedBranchActivationLevelBatchSequentialSegment)
			print("vectorisedBranchActivationLevelBatchSequentialSegments = ", vectorisedBranchActivationLevelBatchSequentialSegments)
			vectorisedBranchActivationLevelBatchSequentialSegments = tf.add(vectorisedBranchActivationLevelBatchSequentialSegments, vectorisedBranchActivationLevelBatchSequentialSegment)	#note if firstSequentialSegmentInSequence (ie sequentialSegmentActivationLevel=2), and higher branch input is zero, then sequential segment will still activate
			vectorisedBranchActivationTimeBatchSequentialSegments = tf.add(vectorisedBranchActivationTimeBatchSequentialSegments, vectorisedBranchActivationTimeBatchSequentialSegment)
			print("2 vectorisedBranchActivationLevelBatchSequentialSegments = ", vectorisedBranchActivationLevelBatchSequentialSegments)
			vectorisedBranchActivationLevelBatchSequentialSegments = tf.cast(tf.greater(vectorisedBranchActivationLevelBatchSequentialSegments, 1), tf.float32)	#or greater_equal(vectorisedBranchActivationLevelBatchSequentialSegments, 2)
			vectorisedBranchActivationTimeBatchSequentialSegments = tf.cast(tf.greater(vectorisedBranchActivationTimeBatchSequentialSegments, 1), tf.float32)	#or greater_equal(vectorisedBranchActivationLevelBatchSequentialSegments, 2)
			print("Posthoc: vectorisedBranchActivationLevelBatchSequentialSegments = ", vectorisedBranchActivationLevelBatchSequentialSegments)
			
			if(drawBiologicalSimulationDynamic):
				if(recordVectorisedBranchObjectList):
					vectorisedBranchObjectBatchSequentialSegment = vectorisedBranchObjectBatch[:, :, :, sequentialSegmentIndex]
				for batchIndex in range(vectorisedBranchObjectBatchSequentialSegment.shape[0]):
					for horizontalBranchIndex in range(vectorisedBranchObjectBatchSequentialSegment.shape[1]):
						for branchIndex2 in range(vectorisedBranchObjectBatchSequentialSegment.shape[2]):
							sequentialSegment = vectorisedBranchObjectBatchSequentialSegment[batchIndex, horizontalBranchIndex, branchIndex2]
							print("sequentialSegment.branch.branchIndex1 = ", sequentialSegment.branch.branchIndex1)
							#print("sequentialSegment.nodeName = ", sequentialSegment.nodeName)
							activationLevel = vectorisedBranchActivationLevelBatchSequentialSegments[batchIndex, horizontalBranchIndex, branchIndex2].numpy()
							#print("\t\tactivationLevel = ", activationLevel)
							sequentialSegment.activationLevel = bool(activationLevel)
							if(sequentialSegmentIndex == 0):
								sequentialSegment.branch.activationLevel = sequentialSegment.activationLevel
					
				if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
					fileName = generateBiologicalSimulationDynamicFileName(True, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
					HFNLPpy_biologicalSimulationDrawSentenceDynamic.clearHopfieldGraph()
					HFNLPpy_biologicalSimulationDrawSentenceDynamic.drawHopfieldGraphSentence(sentenceConceptNodeList)
					HFNLPpy_biologicalSimulationDrawSentenceDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)
				if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
					generateBiologicalSimulationDynamicFileName(False, wSource, branchIndex1, sequentialSegmentIndex, sentenceIndex)
					HFNLPpy_biologicalSimulationDrawNetworkDynamic.clearHopfieldGraph()
					HFNLPpy_biologicalSimulationDrawNetworkDynamic.drawHopfieldGraphNetwork(networkConceptNodeDict)
					HFNLPpy_biologicalSimulationDrawNetworkDynamic.displayHopfieldGraph(drawBiologicalSimulationDynamicPlot, drawBiologicalSimulationDynamicSave, fileName)				
				
				
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

	for batchIndex, batchNeuron in enumerate(batchNeuronsList):
		somaActivationLevel = vectorisedSomaActivationLevel[batchIndex].numpy()
		somaActivationLevel = bool(somaActivationLevel)
		batchNeuron.activationLevel = somaActivationLevel
		
		if(drawBiologicalSimulationDynamic):
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
