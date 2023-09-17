"""HFNLPpy_DendriticSANIPropagateStandard.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI Propagate Standard

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_DendriticSANIGlobalDefs import *
from HFNLPpy_DendriticSANINode import *
import HFNLPpy_DendriticSANIDraw

printVerbose = False
printConnectionTargetActivations = False

	
#emulateVectorisedComputationOrder:
			
def simulateBiologicalHFnetworkSequenceNodesPropagateStandardEmulateVectorisedComputationOrder(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet):
	
	somaActivationFound = False
	
	if(verifyFindTargetConnection(conceptNeuronSourceList)):
		connectionTargetActivationFoundSet = set()
		numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)

		branchSequence = range(numberOfVerticalBranches)
		if(emulateVectorisedComputationOrderReversed):
			branchSequence = reversed(branchSequence)

		for branchIndex1Target in branchSequence:

			sequentialSegmentSequence = range(numberOfBranchSequentialSegments)	
			if(emulateVectorisedComputationOrderReversed):
				sequentialSegmentSequence = reversed(sequentialSegmentSequence)

			for sequentialSegmentIndexTarget in sequentialSegmentSequence:
				for conceptNeuronSource in conceptNeuronSourceList:
					if(simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, branchIndex1Target, sequentialSegmentIndexTarget, connectionTargetActivationFoundSet)):
						somaActivationFound = True
				HFNLPpy_DendriticSANIDraw.drawBiologicalDendriticSANISequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1Target, sequentialSegmentIndexTarget, activationTime, wTarget=wTarget)

		if(activateTargetConnectionSomas(conceptNeuronSourceList, networkConceptNodeDict, conceptNeuronTarget, connectionTargetActivationFoundSet)):
			somaActivationFound = True
		 
		HFNLPpy_DendriticSANIDraw.drawDendriticSANIDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=wTarget)

		for conceptNeuronSource in conceptNeuronSourceList:
			resetSourceNeuronAfterActivation(conceptNeuronSource)
			
	return somaActivationFound
	
def simulateBiologicalHFnetworkSequenceNodePropagateStandardEmulateVectorisedComputationOrder(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet):
	somaActivationFound = False
	if(verifyFindTargetConnection([conceptNeuronSource])):
	
		connectionTargetActivationFoundSet = set()
		numberOfVerticalBranches = calculateNumberOfVerticalBranches(numberOfBranches1)

		branchSequence = range(numberOfVerticalBranches)
		if(emulateVectorisedComputationOrderReversed):
			branchSequence = reversed(branchSequence)

		for branchIndex1Target in branchSequence:

			sequentialSegmentSequence = range(numberOfBranchSequentialSegments)	
			if(emulateVectorisedComputationOrderReversed):
				sequentialSegmentSequence = reversed(sequentialSegmentSequence)

			for sequentialSegmentIndexTarget in sequentialSegmentSequence:
				if(simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, branchIndex1Target, sequentialSegmentIndexTarget, connectionTargetActivationFoundSet)):
					somaActivationFound = True			
				HFNLPpy_DendriticSANIDraw.drawBiologicalDendriticSANISequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, branchIndex1Target, sequentialSegmentIndexTarget, activationTime, wTarget=wTarget)

		if(activateTargetConnectionSomas([conceptNeuronSource], networkConceptNodeDict, conceptNeuronTarget, connectionTargetActivationFoundSet)):
			somaActivationFound = True
		
		HFNLPpy_DendriticSANIDraw.drawDendriticSANIDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=wTarget)

		resetSourceNeuronAfterActivation(conceptNeuronSource)
	
	return somaActivationFound

def verifyFindTargetConnection(conceptNeuronSourceList):
	targetConnectionFound = False
	if(emulateVectorisedComputationOrderVerifyTargetConnectionFound):
		for conceptNeuronSource in conceptNeuronSourceList:
			if(conceptNeuronSource.HFcontextTargetConnectionMultiDict):	#dict not empty
			#for targetConnectionConceptName, connectionList in conceptNeuronSource.HFcontextTargetConnectionMultiDict.items():
				targetConnectionFound = True
	else:
		targetConnectionFound = True
	return targetConnectionFound

def activateTargetConnectionSomas(conceptNeuronSourceList, networkConceptNodeDict, conceptNeuronTarget, connectionTargetActivationFoundSet):
	somaActivationFound = False
	if(emulateVectorisedComputationOrderActivateSomaAfterFinishingPropagation):
		for conceptNeuronSource in conceptNeuronSourceList:
			for targetConnectionConceptName in conceptNeuronSource.HFcontextTargetConnectionMultiDict.keys():
				conceptNeuronConnectionTarget = networkConceptNodeDict[targetConnectionConceptName]
				firstBranchActivationState = conceptNeuronConnectionTarget.dendriticTree.activationLevel	#requires storeBranchActivationState
				somaActivationFoundCurrent = firstBranchActivationState
				if(applySomaActivation(conceptNeuronConnectionTarget, conceptNeuronTarget, somaActivationFoundCurrent, deactivateConnectionTargetIfSomaActivationNotFound, connectionTargetActivationFoundSet)):
					somaActivationFound = True
	return somaActivationFound	
			
def emulateVectorisedComputationOrderConnectionActivationTest(connection, branchIndex1Target, sequentialSegmentIndexTarget):
	activateConnection = True
	if(emulateVectorisedComputationOrder):
		if(connection.nodeTargetSequentialSegmentInput.sequentialSegment.branch.branchIndex1 != branchIndex1Target):
			activateConnection = False
		if(connection.nodeTargetSequentialSegmentInput.sequentialSegment.sequentialSegmentIndex != sequentialSegmentIndexTarget):
			activateConnection = False
	return activateConnection
	


def simulateBiologicalHFnetworkSequenceNodesPropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet):
	somaActivationFound = False
	for conceptNeuronSource in conceptNeuronSourceList:
		if(simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)):
			somaActivationFound = True
	return somaActivationFound
	
def simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet, branchIndex1Target=None, sequentialSegmentIndexTarget=None, connectionTargetActivationFoundSet=None):

	if(printVerbose):
		print("simulateBiologicalHFnetworkSequenceNodePropagateStandard: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
			
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
	conceptNeuronSource.activationLevel = objectAreaActivationLevelOn
	
	for targetConnectionConceptName, connectionList in conceptNeuronSource.HFcontextTargetConnectionMultiDict.items():
		conceptNeuronConnectionTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
		connectionTargetNeuronSet.add(conceptNeuronConnectionTarget)
		for connection in connectionList:
			if(emulateVectorisedComputationOrderPreactivateAxonsAndTargetInputs):
				connection.activationLevel = objectAreaActivationLevelOn
				connection.nodeTargetSequentialSegmentInput.activationLevel = calculateInputActivationLevel(connection)
			if(emulateVectorisedComputationOrderConnectionActivationTest(connection, branchIndex1Target, sequentialSegmentIndexTarget)):
				if(not emulateVectorisedComputationOrderPreactivateAxonsAndTargetInputs):
					connection.activationLevel = objectAreaActivationLevelOn
				somaActivationFoundCurrent = calculateNeuronActivationStandardWrapper(connection, 0, conceptNeuronConnectionTarget.dendriticTree, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
				if(printConnectionTargetActivations):
					print("simulateBiologicalHFnetworkSequenceNodePropagateStandard: conceptNeuronConnectionTarget = ", conceptNeuronConnectionTarget.nodeName, ", somaActivationLevel = ", somaActivationLevel)
				if(not emulateVectorisedComputationOrderActivateSomaAfterFinishingPropagation):
					if(applySomaActivation(conceptNeuronConnectionTarget, conceptNeuronTarget, somaActivationFoundCurrent, deactivateConnectionTargetIfSomaActivationNotFound, connectionTargetActivationFoundSet)):
						somaActivationFound = True
				#elif(calculateSomaActivation(conceptNeuronConnectionTarget, conceptNeuronTarget, somaActivationFoundCurrent)):	#optional (somaActivationFound is recalculated by simulateBiologicalHFnetworkSequenceNodesPropagateStandardEmulateVectorisedComputationOrder/simulateBiologicalHFnetworkSequenceNodePropagateStandardEmulateVectorisedComputationOrder)
					#somaActivationFound = True

			if(not emulateVectorisedComputationOrder):
				HFNLPpy_DendriticSANIDraw.drawDendriticSANIDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=wTarget)
	
	if(not emulateVectorisedComputationOrder):
		resetSourceNeuronAfterActivation(conceptNeuronSource)
			
	return somaActivationFound
						
#orig method;
def simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):

	if(printVerbose):
		print("simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup: wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
	
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
	
	#support for simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain:!biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect
	for wSource, conceptNeuronSource in enumerate(sentenceConceptNodeList):
	#orig: for wSource in range(0, wTarget):
		#conceptNeuronSource = sentenceConceptNodeList[wSource]	#source neuron
		activationTime = calculateActivationTimeSequence(wSource)
		if(simulateBiologicalHFnetworkSequenceNodeTrainPropagateSpecificTarget(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget)):
			somaActivationFound = True
			
	if(resetTargetNeuronDendriteAfterActivation):
		resetDendriticTreeActivation(conceptNeuronTarget)
	
	return somaActivationFound

#only calculateNeuronActivation for specific target
#parameters only used for drawBiologicalSimulationDynamic: wSource, wTarget
def simulateBiologicalHFnetworkSequenceNodeTrainPropagateSpecificTarget(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget):

	somaActivationFound = False
	
	#if(printVerbose):
	#print("simulateBiologicalHFnetworkSequenceNodeTrainPropagateSpecificTarget: wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)

	if(conceptNeuronTarget.nodeName in conceptNeuronSource.HFcontextTargetConnectionMultiDict):
		conceptNeuronSource.activationLevel = objectAreaActivationLevelOn
		connectionList = conceptNeuronSource.HFcontextTargetConnectionMultiDict[conceptNeuronTarget.nodeName]	#only trace connections between source neuron and target neuron
		for connection in connectionList:
			connection.activationLevel = objectAreaActivationLevelOn
			conceptNeuronConnectionTarget = connection.nodeTarget	#conceptNeuronConnectionTarget will be the same for all connection in connectionList (if targetConnectionConceptName == conceptNeuronTarget)
			if(conceptNeuronConnectionTarget != conceptNeuronTarget):
				print("simulateBiologicalHFnetworkSequenceNodeTrainPropagateSpecificTarget error: (conceptNeuronConnectionTarget != conceptNeuronTarget)")
				exit()

			if(calculateNeuronActivationStandardWrapper(connection, 0, conceptNeuronConnectionTarget.dendriticTree, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)):
				somaActivationFound = True
				conceptNeuronConnectionTarget.activationLevel = objectAreaActivationLevelOn
				#if(printVerbose):
				#print("somaActivationFound")

			HFNLPpy_DendriticSANIDraw.drawDendriticSANIDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wTarget=wTarget)	

		if(resetSourceNeuronAxonAfterActivation):
			resetAxonsActivationConnectionList(connectionList)
			conceptNeuronSource.activationLevel = objectAreaActivationLevelOff
			
	return somaActivationFound
					
										
def calculateNeuronActivationStandardWrapper(connection, currentBranchIndex1, currentBranch, activationTime, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):

	somaActivationFoundCurrent = False
	
	if(standardComputationOptimised):
		currentSequentialSegmentInput = connection.nodeTargetSequentialSegmentInput
		currentSequentialSegment = currentSequentialSegmentInput.sequentialSegment
		currentSequentialSegmentIndex = currentSequentialSegment.sequentialSegmentIndex
		currentBranch = currentSequentialSegment.branch
		currentBranchIndex1 = currentBranch.branchIndex1
	
		if(currentSequentialSegmentIndex < numberOfBranchSequentialSegments-1):
			print("currentSequentialSegmentIndex = ", currentSequentialSegmentIndex)
			print("numberOfBranchSequentialSegments = ", numberOfBranchSequentialSegments)
			previousSequentialSegment = currentBranch.sequentialSegments[currentSequentialSegmentIndex+1]
			sequentialSegmentActivationStatePrior = calculateSequentialSegmentActivationState(previousSequentialSegment.activationLevel)
			sequentialSegmentActivationLevelPrior = previousSequentialSegment.activationLevel
			sequentialSegmentActivationTimePrior = previousSequentialSegment.activationTime
			numberOfBranch2active = None
		else:
			recurse = False
			sequentialSegmentActivationStatePrior, sequentialSegmentActivationLevelPrior, sequentialSegmentActivationTimePrior, numberOfBranch2active = calculateSubbranchActivations(recurse, connection, currentBranchIndex1, currentBranch, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	

		sequentialSegmentActivationState, sequentialSegmentActivationLevel, sequentialSegmentActivationTime, sequentialSegmentActivationStateNew = calculateNeuronActivationSequentialSegment(connection, currentBranchIndex1, currentBranch, currentSequentialSegmentIndex, currentSequentialSegment, sequentialSegmentActivationStatePrior, sequentialSegmentActivationLevelPrior, sequentialSegmentActivationTimePrior, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)

		if(currentSequentialSegmentIndex == sequentialSegmentIndexMostProximal):
			sequentialSegmentActivationStateLast = sequentialSegmentActivationState
			sequentialSegmentActivationLevelLast = sequentialSegmentActivationLevel
			sequentialSegmentActivationTimeLast = sequentialSegmentActivationTime
			sequentialSegmentActivationStateLastNew = sequentialSegmentActivationStateNew

			branchActivationFound, branchActivationLevel, branchActivationTime = calculateBranchActivation(connection, currentBranchIndex1, currentBranch, sequentialSegmentActivationLevelLast, sequentialSegmentActivationLevelLast, sequentialSegmentActivationTimeLast, sequentialSegmentActivationStateLastNew, activationTime, numberOfBranch2active, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
		else:
			branchActivationFound = False
	else:
		branchActivationFound, branchActivationLevel, branchActivationTime = calculateNeuronActivationStandard(connection, currentBranchIndex1, currentBranch, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
	
	if(branchActivationFound):
		if((currentBranchIndex1 == branchIndex1MostProximal) and (currentSequentialSegmentIndex == sequentialSegmentIndexMostProximal)):
			somaActivationFoundCurrent = True
			
	return somaActivationFoundCurrent

#parameters only used for drawBiologicalSimulationDynamic: wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList
def calculateNeuronActivationStandard(connection, currentBranchIndex1, currentBranch, activationTime, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):	
	if(reversePropagationOrder):
		recurse = True
		sequentialSegmentActivationStatePreviousBranch, sequentialSegmentActivationLevelPreviousBranch, sequentialSegmentActivationTimePreviousBranch, numberOfBranch2active = calculateSubbranchActivations(recurse, connection, currentBranchIndex1, currentBranch, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
		branchActivationFound, branchActivationLevel, branchActivationTime = calculateNeuronActivationSequentialSegments(connection, currentBranchIndex1, currentBranch, sequentialSegmentActivationStatePreviousBranch, sequentialSegmentActivationLevelPreviousBranch, sequentialSegmentActivationTimePreviousBranch, activationTime, numberOfBranch2active, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	
	else:	
		recurse = False
		sequentialSegmentActivationStatePreviousBranch, sequentialSegmentActivationLevelPreviousBranch, sequentialSegmentActivationTimePreviousBranch, numberOfBranch2active = calculateSubbranchActivations(recurse, connection, currentBranchIndex1, currentBranch, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	
		branchActivationFound, branchActivationLevel, branchActivationTime = calculateNeuronActivationSequentialSegments(connection, currentBranchIndex1, currentBranch, sequentialSegmentActivationStatePreviousBranch, sequentialSegmentActivationLevelPreviousBranch, sequentialSegmentActivationTimePreviousBranch, activationTime, numberOfBranch2active, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
		recurse = True
		sequentialSegmentActivationStatePreviousBranch, sequentialSegmentActivationLevelPreviousBranch, sequentialSegmentActivationTimePreviousBranch, numberOfBranch2active = calculateSubbranchActivations(recurse, connection, currentBranchIndex1, currentBranch, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	
		
	return branchActivationFound, branchActivationLevel, branchActivationTime
	
def calculateSubbranchActivations(recurse, connection, currentBranchIndex1, currentBranch, activationTime, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):
	subbranchesActive = objectAreaActivationLevelOff
	subbranchesActivationTimeMax = minimumActivationTime
	if(performSummationOfSequentialSegmentInputsAcrossBranch):
		branch2activationSum = 0.0		
	else:
		numberOfBranch2active = 0
	if(len(currentBranch.subbranches) > 0):
		for subbranch in currentBranch.subbranches:	
			if(recurse):
				subbranchActivationFound, subbranchActiveLevel, subbranchActivationTime = calculateNeuronActivationStandard(connection, currentBranchIndex1+1, subbranch, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	
			else:
				subbranchActivationFound = calculateSequentialSegmentActivationState(subbranch.activationLevel) 
				subbranchActiveLevel = subbranch.activationLevel
				subbranchActivationTime = subbranch.activationTime
			subbranchActive = False
			if(performSummationOfSequentialSegmentInputsAcrossBranch):
				if(subbranchActiveLevel > objectLocalActivationLevelOff):
					subbranchActive = objectAreaActivationLevelOn
					branch2activationSum = branch2activationSum + subbranchActiveLevel
			else:
				if(subbranchActivationFound):
					subbranchActive = objectAreaActivationLevelOn
					numberOfBranch2active += 1
			if(subbranchActive):
				if(subbranchActivationTime > subbranchesActivationTimeMax):
					subbranchesActivationTimeMax = subbranchActivationTime
		if(performSummationOfSequentialSegmentInputsAcrossBranch):
			#print("Subbranches: currentBranchIndex1 = ", currentBranchIndex1, ", connection.nodeTarget = ", connection.nodeTarget.nodeName, ", connection.nodeSource = ", connection.nodeSource.nodeName, ", branch2activationSum = ", branch2activationSum, ", subbranchesActivationTimeMax = ", subbranchesActivationTimeMax, ", activationTime = ", activationTime)
			if(branch2activationSum >= numberOfHorizontalSubBranchesRequiredForActivation):
				subbranchesActive = objectAreaActivationLevelOn
		else:
			if(numberOfBranch2active >= numberOfHorizontalSubBranchesRequiredForActivation):	#must conform with branch merge method *
				subbranchesActive = objectAreaActivationLevelOn
	else:
		subbranchesActive = objectAreaActivationLevelOn
		if(requireSubbranchOrSequentialSegmentForActivation):
			numberOfBranch2active = 1
		#subbranchesActivationTimeMax = 0

	sequentialSegmentActivationStatePreviousBranch = subbranchesActive	#initialise prior sequential segment activation state to subbranchesActive
	sequentialSegmentActivationLevelPreviousBranch = objectLocalActivationLevelOff
	sequentialSegmentActivationTimePreviousBranch = subbranchesActivationTimeMax
				
	return sequentialSegmentActivationStatePreviousBranch, sequentialSegmentActivationLevelPreviousBranch, sequentialSegmentActivationTimePreviousBranch, numberOfBranch2active
	
def calculateNeuronActivationSequentialSegments(connection, currentBranchIndex1, currentBranch, sequentialSegmentActivationStatePreviousBranch, sequentialSegmentActivationLevelPreviousBranch, sequentialSegmentActivationTimePreviousBranch, activationTime, numberOfBranch2active, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):
		
	sequentialSegmentActivationStateLastNew = False
	
	sequentialSegmentSequence = list(enumerate(currentBranch.sequentialSegments))
	if(reversePropagationOrder):
		sequentialSegmentSequence = reversed(sequentialSegmentSequence)
		
	for currentSequentialSegmentIndex, currentSequentialSegment in sequentialSegmentSequence:
	
		if(currentSequentialSegmentIndex < numberOfBranchSequentialSegments-1):
			if(reversePropagationOrder):
				#sequentialSegmentActivationStatePrior etc already calculated
				pass
			else:
				previousSequentialSegment = currentBranch.sequentialSegments[currentSequentialSegmentIndex+1]
				sequentialSegmentActivationStatePrior = calculateSequentialSegmentActivationState(previousSequentialSegment.activationLevel)
				sequentialSegmentActivationLevelPrior = previousSequentialSegment.activationLevel
				sequentialSegmentActivationTimePrior = previousSequentialSegment.activationTime
		else:
			sequentialSegmentActivationStatePrior = sequentialSegmentActivationStatePreviousBranch
			sequentialSegmentActivationLevelPrior = sequentialSegmentActivationLevelPreviousBranch
			sequentialSegmentActivationTimePrior = 	sequentialSegmentActivationTimePreviousBranch	
		
		sequentialSegmentActivationState, sequentialSegmentActivationLevel, sequentialSegmentActivationTime, sequentialSegmentActivationStateNew = calculateNeuronActivationSequentialSegment(connection, currentBranchIndex1, currentBranch, currentSequentialSegmentIndex, currentSequentialSegment, sequentialSegmentActivationStatePrior, sequentialSegmentActivationLevelPrior, sequentialSegmentActivationTimePrior, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)

		if(reversePropagationOrder):
			sequentialSegmentActivationStatePrior = sequentialSegmentActivationState
			sequentialSegmentActivationLevelPrior = sequentialSegmentActivationLevel
			sequentialSegmentActivationTimePrior = sequentialSegmentActivationTime
			
		if(currentSequentialSegmentIndex == sequentialSegmentIndexMostProximal):
			if(sequentialSegmentActivationStateNew):
				sequentialSegmentActivationStateLastNew = True
			sequentialSegmentActivationStateLast = sequentialSegmentActivationState
			sequentialSegmentActivationLevelLast = sequentialSegmentActivationLevel
			sequentialSegmentActivationTimeLast = sequentialSegmentActivationTime
	
	branchActivationFound, branchActivationLevel, branchActivationTime = calculateBranchActivation(connection, currentBranchIndex1, currentBranch, sequentialSegmentActivationStateLast, sequentialSegmentActivationLevelLast, sequentialSegmentActivationTimeLast, sequentialSegmentActivationStateLastNew, activationTime, numberOfBranch2active, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
			
	return branchActivationFound, branchActivationLevel, branchActivationTime

					
def calculateNeuronActivationSequentialSegment(connection, currentBranchIndex1, currentBranch, currentSequentialSegmentIndex, currentSequentialSegment, sequentialSegmentActivationStatePrior, sequentialSegmentActivationLevelPrior, sequentialSegmentActivationTimePrior, activationTime, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):

	sequentialSegmentActivationStateNew = False

	#activationFound = False
	targetConceptNeuron = connection.nodeTarget
	wTarget = targetConceptNeuron.w	#debug
	
	if((currentBranchIndex1 > 0) or expectFirstBranchSequentialSegmentConnection):
		#calculate branch segment activations:

		sequentialSegmentActivationState = objectAreaActivationLevelOff
		sequentialSegmentActivationLevel = objectLocalActivationLevelOff	#current activation level
		sequentialSegmentActivationTime = None	#current activation time

		sequentialSegmentAlreadyActive = False
		if(sequentialSegmentActivationLevelAboveZero(currentSequentialSegment.activationLevel)):	#required to ensure currentSequentialSegment.activationTime is valid
			if(calculateSequentialSegmentActivationState(currentSequentialSegment.activationLevel)):
				sequentialSegmentAlreadyActive = True
				sequentialSegmentActivationState = objectAreaActivationLevelOn
			sequentialSegmentActivationLevel = currentSequentialSegment.activationLevel
			sequentialSegmentActivationTime = currentSequentialSegment.activationTime

		passSegmentActivationOverwriteTests = True

		if(overwriteSequentialSegments):
			if(currentSequentialSegment.frozen):
				passSegmentActivationOverwriteTests = False
				#print("currentSequentialSegment.frozen")
		if(sequentialSegmentAlreadyActive and not overwriteSequentialSegments):
			passSegmentActivationOverwriteTests = False

		if(passSegmentActivationOverwriteTests):
			foundConnectionSynapse, currentSequentialSegmentInput = findConnectionSynapseInSequentialSegment(currentSequentialSegment, connection)
			if(foundConnectionSynapse):
				inputActivationLevel = calculateInputActivationLevel(connection)
				if(recordSequentialSegmentInputActivationLevels):
					currentSequentialSegmentInput.activationLevel = inputActivationLevel
					currentSequentialSegmentInput.activationTime = activationTime
					if(drawBiologicalSimulationDynamicHighlightNewActivations):
						currentSequentialSegmentInput.activationStateNew = True
				if(printVerbose):
					printIndentation(currentBranchIndex1+1)
					print("activate currentSequentialSegmentInput, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)

				passSegmentActivationTimeTests = False
				if(currentSequentialSegmentInput.firstInputInSequence):
					if(verifyReactivationTime(currentSequentialSegment, activationTime)):	#ensure that the segment isnt in a repolarisation state (ie it can be activated)
						passSegmentActivationTimeTests = True	#if input corresponds to first in sequence, then enforce no previous dendritic activation requirements
				else:
					if(sequentialSegmentActivationStatePrior):	#previous sequential segment/subbranch was activated		#only accept sequential segment activation if previous was activated
						#print("currentBranchIndex1 = ", currentBranchIndex1)
						if(verifySequentialActivationTime(activationTime, sequentialSegmentActivationTimePrior)):	#ignore existing activation level if it occured at an earlier/same time than/as sequentialSegmentActivationTimePrior
							if(verifyReactivationTime(currentSequentialSegment, activationTime)):	#ensure that the segment isnt in a repolarisation state (ie it can be activated)
								passSegmentActivationTimeTests = True	#sequentialSegmentActivationLevel implies subbranchesActive: previous (ie more distal) branch was active

				if(passSegmentActivationTimeTests):
					if(performSummationOfSequentialSegmentInputs):
						sequentialSegmentActivationLevel = sequentialSegmentActivationLevel + inputActivationLevel
						sequentialSegmentActivationTime = activationTime	#CHECKTHIS: always record latest activation time for sequential segment activation
						sequentialSegmentActivationState = calculateSequentialSegmentActivationState(sequentialSegmentActivationLevel)
					else:
						#printIndentation(currentBranchIndex1+1)
						#print("passSegmentActivationTimeTests: activate currentSequentialSegmentInput, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName, ", inputActivationLevel = ", inputActivationLevel, ", activationTime = ", activationTime)
						sequentialSegmentActivationLevel = inputActivationLevel
						sequentialSegmentActivationTime = activationTime
						sequentialSegmentActivationState = objectAreaActivationLevelOn
					currentSequentialSegment.activationLevel = sequentialSegmentActivationLevel
					currentSequentialSegment.activationTime = sequentialSegmentActivationTime

					#if(resetConnectionTargetNeuronDendriteAfterSequence):
					if(sequentialSegmentActivationState):
						if(resetConnectionTargetNeuronDendriteDuringActivation):
							deactivatePreviousSequentialSegmentOrSubbranch(currentSequentialSegment)
						sequentialSegmentActivationStateNew = True
						if(drawBiologicalSimulationDynamicHighlightNewActivations):
							currentSequentialSegment.activationStateNew = True
						if(overwriteSequentialSegmentsAfterPropagatingSignal):
							if(not ((currentBranchIndex1 == branchIndex1MostProximal) and (currentSequentialSegmentIndex == sequentialSegmentIndexMostProximal))):	#never freeze most proximal sequential segment in tree
								#freeze all newly activated sequential segment states
								currentSequentialSegment.frozen = True
							for subbranch in currentBranch.subbranches:	
								previousSequentialSegment = subbranch.sequentialSegments[sequentialSegmentIndexMostProximal]
								previousSequentialSegment.frozen = False

						if(not emulateVectorisedComputationOrder):
							HFNLPpy_DendriticSANIDraw.drawBiologicalDendriticSANISequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, currentBranchIndex1, currentSequentialSegmentIndex, activationTime, wTarget=wTarget)

				else:
					if(deactivateSequentialSegmentsIfTimeTestsFail):
						#print("deactivateSequentialSegmentsIfTimeTestsFail")
						#sequentiality requirements (no longer) met - deactivate sequential segment
						sequentialSegmentActivationLevel = objectLocalActivationLevelOff
						sequentialSegmentActivationTime = currentSequentialSegment.activationTime	#no change in last activation time
						sequentialSegmentActivationState = objectAreaActivationLevelOff
						currentSequentialSegment.activationLevel = sequentialSegmentActivationLevel
						currentSequentialSegment.activationTime = sequentialSegmentActivationTime	

						if(not emulateVectorisedComputationOrder):
							HFNLPpy_DendriticSANIDraw.drawBiologicalDendriticSANISequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, currentBranchIndex1, currentSequentialSegmentIndex, activationTime, wTarget=wTarget)
	else:
		sequentialSegmentActivationState = sequentialSegmentActivationStatePrior
		sequentialSegmentActivationLevel = sequentialSegmentActivationLevelPrior
		sequentialSegmentActivationTime = sequentialSegmentActivationTimePrior
		currentSequentialSegment.activationLevel = sequentialSegmentActivationLevel
		currentSequentialSegment.activationTime = sequentialSegmentActivationTime

		if(resetConnectionTargetNeuronDendriteDuringActivation):
			if(sequentialSegmentActivationState):
				deactivatePreviousSequentialSegmentOrSubbranch(currentSequentialSegment)

	return sequentialSegmentActivationState, sequentialSegmentActivationLevel, sequentialSegmentActivationTime, sequentialSegmentActivationStateNew

						
def calculateBranchActivation(connection, currentBranchIndex1, currentBranch, sequentialSegmentActivationStateLast, sequentialSegmentActivationLevelLast, sequentialSegmentActivationTimeLast, sequentialSegmentActivationStateLastNew, activationTime, numberOfBranch2active, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):
	
	#print("branch: currentBranchIndex1 = ", currentBranchIndex1, ", connection.nodeTarget = ", connection.nodeTarget.nodeName, ", connection.nodeSource = ", connection.nodeSource.nodeName, ", sequentialSegmentActivationLevelLast = ", sequentialSegmentActivationLevelLast)

	if(requireSubbranchOrSequentialSegmentForActivation):
		if(not sequentialSegmentActivationStateLast):
			if(numberOfBranch2active == numberOfHorizontalSubBranchesOrSequentialSegmentsRequiredForActivation):
				sequentialSegmentActivationStateLast = True
				sequentialSegmentActivationLevelLast = objectLocalActivationLevelOn
				sequentialSegmentActivationTimeLast = subbranchesActivationTimeMax
				#resetConnectionTargetNeuronDendriteAfterSequence:sequentialSegmentActivationStateLastNew not supported (most proximal sequential segment in dendritic tree must be active)
		
	branchActivationState = sequentialSegmentActivationStateLast
	branchActivationLevel = sequentialSegmentActivationLevelLast
	branchActivationTime = sequentialSegmentActivationTimeLast
	if(storeBranchActivationState):
		currentBranch.activationLevel = branchActivationState
	else:
		currentBranch.activationLevel = branchActivationLevel
	currentBranch.activationTime = branchActivationTime

	branchActivationFound = branchActivationState
	if(resetConnectionTargetNeuronDendriteAfterSequence):
		if(currentBranchIndex1 == branchIndex1MostProximal):
			branchActivationFound = sequentialSegmentActivationStateLastNew
		
	if(branchActivationLevel):
		if(printVerbose):
			printIndentation(currentBranchIndex1+1)
			print("branchActivationLevel: activate currentBranch, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
			#print("activationFound")
			
	return branchActivationFound, branchActivationLevel, branchActivationTime


	
def findConnectionSynapseInSequentialSegment(currentSequentialSegment, connection):
	foundConnectionSynapse = False
	currentSequentialSegmentInput = None
	if(preventGenerationOfDuplicateConnections):
		foundSequentialSegmentInput, currentSequentialSegmentInput = findSequentialSegmentInputBySourceNode(currentSequentialSegment, connection.nodeSource)
		if(foundSequentialSegmentInput):
			if(connection.nodeTargetSequentialSegmentInput == currentSequentialSegmentInput):
				foundConnectionSynapse = True
	else:
		for currentSequentialSegmentInputTest in currentSequentialSegment.inputs.values():
			if(connection.nodeTargetSequentialSegmentInput == currentSequentialSegmentInputTest):
				foundConnectionSynapse = True
				currentSequentialSegmentInput = currentSequentialSegmentInputTest
	return foundConnectionSynapse, currentSequentialSegmentInput	
	

