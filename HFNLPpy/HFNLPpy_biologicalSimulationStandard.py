"""HFNLPpy_biologicalSimulationStandard.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Standard

"""


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_biologicalSimulationNode import *


printVerbose = False



def simulateBiologicalHFnetworkSequenceNodeTrainStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, w, conceptNeuron, connectionTargetNeuronSet):
	
	activationTime = calculateActivationTimeSequence(wSource)
	
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
	conceptNeuronSource.activationLevel = True
	
	for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():
		conceptNeuronTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
		connectionTargetNeuronSet.add(conceptNeuronTarget)
		for connection in connectionList:
			if(calculateNeuronActivationStandard(connection, 0, conceptNeuronTarget.dendriticTree, activationTime)[0]):
				if(conceptNeuronTarget == conceptNeuron):
					somaActivationFound = True
				conceptNeuronTarget.activationLevel = True

	resetAxonsActivation(conceptNeuronSource)
	if(resetWsourceNeuronDendriteAfterActivation):
		resetDendriticTreeActivation(conceptNeuronSource)
			
	return somaActivationFound
	
	
#orig method;
def simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup(sentenceIndex, sentenceConceptNodeList, w, conceptNeuron):

	#if(printVerbose):
	print("simulateBiologicalHFnetworkSequenceNodeTrain: w = ", w, ", conceptNeuron = ", conceptNeuron.nodeName)
	
	somaActivationFound = False	#is conceptNeuron activated by its prior context?
	for wSource in range(0, w):
		activationTime = calculateActivationTimeSequence(wSource)
		conceptNeuronSource = sentenceConceptNodeList[wSource]	#source neuron
		conceptNeuronSource.activationLevel = True
		if(conceptNeuron.nodeName in conceptNeuronSource.targetConnectionDict):
			connectionList = conceptNeuronSource.targetConnectionDict[conceptNeuron.nodeName]	#only trace connections between source neuron and target neuron
			for connection in connectionList:
				targetNeuron = connection.nodeTarget	#targetNeuron will be the same for all connection in connectionList (if targetConnectionConceptName == conceptNeuron)
				if(targetNeuron != conceptNeuron):
					print("simulateBiologicalHFnetworkSequenceNodeTrain error: (targetNeuron != conceptNeuron)")
					exit()

				#FUTURE: vectoriseComputationCurrentDendriticInput: perform parallel processing (add target concept synapse/sequentialSegment/branch to tensor)
				#print("calculateNeuronActivationStandard")
				if(calculateNeuronActivationStandard(connection, 0, targetNeuron.dendriticTree, activationTime)[0]):
					somaActivationFound = True
					targetNeuron.activationLevel = True
					#if(printVerbose):
					#print("somaActivationFound")
			
			resetAxonsActivation(conceptNeuronSource)

	resetDendriticTreeActivation(conceptNeuron)
	
	return somaActivationFound
					

def calculateNeuronActivationStandard(connection, currentBranchIndex1, currentBranch, activationTime):
	
	connection.activationLevel = True
	
	#activationFound = False
	targetConceptNeuron = connection.nodeTarget
		
	#calculate subbranch activations:
	subbranchesActive = False
	subbranchesActivationTimeMax = 0	 #alternatively set -1; initial activation time of dendritic sequence set artificially low such that passSegmentActivationTimeTests automatically pass (not required (as passSegmentActivationTimeTests are ignored for currentSequentialSegmentInput.firstInputInSequence)
	if(len(currentBranch.subbranches) > 0):
		numberOfBranch2active = 0
		for subbranch in currentBranch.subbranches:	
			subbranchActive, subbranchActivationTime = calculateNeuronActivationStandard(connection, currentBranchIndex1+1, subbranch, activationTime)
			if(subbranchActive):
				numberOfBranch2active += 1
				if(subbranchActivationTime > subbranchesActivationTimeMax):
					subbranchesActivationTimeMax = subbranchActivationTime
		if(numberOfBranch2active >= numberOfHorizontalSubBranchesRequiredForActivation):	#must conform with branch merge method *
			subbranchesActive = True
	else:
	 	subbranchesActive = True
	
	sequentialSegmentActivationLevel = subbranchesActive	#initialise sequential segment activation to subbranchesActive
	sequentialSegmentActivationTime = subbranchesActivationTimeMax
	
	#calculate branch segment activations:
	for currentSequentialSegmentIndex, currentSequentialSegment in reversed(list(enumerate(currentBranch.sequentialSegments))):
		
		sequentialSegmentAlreadyActive = False
		if(sequentialSegmentActivationLevel):	#only accept sequential segment activation if previous was activated
			if(currentSequentialSegment.activationLevel):
				if(verifySequentialActivationTime(currentSequentialSegment.activationTime, sequentialSegmentActivationTime)):	#this test should not be required with preventReactivationOfSequentialSegments
					sequentialSegmentAlreadyActive = True
					sequentialSegmentActivationLevel = currentSequentialSegment.activationLevel
					sequentialSegmentActivationTime = currentSequentialSegment.activationTime
		
		if(not preventReactivationOfSequentialSegments or not sequentialSegmentAlreadyActive):
			sequentialSegmentActivationLevelNew = False
			for currentSequentialSegmentInputIndex, currentSequentialSegmentInput in enumerate(currentSequentialSegment.inputs):
				if(connection.nodeTargetSequentialSegmentInput == currentSequentialSegmentInput):
					if(recordSequentialSegmentInputActivationLevels):
						currentSequentialSegmentInput.activationLevel = True
						currentSequentialSegmentInput.activationTime = activationTime
					if(printVerbose):
						printIndentation(currentBranchIndex1+1)
						print("activate currentSequentialSegmentInput, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
					passSegmentActivationTimeTests = False
					if(currentSequentialSegmentInput.firstInputInSequence):
						passSegmentActivationTimeTests = True	#if input corresponds to first in sequence, then enforce no previous dendritic activation requirements
					else:
						if(sequentialSegmentActivationLevel):	#previous sequential segment was activated
							if(verifySequentialActivationTime(activationTime, sequentialSegmentActivationTime)):
								if(verifyRepolarised(currentSequentialSegment, activationTime)):	#ensure that the segment isnt in a repolarisation state (ie it can be activated)
									#if(activationTime > previousVerticalBranchActivationTime):	#guaranteed
									passSegmentActivationTimeTests = True	#sequentialSegmentActivationLevel implies subbranchesActive: previous (ie more distal) branch was active
					if(passSegmentActivationTimeTests):
						sequentialSegmentActivationLevelNew = True
						sequentialSegmentActivationTimeNew = activationTime

			if(sequentialSegmentActivationLevelNew):
				if(printVerbose):
					printIndentation(currentBranchIndex1+1)
					print("activate currentSequentialSegment, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
				#if(resetSequentialSegments):
				#	if(currentSequentialSegmentIndex == 0):
				#		resetBranchActivation(currentBranch)
				#		numberOfSequentialSegmentsActive = 0
				sequentialSegmentActivationLevel = True
				sequentialSegmentActivationTime = activationTime
				currentSequentialSegment.activationLevel = sequentialSegmentActivationLevel
				currentSequentialSegment.activationTime = sequentialSegmentActivationTime
			#if(sequentialSegmentActivationLevel):
			#	numberOfSequentialSegmentsActive += 1	#CHECKTHIS
			
	sequentialSegmentActivationLevelLast = sequentialSegmentActivationLevel
	sequentialSegmentActivationTimeLast = sequentialSegmentActivationTime
	#sequentialSegmentActivationLevelLastNew = sequentialSegmentActivationLevelLast
	#sequentialSegmentsActive = False
	
	#overwrite activation level of branch if last sequential segment inactive	#OLD: if(sequentialSegmentActivationLevelLast):	
	if(printVerbose):
		printIndentation(currentBranchIndex1+1)
		print("activate currentBranch, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
	branchActivationLevel = sequentialSegmentActivationLevelLast	#activate branch2	#activate whole currentSequentialSegment
	branchActivationTime = sequentialSegmentActivationTimeLast
	currentBranch.activationLevel = branchActivationLevel
	currentBranch.activationTime = branchActivationTime
	#sequentialSegmentsActive = True	

	if(branchActivationLevel):
		if(printVerbose):
			printIndentation(currentBranchIndex1+1)
			print("activationFound")
		#activationFound = True
			
	return branchActivationLevel, branchActivationTime							
	
	
