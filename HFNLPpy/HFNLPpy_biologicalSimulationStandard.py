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

printVerbose = False



def simulateBiologicalHFnetworkSequenceNodeTrainStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, w, conceptNeuron):
	
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


	activationTime = calculateActivationTime(sentenceIndex)
	
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
	
	for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():
		conceptNeuronTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
		for connection in connectionList:
			if(calculateNeuronActivation(connection, 0, conceptNeuronTarget.dendriticTree, activationTime)):
				if(conceptNeuronTarget == conceptNeuron):
					somaActivationFound = True

	return somaActivationFound
	
	
#orig method;
def simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup(sentenceIndex, sentenceConceptNodeList, w, conceptNeuron):

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
	
	return somaActivationFound
					

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
				if(recordSequentialSegmentInputActivationLevels):
					currentSequentialSegmentInput.activationLevel = True
					currentSequentialSegmentInput.activationTime = activationTime
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
	
	
