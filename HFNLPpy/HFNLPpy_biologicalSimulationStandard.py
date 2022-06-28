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

#if(biologicalSimulationForward):	#required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic?
drawBiologicalSimulationDynamic = False	#draw dynamic activation levels of biological simulation
if(drawBiologicalSimulationDynamic):
	drawBiologicalSimulationDynamicPlot = True	#default: False
	drawBiologicalSimulationDynamicSave = False	#default: True	#save to file
	drawBiologicalSimulationDendriticTreeSentenceDynamic = True	#default: True	#draw graph for sentence neurons and their dendritic tree
	if(drawBiologicalSimulationDendriticTreeSentenceDynamic):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentenceDynamic
	drawBiologicalSimulationDendriticTreeNetworkDynamic = False	#default: True	#draw graph for entire network (not just sentence)
	if(drawBiologicalSimulationDendriticTreeNetworkDynamic):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetworkDynamic


def simulateBiologicalHFnetworkSequenceNodeTrainStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wSource, conceptNeuronSource, w, conceptNeuron, connectionTargetNeuronSet):

	#if(printVerbose):
	print("simulateBiologicalHFnetworkSequenceNodeTrainStandard: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName)
		
	activationTime = calculateActivationTimeSequence(wSource)
	
	somaActivationFound = False	#is conceptNeuronTarget activated by its prior context?
	conceptNeuronSource.activationLevel = objectAreaActivationLevelOn
	
	for targetConnectionConceptName, connectionList in conceptNeuronSource.targetConnectionDict.items():
		conceptNeuronTarget = networkConceptNodeDict[targetConnectionConceptName] #or connectionList[ANY].nodeTarget
		connectionTargetNeuronSet.add(conceptNeuronTarget)
		for connection in connectionList:
			if(calculateNeuronActivationStandard(connection, 0, conceptNeuronTarget.dendriticTree, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)[0]):
				if(conceptNeuronTarget == conceptNeuron):
					somaActivationFound = True
				conceptNeuronTarget.activationLevel = objectAreaActivationLevelOn

			drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	
	
	resetAxonsActivation(conceptNeuronSource)
	if(resetWsourceNeuronDendriteAfterActivation):
		resetDendriticTreeActivation(conceptNeuronSource)
			
	return somaActivationFound
	
	
#orig method;
def simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, w, conceptNeuron):

	#if(printVerbose):
	print("simulateBiologicalHFnetworkSequenceNodeTrainStandardReverseLookup: w = ", w, ", conceptNeuron = ", conceptNeuron.nodeName)
	
	somaActivationFound = False	#is conceptNeuron activated by its prior context?
	for wSource in range(0, w):
		activationTime = calculateActivationTimeSequence(wSource)
		conceptNeuronSource = sentenceConceptNodeList[wSource]	#source neuron
		conceptNeuronSource.activationLevel = objectAreaActivationLevelOn
		if(conceptNeuron.nodeName in conceptNeuronSource.targetConnectionDict):
			connectionList = conceptNeuronSource.targetConnectionDict[conceptNeuron.nodeName]	#only trace connections between source neuron and target neuron
			for connection in connectionList:
				targetNeuron = connection.nodeTarget	#targetNeuron will be the same for all connection in connectionList (if targetConnectionConceptName == conceptNeuron)
				if(targetNeuron != conceptNeuron):
					print("simulateBiologicalHFnetworkSequenceNodeTrain error: (targetNeuron != conceptNeuron)")
					exit()

				#FUTURE: vectoriseComputationCurrentDendriticInput: perform parallel processing (add target concept synapse/sequentialSegment/branch to tensor)
				#print("calculateNeuronActivationStandard")
				if(calculateNeuronActivationStandard(connection, 0, targetNeuron.dendriticTree, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)[0]):
					somaActivationFound = True
					targetNeuron.activationLevel = objectAreaActivationLevelOn
					#if(printVerbose):
					#print("somaActivationFound")
			
				drawBiologicalSimulationDynamicNeuronActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	

			resetAxonsActivation(conceptNeuronSource)

	resetDendriticTreeActivation(conceptNeuron)
	
	return somaActivationFound
					

#parameters only used for drawBiologicalSimulationDynamic: wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList
def calculateNeuronActivationStandard(connection, currentBranchIndex1, currentBranch, activationTime, wSource=None, networkConceptNodeDict=None, sentenceIndex=None, sentenceConceptNodeList=None):
	
	connection.activationLevel = objectAreaActivationLevelOn
	
	#activationFound = False
	targetConceptNeuron = connection.nodeTarget
		
	#calculate subbranch activations:
	subbranchesActive = objectAreaActivationLevelOff
	subbranchesActivationTimeMax = 0	 #alternatively set -1; initial activation time of dendritic sequence set artificially low such that passSegmentActivationTimeTests automatically pass (not required (as passSegmentActivationTimeTests are ignored for currentSequentialSegmentInput.firstInputInSequence)
	if(len(currentBranch.subbranches) > 0):
		numberOfBranch2active = 0
		for subbranch in currentBranch.subbranches:	
			subbranchActive, subbranchActivationTime = calculateNeuronActivationStandard(connection, currentBranchIndex1+1, subbranch, activationTime, wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)
			if(subbranchActive):
				numberOfBranch2active += 1
				if(subbranchActivationTime > subbranchesActivationTimeMax):
					subbranchesActivationTimeMax = subbranchActivationTime
		#print("numberOfBranch2active = ", numberOfBranch2active)
		if(numberOfBranch2active >= numberOfHorizontalSubBranchesRequiredForActivation):	#must conform with branch merge method *
			subbranchesActive = objectAreaActivationLevelOn
			#print("subbranchesActivationTimeMax = ", subbranchesActivationTimeMax)
	else:
	 	subbranchesActive = objectAreaActivationLevelOn
	
	sequentialSegmentActivationStatePrior = subbranchesActive	#initialise prior sequential segment activation state to subbranchesActive
	sequentialSegmentActivationTimePrior = subbranchesActivationTimeMax
	
	#calculate branch segment activations:
	for currentSequentialSegmentIndex, currentSequentialSegment in reversed(list(enumerate(currentBranch.sequentialSegments))):

		sequentialSegmentActivationState = objectAreaActivationLevelOff
		sequentialSegmentActivationLevel = objectLocalActivationLevelOff	#current activation level
		sequentialSegmentActivationTime = None	#current activation time
			
		sequentialSegmentAlreadyActive = False
		if(sequentialSegmentActivationLevelAboveZero(currentSequentialSegment.activationLevel)):	#required to ensure currentSequentialSegment.activationTime is valid
			#if(not(sequentialSegmentActivationStatePrior) or verifySequentialActivationTime(currentSequentialSegment.activationTime, sequentialSegmentActivationTimePrior)):	#ignore existing activation level if it occured at an earlier/same time than/as sequentialSegmentActivationTimePrior	#this test should not be required with preventReactivationOfSequentialSegments
			if(calculateSequentialSegmentActivationState(currentSequentialSegment.activationLevel)):
				sequentialSegmentAlreadyActive = True
				sequentialSegmentActivationState = objectAreaActivationLevelOn
			sequentialSegmentActivationLevel = currentSequentialSegment.activationLevel
			sequentialSegmentActivationTime = currentSequentialSegment.activationTime
					
		if(not preventReactivationOfSequentialSegments or not sequentialSegmentAlreadyActive):
			for currentSequentialSegmentInputIndex, currentSequentialSegmentInput in enumerate(currentSequentialSegment.inputs):
				if(connection.nodeTargetSequentialSegmentInput == currentSequentialSegmentInput):
					inputActivationLevel = calculateInputActivationLevel(connection)
					if(recordSequentialSegmentInputActivationLevels):
						currentSequentialSegmentInput.activationLevel = inputActivationLevel
						currentSequentialSegmentInput.activationTime = activationTime
					if(printVerbose):
						printIndentation(currentBranchIndex1+1)
						print("activate currentSequentialSegmentInput, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)

					passSegmentActivationTimeTests = False
					if(currentSequentialSegmentInput.firstInputInSequence):
						#print("passSegmentActivationTimeTests")
						passSegmentActivationTimeTests = True	#if input corresponds to first in sequence, then enforce no previous dendritic activation requirements	#CHECKTHIS - check implementation compatibility with performSummationOfSequentialSegmentInputs (HFNLPpy_biologicalSimulationVectorised currently uses a different requirement that is also dependent on input activation levels)
					else:
						if(sequentialSegmentActivationStatePrior):	#previous sequential segment/subbranch was activated		#only accept sequential segment activation if previous was activated
							if(verifySequentialActivationTime(activationTime, sequentialSegmentActivationTimePrior)):	#ignore existing activation level if it occured at an earlier/same time than/as sequentialSegmentActivationTimePrior
								#if(verifyRepolarised(currentSequentialSegment, activationTime)):	#ensure that the segment isnt in a repolarisation state (ie it can be activated)
								passSegmentActivationTimeTests = True	#sequentialSegmentActivationLevel implies subbranchesActive: previous (ie more distal) branch was active
					
					if(passSegmentActivationTimeTests):
						if(performSummationOfSequentialSegmentInputs):
							sequentialSegmentActivationLevel = sequentialSegmentActivationLevel + inputActivationLevel
							sequentialSegmentActivationTime = activationTime	#CHECKTHIS: always record latest activation time for sequential segment activation
							sequentialSegmentActivationState = calculateSequentialSegmentActivationState(sequentialSegmentActivationLevel)
						else:
							sequentialSegmentActivationLevel = inputActivationLevel
							sequentialSegmentActivationTime = activationTime
							sequentialSegmentActivationState = objectAreaActivationLevelOn
						currentSequentialSegment.activationLevel = sequentialSegmentActivationLevel
						currentSequentialSegment.activationTime = sequentialSegmentActivationTime
						#print("currentSequentialSegment.activationLevel = ", currentSequentialSegment.activationLevel)
						
						drawBiologicalSimulationDynamicSequentialSegmentActivation(wSource, networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, currentBranchIndex1, currentSequentialSegmentIndex)
		
		sequentialSegmentActivationStatePrior = sequentialSegmentActivationState
		sequentialSegmentActivationTimePrior = sequentialSegmentActivationTime
		#print("sequentialSegmentActivationStatePrior = ", sequentialSegmentActivationStatePrior)

			
	#overwrite activation level of branch if last sequential segment inactive	#OLD: if(sequentialSegmentActivationLevelLast):	
	branchActivationLevel = sequentialSegmentActivationStatePrior	#activate branch2	#activate whole currentSequentialSegment
	branchActivationTime = sequentialSegmentActivationTimePrior
	currentBranch.activationLevel = branchActivationLevel
	currentBranch.activationTime = branchActivationTime
	#sequentialSegmentsActive = True

	if(branchActivationLevel):
		if(printVerbose):
			printIndentation(currentBranchIndex1+1)
			print("branchActivationLevel: activate currentBranch, connection.nodeSource = ", connection.nodeSource.nodeName, ", connection.nodeTarget = ", connection.nodeTarget.nodeName)
			#print("activationFound")
	return branchActivationLevel, branchActivationTime							
	
	
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
			
