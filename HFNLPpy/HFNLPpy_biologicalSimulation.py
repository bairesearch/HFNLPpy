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

- for every time step (word w):
	for every concept neuron:
		for every vertical branch/segment (from outer to inner):
			for every horizontal branch:
				for every sequential synapse in segment (1+):
					for every non-sequential synapse in segment (1+):
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
	calculate neuron firing exclusively from subsequence detections

"""


# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import random

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
import HFNLPpy_hopfieldOperations

probabilityOfSubsequenceThreshold = 0.01	#FUTURE: calibrate depending on number of branches/sequentialSegments etc



numberOfHorizontalSubBranchesRequiredForActivation = 2	#calibrate
activationRepolarisationTime = 1	#calibrate

resetSequentialSegments = True

def simulateBiologicalHFnetworkSequenceTrain(sentenceIndex, networkConceptNodeDict, sentenceConceptNodeList):

	activationTime = calculateActivationTime(sentenceIndex)
	
	for w, conceptNeuron in enumerate(sentenceConceptNodeList):
		print("simulateBiologicalHFnetworkSequenceTrain: w = ", w)
		
		somaActivationFound = False	#is conceptNeuron activated by its prior context?
		for w2 in range(0, w):
			previousConceptNeuron = sentenceConceptNodeList[w2]	#source neuron
			for targetConnectionConceptName, connectionList in previousConceptNeuron.targetConnectionDict.items():
				if(targetConnectionConceptName == conceptNeuron):
					for connection in connectionList:
						targetNeuron = connection.nodeTarget	#targetNeuron will be the same for all connection in connectionList (if targetConnectionConceptName == conceptNeuron)
						
						if(targetNeuron != conceptNeuron):
							print("simulateBiologicalHFnetworkSequenceTrain error: (targetNeuron != conceptNeuron)")
							exit()
							
						#FUTURE: perform parallel processing (add target concept synapse/sequentialSegment/branch to tensor)
						if(calculateNeuronActivation(connection, activationTime)):
							somaActivationFound = True
							print("somaActivationFound")
							
		if(not somaActivationFound):
			addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex)
			
def addPredictiveSequenceToNeuron(conceptNeuron, w, sentenceConceptNodeList, sentenceIndex):

	activationTime = calculateActivationTime(sentenceIndex)
	numberOfWordsInSequence = len(sentenceConceptNodeList)
	
	#no prediction found for previous sequence; generate prediction for conceptNeuron (encode subsequences in dendrite)
	print("addPredictiveSequenceToNeuron:")
	#rule: encode in order of branch
	#rule: if no soma activation found, train prior sequence of w in w concept neuron dendrite:
	#take random subset of subsequences
	currentBranchIndex1	= 0	#vertical (from distal to proximal)
	currentBranchIndex2 = 0	#horizontal
	currentSequentialSegmentIndex = 0	# (from distal to proximal)
	for wPrevious1 in range(0, w):
		#currentSequentialSegmentInputIndex = 0	#not used; currently encode infinite number of inputs (connected synapses) at segment	#number of inputs at sequential segment is dynamically increased on demand
		for wPrevious2 in range(wPrevious1+1, w):
			#for every pair of prior context concepts 
			#probability of forming subsequence (dependent on distance between wPrevious and wPrevious2)
			probabilityOfSubsequence = random.uniform(0, 1)
			probabilityOfSubsequenceNormalised = probabilityOfSubsequence
			distanceBetweenWPrevious1and2 = wPrevious2-wPrevious1
			probabilityOfSubsequenceNormalised = probabilityOfSubsequenceNormalised/distanceBetweenWPrevious1and2	#normalise probabilityOfSubsequence wrt distance between wPrevious1 and wPrevious2
			probabilityOfSubsequenceNormalised = probabilityOfSubsequenceNormalised/numberOfWordsInSequence	#normalise probabilityOfSubsequence wrt sequence length (average number of subsequences encoded should be relatively stable across sentence length; reduce from w^w -> ~wlog(w))
			if(probabilityOfSubsequenceNormalised > probabilityOfSubsequenceThreshold):
				print("\tprobabilityOfSubsequence = ", probabilityOfSubsequence)
				print("\tprobabilityOfSubsequenceNormalised = ", probabilityOfSubsequenceNormalised)
				previousConceptNeuron1 = sentenceConceptNodeList[wPrevious1]
				previousConceptNeuron2 = sentenceConceptNodeList[wPrevious2]

				spatioTemporalIndex = calculateSpatioTemporalIndex(sentenceIndex)

				newSequentialSegmentSegmentInputIndex = len(conceptNeuron.branchSequentialSegmentInputSize[currentBranchIndex1][currentBranchIndex2][currentSequentialSegmentIndex])
				#print("newSequentialSegmentSegmentInputIndex = ", newSequentialSegmentSegmentInputIndex)
				newSequentialSegmentSegmentInputIndex =+ 1						
				currentBranchIndex1, currentBranchIndex2, currentSequentialSegmentIndex = addPredictiveSynapseToNeuron(currentBranchIndex1, currentBranchIndex2, currentSequentialSegmentIndex, previousConceptNeuron1, conceptNeuron, activationTime, spatioTemporalIndex, biologicalImplementation=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetBranchIndex1=currentBranchIndex1, nodeTargetBranchIndex2=currentBranchIndex2, nodeTargetSequentialSegmentIndex=currentSequentialSegmentIndex, nodeTargetSequentialSegmentInputIndex=newSequentialSegmentSegmentInputIndex)
				
				newSequentialSegmentSegmentInputIndex = len(conceptNeuron.branchSequentialSegmentInputSize[currentBranchIndex1][currentBranchIndex2][currentSequentialSegmentIndex])
				#print("newSequentialSegmentSegmentInputIndex = ", newSequentialSegmentSegmentInputIndex)
				newSequentialSegmentSegmentInputIndex =+ 1		
				currentBranchIndex1, currentBranchIndex2, currentSequentialSegmentIndex = addPredictiveSynapseToNeuron(currentBranchIndex1, currentBranchIndex2, currentSequentialSegmentIndex, previousConceptNeuron2, conceptNeuron, activationTime, spatioTemporalIndex, biologicalImplementation=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=True, biologicalSynapse=True, nodeTargetBranchIndex1=currentBranchIndex1, nodeTargetBranchIndex2=currentBranchIndex2, nodeTargetSequentialSegmentIndex=currentSequentialSegmentIndex, nodeTargetSequentialSegmentInputIndex=newSequentialSegmentSegmentInputIndex)


#adds predictive synapse such that subsequences occur in order
def addPredictiveSynapseToNeuron(currentBranchIndex1, currentBranchIndex2, currentSequentialSegmentIndex, nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalImplementation=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, biologicalSimulation=False, biologicalSynapse=False, nodeTargetBranchIndex1=None, nodeTargetBranchIndex2=None, nodeTargetSequentialSegmentIndex=None, nodeTargetSequentialSegmentInputIndex=None):

	HFNLPpy_hopfieldOperations.addConnectionToNode(nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalImplementation=biologicalImplementation, weight=weight, subsequenceConnection=subsequenceConnection, contextConnection=contextConnection, contextConnectionSANIindex=contextConnectionSANIindex, biologicalSimulation=biologicalSimulation, biologicalSynapse=biologicalSynapse, nodeTargetBranchIndex1=nodeTargetBranchIndex1, nodeTargetBranchIndex2=nodeTargetBranchIndex2, nodeTargetSequentialSegmentIndex=nodeTargetSequentialSegmentIndex, nodeTargetSequentialSegmentInputIndex=nodeTargetSequentialSegmentInputIndex)

	newSequentialSegmentIndex = False
	if(currentSequentialSegmentIndex+1 == numberOfBranchSequentialSegments):
		currentSequentialSegmentIndex = 0	#reset currentBranchIndex
		newSequentialSegmentIndex = True
	else:
		currentSequentialSegmentIndex += 1

	if(newSequentialSegmentIndex):
		newBranchIndex2 = False
		if(currentBranchIndex2+1 == numberOfBranches2):
			currentBranchIndex2 = 0	#reset currentBranchIndex
			newBranchIndex2 = True
		else:
			currentBranchIndex2 += 1

	if(newBranchIndex2):
		newBranchIndex1 = False
		if(currentBranchIndex1+1 == numberOfBranches1):
			currentBranchIndex1 = 0	#reset currentBranchIndex
			newBranchIndex1 = True
		else:
			currentBranchIndex1 += 1

	return currentBranchIndex1, currentBranchIndex2, currentSequentialSegmentIndex
																				
def filledList(lst):
	result = False
	if(len(lst) > 0):
		result = True
	return result
		
def calculateNeuronActivation(connection, activationTime):
	
	somaActivationFound = False
	targetConceptNeuron = connection.nodeTarget

	#calculateBranchSegmentActivation():
	numberOfBranch1active = 0
	branch1ActivationLevelPrevious = False
	for branchIndex1 in range(numberOfBranches1):
	
		#branch1ActivationLevel = targetConceptNeuron.branch1ActivationLevel[branchIndex1]
		#branch1ActivationTime = targetConceptNeuron.branch1ActivationTime[branchIndex1]
		#branch1ActivationLevel = False
		numberOfBranch2active = 0
		#branch2ActivationTimeMax = 0
		
		for branchIndex2 in range(numberOfBranches2):
		
			branch2ActivationLevel = targetConceptNeuron.branch2ActivationLevel[branchIndex1][branchIndex2]
			branch2ActivationTime = targetConceptNeuron.branch2ActivationTime[branchIndex1][branchIndex2]
			branch2ActivationLevel = False
			numberOfSequentialSegmentsActive = 0	#not currently used
			branch2ActivationLevel = False
			
			for currentSequentialSegmentIndex in range(numberOfBranchSequentialSegments):
			
				sequentialSegmentActivationLevel = targetConceptNeuron.branchSequentialSegmentActivationLevel[branchIndex1][branchIndex2][currentSequentialSegmentIndex]
				sequentialSegmentActivationTime = targetConceptNeuron.branchSequentialSegmentActivationTime[branchIndex1][branchIndex2][currentSequentialSegmentIndex]
				if(currentSequentialSegmentIndex == 0):
					sequentialSegmentActivationLevel = True	#no sequential requirement @index0
				sequentialSegmentActivationLevelNew = False
				
				if(connection.nodeTargetBranchIndex1 == branchIndex1):
					if(connection.nodeTargetBranchIndex2 == branchIndex2):
						if(connection.nodeTargetSequentialSegmentIndex == currentSequentialSegmentIndex):
							#fire nodeTargetSequentialSegmentInputIndex
							if(sequentialSegmentActivationLevel):	#previous sequential segment was activated
								passSegmentActivationTimeTests = False
								if((currentSequentialSegmentIndex == 0) or (activationTime > sequentialSegmentActivationTime+activationRepolarisationTime)):	#ensure that the segment isnt in a repolarisation state (ie it can be activated)
									#if(activationTime > previousVerticalBranchActivationTime):	#guaranteed
									if(branchIndex1 > 0):
										#branch2ActivationLevelPreviousMax = 0
										numberOfBranch2activePrevious = 0
										for branchIndex2b in range(numberOfBranches2):
											branch2ActivationLevelPrevious = targetConceptNeuron.branch2ActivationLevel[branchIndex1-1][branchIndex2b]
											numberOfBranch2activePrevious = numberOfBranch2activePrevious + int(branch2ActivationLevelPrevious)
											#if(branch2ActivationLevelPrevious > branch2ActivationLevelPreviousMax):
											#	branch2ActivationLevelPreviousMax = branch2ActivationLevelPrevious
										if(numberOfBranch2activePrevious > numberOfHorizontalSubBranchesRequiredForActivation):
											passSegmentActivationTimeTests = True	#previous (ie more distal) branch1 was active
									else:
										passSegmentActivationTimeTests = True
								if(passSegmentActivationTimeTests):
									sequentialSegmentActivationLevelNew = True
									sequentialSegmentActivationTimeNew = activationTime
				
				if(sequentialSegmentActivationLevelNew):
					if(resetSequentialSegments):
						if(currentSequentialSegmentIndex == 0):
							for currentSequentialSegmentIndex2 in range(1, numberOfBranchSequentialSegments):	#skip currentSequentialSegmentIndex==0
								targetConceptNeuron.branchSequentialSegmentActivationLevel[branchIndex1][branchIndex2][currentSequentialSegmentIndex2] = False
								targetConceptNeuron.branch2ActivationLevel[branchIndex1][branchIndex2] = False	#deactivate horizontal branch
							numberOfSequentialSegmentsActive = 0
					numberOfSequentialSegmentsActive += 1	#CHECKTHIS
					sequentialSegmentActivationLevel = True
					sequentialSegmentActivationTime = activationTime
					targetConceptNeuron.branchSequentialSegmentActivationLevel[branchIndex1][branchIndex2][currentSequentialSegmentIndex] = sequentialSegmentActivationLevel
					targetConceptNeuron.branchSequentialSegmentActivationTime[branchIndex1][branchIndex2][currentSequentialSegmentIndex] = sequentialSegmentActivationTime
				
			sequentialSegmentActivationLevelLast = sequentialSegmentActivationLevel
			sequentialSegmentActivationTimeLast = sequentialSegmentActivationTime
			#sequentialSegmentActivationLevelLastNew = sequentialSegmentActivationLevelLast
			if(sequentialSegmentActivationLevelLast):
				numberOfBranch2active += 1
				branch2ActivationLevel = sequentialSegmentActivationLevelLast	#activate branch2	#activate whole sequentialSegment
				branch2ActivationTime = sequentialSegmentActivationTimeLast
				targetConceptNeuron.branch2ActivationLevel[branchIndex1][branchIndex2] = branch2ActivationLevel
				targetConceptNeuron.branch2ActivationTime[branchIndex1][branchIndex2] = branch2ActivationTime	

	numberOfBranch2activeLast = numberOfBranch2active	#most proximal subbranch
	if(numberOfBranch2active > numberOfHorizontalSubBranchesRequiredForActivation):
		somaActivationFound = True	#fire neuron
	
	return somaActivationFound							
