"""HFNLPpy_DendriticSANI.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI - simulate training/inference of biological hopfield graph/network based on textual input

pseudo code;
for every time step/concept neuron (word w):
	for every branch in dendriticTree (2+, recursively parsed from outer to inner; sequentially dependent):
		for every sequential segment in branch (1+, sequentially dependent):
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


import numpy as np

from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from HFNLPpy_DendriticSANIGlobalDefs import *
from HFNLPpy_DendriticSANINode import *
import HFNLPpy_DendriticSANIGenerate
if(vectoriseComputation):
	import HFNLPpy_DendriticSANIPropagateVectorised
else:
	import HFNLPpy_DendriticSANIPropagateStandard
import HFNLPpy_DendriticSANIDraw
import HFNLPpy_hopfieldOperations

printVerbose = False


#alwaysAddPredictionInputFromPreviousConcept = False
#if(vectoriseComputation):
#	alwaysAddPredictionInputFromPreviousConcept = True #ensures that simulateBiologicalHFnetworkSequenceNodePropagateParallel:conceptNeuronBatchIndexFound



def seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, seedSentenceConceptNodeList, numberOfSentences):
	
	targetSentenceConceptNodeList = seedSentenceConceptNodeList
	
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	if(not seedHFnetworkSubsequenceBasic):
		conceptNeuronSourceList = []

	for wSource in range(len(targetSentenceConceptNodeList)-1):
		wTarget = wSource+1
		conceptNeuronSource = targetSentenceConceptNodeList[wSource]
		conceptNeuronTarget = targetSentenceConceptNodeList[wTarget]
		print("seedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)
		
		if(seedHFnetworkSubsequenceBasic):
			activationTime = calculateActivationTimeSequence(wSource)
			somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)
		else:
			connectionTargetNeuronSetLocal = set()
			activationTime = calculateActivationTimeSequence(wSource)
			if(wSource < seedHFnetworkSubsequenceLength):
				somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSetLocal)
			else:
				somaActivationFound = simulateBiologicalHFnetworkSequenceNodesPropagateForward(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSourceList, connectionTargetNeuronSetLocal)
			
			connectionTargetNeuronSetLocalFiltered = selectActivatedNeurons(networkConceptNodeDict, connectionTargetNeuronSetLocal)
				
			conceptNeuronSourceList.clear()
			for connectionTargetNeuron in connectionTargetNeuronSetLocalFiltered:
				if(connectionTargetNeuron.activationLevel):
					#print("conceptNeuronSourceList.append connectionTargetNeuron = ", connectionTargetNeuron.nodeName)
					conceptNeuronSourceList.append(connectionTargetNeuron)
			connectionTargetNeuronSet = connectionTargetNeuronSet.union(connectionTargetNeuronSetLocal)
			resetConnectionTargetNeurons(connectionTargetNeuronSetLocal, True, conceptNeuronTarget)	

		expectPredictiveSequenceToBeFound = False
		if(enforceMinimumEncodedSequenceLength):
			if(wSource >= minimumEncodedSequenceLength-1):
				expectPredictiveSequenceToBeFound = True
		else:
			expectPredictiveSequenceToBeFound = True
		if(expectPredictiveSequenceToBeFound):
			if(somaActivationFound):
				#if(printVerbose):
				print("somaActivationFound")
			else:
				#if(printVerbose):
				print("!somaActivationFound: HFNLP algorithm error detected")
		else:
			print("!expectPredictiveSequenceToBeFound: wSource < minimumEncodedSequenceLength-1")
			
	resetConnectionTargetNeurons(connectionTargetNeuronSet, False)

	HFNLPpy_DendriticSANIDraw.drawDendriticSANIStatic(networkConceptNodeDict, sentenceIndex, targetSentenceConceptNodeList, numberOfSentences)

def trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences):
	simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences)	


#if (!useAlgorithmDendriticSANIbiologicalSimulation:useDependencyParseTree):

def simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences):

	#cannot clear now as HFNLPpy_DendriticSANIDrawSentence/HFNLPpy_DendriticSANIDrawNetwork memory structure is not independent (diagnose reason for this);
	
	sentenceLength = len(sentenceConceptNodeList)
	
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	
	for wTarget in range(1, sentenceLength):	#wTarget>=1: do not create (recursive) connection from conceptNode to conceptNode branchIndex1=0
		conceptNeuronTarget = sentenceConceptNodeList[wTarget]
		
		connectionTargetNeuronSetLocal = set()
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateWrapper(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, connectionTargetNeuronSetLocal)
		
		connectionTargetNeuronSet = connectionTargetNeuronSet.union(connectionTargetNeuronSetLocal)
		resetConnectionTargetNeurons(connectionTargetNeuronSetLocal, True, conceptNeuronTarget)	
						
		if(somaActivationFound):
			#if(printVerbose):
			print("somaActivationFound")
		else:
			#if(printVerbose):
			print("!somaActivationFound: ", end='')
			predictiveSequenceLength = wTarget	#wSource+1
			dendriticBranchMaxW = wTarget-1
			expectFurtherSubbranches = True
			if(wTarget == 1):
				expectFurtherSubbranches = False
			
			addPredictiveSequenceToNeuron = False
			if(enforceMinimumEncodedSequenceLength):
				if(dendriticBranchMaxW+1 >= minimumEncodedSequenceLength):
					addPredictiveSequenceToNeuron = True
			else:
				addPredictiveSequenceToNeuron = True
			if(addPredictiveSequenceToNeuron):
				print("addPredictiveSequenceToNeuron")
				HFNLPpy_DendriticSANIGenerate.addPredictiveSequenceToNeuron(conceptNeuronTarget, sentenceIndex, sentenceConceptNodeList, conceptNeuronTarget.dendriticTree, predictiveSequenceLength, dendriticBranchMaxW, 0, 0, expectFurtherSubbranches)
			else:
				print("")	#add new line
				
	#reset dendritic trees
	resetConnectionTargetNeurons(connectionTargetNeuronSet, False)

	HFNLPpy_DendriticSANIDraw.drawDendriticSANIStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, numberOfSentences)

			
def simulateBiologicalHFnetworkSequenceNodePropagateWrapper(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, connectionTargetNeuronSet):
	somaActivationFound = False
	if(biologicalSimulationForward):
		wSource = wTarget-1
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		conceptNeuronTarget = sentenceConceptNodeList[wTarget]
		activationTime = calculateActivationTimeSequence(wSource)
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)
	else:
		conceptNeuronTarget = sentenceConceptNodeList[wTarget]
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)
	
	return somaActivationFound


def simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet):
	print("simulateBiologicalHFnetworkSequenceNodePropagateForward: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName)	
	if(vectoriseComputationCurrentDendriticInput):
		somaActivationFound = HFNLPpy_DendriticSANIPropagateVectorised.simulateBiologicalHFnetworkSequenceNodePropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)
	else:
		if(emulateVectorisedComputationOrder):
			somaActivationFound = HFNLPpy_DendriticSANIPropagateStandard.simulateBiologicalHFnetworkSequenceNodePropagateStandardEmulateVectorisedComputationOrder(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)					
		else:
			somaActivationFound = HFNLPpy_DendriticSANIPropagateStandard.simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)			
	return somaActivationFound

def simulateBiologicalHFnetworkSequenceNodesPropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSourceList, connectionTargetNeuronSet):
	if(vectoriseComputationCurrentDendriticInput):
		somaActivationFound = HFNLPpy_DendriticSANIPropagateVectorised.simulateBiologicalHFnetworkSequenceNodesPropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)
	else:
		if(emulateVectorisedComputationOrder):
			somaActivationFound = HFNLPpy_DendriticSANIPropagateStandard.simulateBiologicalHFnetworkSequenceNodesPropagateStandardEmulateVectorisedComputationOrder(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)				
		else:
			somaActivationFound = HFNLPpy_DendriticSANIPropagateStandard.simulateBiologicalHFnetworkSequenceNodesPropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSourceList, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)			
	return somaActivationFound
	
def simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
	somaActivationFound = HFNLPpy_DendriticSANIPropagateStandard.simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)
	return somaActivationFound
	
#independent method (does not need to be executed in order of wSource)
def simulateBiologicalHFnetworkSequenceNodePropagateForwardFull(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
	somaActivationFound = False
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	
	for wSource, conceptNeuronSource in enumerate(sentenceConceptNodeList):	#support for simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain:!biologicalSimulationEncodeSyntaxInDendriticBranchStructureFormat
	#orig for wSource in range(0, wTarget):
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		activationTime = calculateActivationTimeSequence(wSource)
		
		connectionTargetNeuronSetLocal = set()
		somaActivationFoundTemp = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSetLocal)
		
		if(wSource == len(sentenceConceptNodeList)-1):
			if(somaActivationFoundTemp):
				somaActivationFound = True
		
		connectionTargetNeuronSet = connectionTargetNeuronSet.union(connectionTargetNeuronSetLocal)
		resetConnectionTargetNeurons(connectionTargetNeuronSetLocal, True, conceptNeuronTarget)
	
	resetConnectionTargetNeurons(connectionTargetNeuronSet, False)
		
	return somaActivationFound

def selectActivatedNeurons(networkConceptNodeDict, connectionTargetNeuronSet):
	connectionTargetNeuronSetLocalFiltered = connectionTargetNeuronSet
	if(linkSimilarConceptNodes):
		connectionTargetNeuronSetLocalFiltered = HFNLPpy_hopfieldOperations.retrieveSimilarConcepts(networkConceptNodeDict, connectionTargetNeuronSetLocalFiltered)
	if(selectActivatedTop):
		connectionTargetNeuronSetLocalFiltered = selectTopKactivatedNeurons(connectionTargetNeuronSetLocalFiltered)
	return connectionTargetNeuronSetLocalFiltered
	

def selectTopKactivatedNeurons(connectionTargetNeuronSetLocal):
	selectActivatedTopKChecked = max([selectActivatedTopK, len(connectionTargetNeuronSetLocal)])
	connectionTargetNeuronList = list(connectionTargetNeuronSetLocal)
	connectionTargetNeuronActivationStrengthList = []
	for targetNeuron in connectionTargetNeuronList:
		activationStrength = calculateNeuronActivationStrength(targetNeuron)
		connectionTargetNeuronActivationStrengthList.append(activationStrength)
	#print("connectionTargetNeuronActivationStrengthList = ", connectionTargetNeuronActivationStrengthList)
	#print("len(connectionTargetNeuronList) = ", len(connectionTargetNeuronList))
	connectionTargetNeuronListFiltered = sortListByList(connectionTargetNeuronList, connectionTargetNeuronActivationStrengthList)
	connectionTargetNeuronListFiltered = connectionTargetNeuronListFiltered[0:selectActivatedTopKChecked]
	connectionTargetNeuronSetLocalFiltered = set(connectionTargetNeuronListFiltered)
	return connectionTargetNeuronSetLocalFiltered
		
def calculateNeuronActivationStrength(targetNeuron):
	#does not support resetConnectionTargetNeuronDendriteDuringActivation
	if(updateNeuronObjectActivationLevels):
		branchActivation = measureBranchActivationRecurse(targetNeuron.dendriticTree)
	elif(vectoriseComputationCurrentDendriticInput):
		branchActivation = measureDendriticTreeActivationVectorised(targetNeuron)
	return branchActivation
		
def sortListByList(A, B):
	#sorted_A = sorted(A, key=lambda x: B[A.index(x)])
    sorted_A = []
    for i in range(len(A)):
        index = B.index(min(B))
        sorted_A.append(A[index])
        A.pop(index)
        B.pop(index)
    return sorted_A
		

def connectionExists(nodeSource, nodeTarget, contextConnection):
	result = False
	if(contextConnection):
		if(nodeTarget.nodeName in nodeSource.HFcontextTargetConnectionDict):
			result = True
	else:
		if(nodeTarget.nodeName in nodeSource.HFcausalTargetConnectionDict):
			result = True
	return result
	
	
