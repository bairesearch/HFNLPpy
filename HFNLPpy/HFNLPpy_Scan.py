"""HFNLPpy_Scan.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Scan - simulate training/inference of biological hopfield graph/network based on textual input

"""


import numpy as np
import torch as pt

from HFNLPpy_ScanGlobalDefs import *

if(drawBiologicalSimulation):
	import HFNLPpy_hopfieldGraphDraw

#based on HFNLPpy_DendriticSANI:seedBiologicalHFnetwork
def seedBiologicalHFnetwork(networkConceptNodeDict, networkSize, sentenceIndex, HFconnectionGraphObject, seedSentenceConceptNodeList, numberOfSentences):
	neuronNamelist, neuronIDdict, HFconnectionGraph = (HFconnectionGraphObject.neuronNamelist, HFconnectionGraphObject.neuronIDdict, HFconnectionGraphObject.HFconnectionGraph)
	
	targetSentenceConceptNodeList = seedSentenceConceptNodeList
		
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	connectionTargetNeuronIDsetFilteredPrevious = set()
	if(not seedHFnetworkSubsequenceBasic):
		conceptNeuronSourceList = []
		
	for wSource in range(len(targetSentenceConceptNodeList)-1):
		wTarget = wSource+1
		conceptNeuronSource = targetSentenceConceptNodeList[wSource]
		conceptNeuronTarget = targetSentenceConceptNodeList[wTarget]
		
		#ensure source/target neuronIDs are in dictionary;
		if(conceptNeuronSource.nodeName not in neuronIDdict):
			printe("HFNLPpy_Scan:seedBiologicalHFnetwork error: (conceptNeuronSource.nodeName not in neuronIDdict)")
		if(conceptNeuronTarget.nodeName not in neuronIDdict):
			printe("HFNLPpy_Scan:seedBiologicalHFnetwork error: (conceptNeuronTarget.nodeName not in neuronIDdict)")
		sourceNeuronID = neuronIDdict[conceptNeuronSource.nodeName]
		targetNeuronID = neuronIDdict[conceptNeuronTarget.nodeName]

		print("seedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName, ", wTarget = ", wTarget, ", conceptNeuronTarget = ", conceptNeuronTarget.nodeName, ", sourceNeuronID = ", sourceNeuronID, ", targetNeuronID = ", targetNeuronID)

		connectionTargetNeuronIDset = set()
		connectionTargetNeuronIDsetFiltered = set()
		if(wSource < seedHFnetworkSubsequenceLength):
			HFconnectionGraph.activationState[sourceNeuronID] = HFactivationStateOn
			HFconnectionGraph.activationLevel[sourceNeuronID] = HFactivationLevelOn
			if(seedHFnetworkSubsequenceSimulateScan):
				simulateBiologicalHFnetworkSequencePropagateForward(sentenceIndex, HFconnectionGraph, HFnumberOfScanIterations, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered)	
		else:
			simulateBiologicalHFnetworkSequencePropagateForward(sentenceIndex, HFconnectionGraph, HFnumberOfScanIterations, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered)	

		#print("connectionTargetNeuronIDsetFiltered = ", connectionTargetNeuronIDsetFiltered)
		conceptNeuronSourceList = connectionTargetNeuronIDsetFiltered
		for neuronID in connectionTargetNeuronIDset:
			conceptNode = getConceptNode(networkConceptNodeDict, neuronNamelist, neuronID)
			connectionTargetNeuronSet.add(conceptNode)
		somaActivationFound = resetConnectionTargetNeurons(True, HFconnectionGraph, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered, connectionTargetNeuronIDsetFilteredPrevious, sourceNeuronID, targetNeuronID)	

		if(drawBiologicalSimulation):
			updateNeuronObjectActivationStates(networkConceptNodeDict, neuronNamelist, neuronIDdict, HFconnectionGraph, connectionTargetNeuronSet, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered)
			if(drawBiologicalSimulationSentence):
				HFNLPpy_hopfieldGraphDraw.drawHopfieldGraphSentenceStatic(sentenceIndex, targetSentenceConceptNodeList, networkSize, drawBiologicalSimulationPlot, drawBiologicalSimulationSave)
			if(drawBiologicalSimulationNetwork):
				HFNLPpy_hopfieldGraphDraw.drawHopfieldGraphNetworkStatic(sentenceIndex, networkConceptNodeDict, drawBiologicalSimulationPlot, drawBiologicalSimulationSave)
		
		connectionTargetNeuronIDsetFilteredPrevious = connectionTargetNeuronIDsetFiltered.copy()
		
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
				
def simulateBiologicalHFnetworkSequencePropagateForward(sentenceIndex, HFconnectionGraph, num_time_steps, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered):
	if(vectoriseComputation):
		return simulateBiologicalHFnetworkSequencePropagateForwardParallel(sentenceIndex, HFconnectionGraph, num_time_steps, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered)
	else:
		return simulateBiologicalHFnetworkSequencePropagateForwardStandard(sentenceIndex, HFconnectionGraph, num_time_steps, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered)

def simulateBiologicalHFnetworkSequencePropagateForwardParallel(sentenceIndex, HFconnectionGraph, num_time_steps, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered):
	# Simulate the flow of information (activations) between adjacent neurons for each time step
	#print("before HFconnectionGraph.activationLevel = ", HFconnectionGraph.activationLevel)
	for t in range(num_time_steps):
		# Use a vectorized operation to update the activation state of each neuron at time t+1
		sourceNeurons = HFconnectionGraph.edge_index[0]
		targetNeurons = HFconnectionGraph.edge_index[1]
		sourceActivationLevelThresholded = activationFunction(HFconnectionGraph.activationLevel[sourceNeurons])
		HFconnectionGraph.activationLevel.scatter_add_(0, targetNeurons, HFconnectionGraph.activationLevel[sourceNeurons] * HFconnectionGraph.edge_attr)
		HFconnectionGraph.activationLevel[targetNeurons] = activationThreshold(HFconnectionGraph.activationLevel[targetNeurons])
		HFconnectionGraph.activationState[targetNeurons] = (HFconnectionGraph.activationLevel[targetNeurons] >= HFactivationFunctionThreshold)
	#print("after HFconnectionGraph.activationLevel = ", HFconnectionGraph.activationLevel)
	
	connectionTargetNeuronID = pt.nonzero(HFconnectionGraph.activationState).squeeze(1)
	connectionTargetNeuronIDList = connectionTargetNeuronID.tolist()
	connectionTargetNeuronIDset.update(set(connectionTargetNeuronIDList))
	if(selectActivatedTop):
		connectionTargetNeuronIDListFiltered = selectTopKactivatedNeuronsParallel(HFconnectionGraph, connectionTargetNeuronIDList)
		connectionTargetNeuronIDsetFiltered.update(set(connectionTargetNeuronIDListFiltered))
	else:
		connectionTargetNeuronIDsetFiltered.update(connectionTargetNeuronIDset)	#no difference

def debugPrintActivationState(HFconnectionGraph):
	print("HFconnectionGraph.activationLevel = ", HFconnectionGraph.activationLevel)
	print("source HFconnectionGraph.edge_index[0] = ", HFconnectionGraph.edge_index[0])
	print("target HFconnectionGraph.edge_index[1] = ", HFconnectionGraph.edge_index[1])
	print("HFconnectionGraph.edge_attr = ", HFconnectionGraph.edge_attr)
		
#this method currently does not work because it will keep propagating activations at each new i
def simulateBiologicalHFnetworkSequencePropagateForwardStandard(sentenceIndex, HFconnectionGraph, num_time_steps, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered):
	# Simulate the flow of information (activations) between adjacent neurons for each time step
	print("before HFconnectionGraph.activationLevel = ", HFconnectionGraph.activationLevel)
	if(selectActivatedTop):
		connectionTargetNeuronActivationLevelList = []
		connectionTargetNeuronIDList = []
	for t in range(num_time_steps):
		for i in range(HFconnectionGraph.edge_index.shape[1]):	#for each edge
			#print("i = ", i)
			sourceNeuron, targetNeuron = HFconnectionGraph.edge_index[:, i]
			sourceActivationLevelThresholded = activationFunction(HFconnectionGraph.activationLevel[sourceNeuron])
			HFconnectionGraph.activationLevel[targetNeuron] += sourceActivationLevelThresholded * HFconnectionGraph.edge_attr[i]
			HFconnectionGraph.activationLevel[targetNeuron] = activationThreshold(HFconnectionGraph.activationLevel[targetNeuron])
			HFconnectionGraph.activationState[targetNeuron] = (HFconnectionGraph.activationLevel[targetNeuron] >= HFactivationFunctionThreshold)
	for i in range(HFconnectionGraph.edge_index.shape[1]):	#for each edge
		if(HFconnectionGraph.activationState[sourceNeuron]):
			if(selectActivatedTop):
				connectionTargetNeuronIDList.append(targetNeuron)
				connectionTargetNeuronActivationLevelList.append(HFconnectionGraph.activationLevel[sourceNeuron])
			connectionTargetNeuronIDset.add(targetNeuron)
	print("after HFconnectionGraph.activationLevel = ", HFconnectionGraph.activationLevel)
	
	if(selectActivatedTop):
		#connectionTargetNeuronIDset.update(set(connectionTargetNeuronIDList))	#already updated
		connectionTargetNeuronIDListFiltered = selectTopKactivatedNeuronsStandard(connectionTargetNeuronIDList, connectionTargetNeuronActivationLevelList)
		connectionTargetNeuronIDsetFiltered.update(set(connectionTargetNeuronIDListFiltered))
	else:
		connectionTargetNeuronIDsetFiltered.update(connectionTargetNeuronIDset)	#no difference

'''
def performTargetNeuronActivationDrain(connectionTargetNeuronIDset):
	#performs neuron activation loss for a time step
	for targetNeuronID in connectionTargetNeuronIDset:
		HFactivationDrain
'''

def selectTopKactivatedNeuronsStandard(connectionTargetNeuronIDList, connectionTargetNeuronActivationLevelList):
	selectActivatedTopKChecked = min([selectActivatedTopK, len(connectionTargetNeuronActivationLevelList)])
	connectionTargetNeuronIDListFiltered = sortListByList(connectionTargetNeuronIDList, connectionTargetNeuronActivationLevelList)
	connectionTargetNeuronIDListFiltered = connectionTargetNeuronIDListFiltered[0:selectActivatedTopKChecked]
	return connectionTargetNeuronIDListFiltered
	
def selectTopKactivatedNeuronsParallel(HFconnectionGraph, connectionTargetNeuronIDList):
	selectActivatedTopKChecked = min([selectActivatedTopK, len(connectionTargetNeuronIDList)])
	connectionTargetNeuronIDListFiltered = pt.topk(HFconnectionGraph.activationLevel, selectActivatedTopKChecked).indices.tolist()
	#print("connectionTargetNeuronIDListFiltered = ", connectionTargetNeuronIDListFiltered)
	return connectionTargetNeuronIDListFiltered
	
def sortListByList(A, B):
	#sorted_A = sorted(A, key=lambda x: B[A.index(x)])
    sorted_A = []
    for i in range(len(A)):
        index = B.index(min(B))
        sorted_A.append(A[index])
        A.pop(index)
        B.pop(index)
    return sorted_A

def resetConnectionTargetNeurons(duringSourcePropagation, HFconnectionGraph, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered, connectionTargetNeuronIDsetFilteredPrevious, sourceNeuronID, targetNeuronID):	
	if(HFresetActivations):
		#neuron reset/repolarisation upon activation
		connectionTargetNeuronIDsetReset = connectionTargetNeuronIDset.difference(connectionTargetNeuronIDsetFiltered)	#deactivate newly activated neurons that are below the threshold	#ie connectionTargetNeuronIDsetUnfiltered	
			#connectionTargetNeuronIDsetReset = set()
		#print("connectionTargetNeuronIDsetFiltered = ", connectionTargetNeuronIDsetFiltered)
		#print("connectionTargetNeuronIDsetFilteredPrevious = ", connectionTargetNeuronIDsetFilteredPrevious)
		if(HFresetActivationsPrevious):
			connectionTargetNeuronIDsetReset = connectionTargetNeuronIDsetReset.union(connectionTargetNeuronIDsetFilteredPrevious)	#deactivate previously activated neurons (that have already been propagated)	#& symbol does not work
		connectionTargetNeuronIDsetReset = connectionTargetNeuronIDsetReset - set([sourceNeuronID])	#keep artifically activated source neuron on	#assume part of connectionTargetNeuronIDsetFiltered
		#print("connectionTargetNeuronIDsetReset = ", connectionTargetNeuronIDsetReset)
		targetNeurons = pt.tensor(list(connectionTargetNeuronIDsetReset), dtype=pt.int64)
		HFconnectionGraph.activationState[targetNeurons] = HFactivationStateReset
		HFconnectionGraph.activationLevel[targetNeurons] = HFactivationLevelReset	#performs topk selection (deactivates all neurons that do not meet threshold)
	
		#if(drawBiologicalSimulation):
		#	connectionTargetNeuronIDsetFiltered.difference_update(connectionTargetNeuronIDsetReset)
		#OLD: targetNeuron reset is not currently required with HFactivationFunctionThresholdApply applied before propagating source activations
		
	somaActivationFound = False
	if(targetNeuronID in connectionTargetNeuronIDsetFiltered):
		somaActivationFound = True
		
	return somaActivationFound

def activationFunction(activationLevel):
	if(HFactivationFunctionThresholdApply):
		activationLevel = pt.where(activationLevel < HFactivationFunctionThreshold, pt.tensor(HFactivationLevelOff), pt.tensor(HFactivationLevelOn))
	return activationLevel
	
def activationThreshold(activationLevel):
	if(HFactivationThresholdApply):
		activationLevel = pt.clamp(activationLevel, max=HFactivationThreshold)
	return activationLevel

def getConceptNode(networkConceptNodeDict, neuronNamelist, neuronID):
	#neuronID = neuronIDdict[neuronName]
	neuronName = neuronNamelist[neuronID]
	#print("neuronName = ", neuronName)
	#print("networkConceptNodeDict = ", networkConceptNodeDict)
	conceptNode = networkConceptNodeDict[neuronName]
	return conceptNode

def updateNeuronObjectActivationStates(networkConceptNodeDict, neuronNamelist, neuronIDdict, HFconnectionGraph, connectionTargetNeuronSet, connectionTargetNeuronIDset, connectionTargetNeuronIDsetFiltered):
	#reset all neuron object states to unactivated;
	for conceptNode in connectionTargetNeuronSet:	#optimisation: only consider currently or previously activated neurons
		#conceptNode.activationLevel = 0
		conceptNode.activationState = False
		conceptNode.activationStateFiltered = False
	#set all activated neuron object states;
	for connectionTargetNeuronID in connectionTargetNeuronIDsetFiltered:
		conceptNode = getConceptNode(networkConceptNodeDict, neuronNamelist, connectionTargetNeuronID)
		#conceptNode.activationLevel = 
		conceptNode.activationStateFiltered = True
	for connectionTargetNeuronID in connectionTargetNeuronIDset:
		conceptNode = getConceptNode(networkConceptNodeDict, neuronNamelist, connectionTargetNeuronID)
		#conceptNode.activationLevel = 
		conceptNode.activationState = True
		
