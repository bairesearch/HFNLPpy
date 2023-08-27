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
HFNLP Biological Simulation - simulate training/inference of biological hopfield graph/network based on textual input

pseudo code;
for every time step/concept neuron (word w):
	calculate neuron activation...

training;
	activate concept neurons in order of sentence word order
	strengthen those synapses which directly precede/contribute to firing
		weaken those that do not
	this will enable neuron to fire when specific contextual instances are experienced
inference;
	calculate neuron firing exclusively from prior/contextual subsequence detections

"""


import numpy as np

from HFNLPpy_ScanGlobalDefs import *

def seedBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, seedSentenceConceptNodeList, numberOfSentences):
	for wSource, conceptNeuronSource in enumerate(seedSentenceConceptNodeList):
		sourceNeuronID = neuronIDdict[conceptNeuronSource.nodeName]
		#print("seedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName)
		HFconnectionMatrix.activation_state[sourceNeuronID] = HFactivationStateOn
		if(seedHFnetworkSubsequenceSimulateScan):
			simulateBiologicalHFnetworkSequencePropagateForward(networkConceptNodeDict, sentenceIndex, HFconnectionMatrix, HFnumberOfScanIterations)	
			
def trainBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, sentenceConceptNodeList, numberOfSentences):
	simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, sentenceConceptNodeList, numberOfSentences)	

def simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, neuronIDdict, HFconnectionMatrix, sentenceConceptNodeList, numberOfSentences):
	sentenceLength = len(sentenceConceptNodeList)
	for wSource, conceptNeuronSource in enumerate(sentenceConceptNodeList):
		sourceNeuronID = neuronIDdict[conceptNeuronSource.nodeName]
		HFconnectionMatrix.activation_state[sourceNeuronID] = HFactivationStateOn
		#print("seedBiologicalHFnetwork: wSource = ", wSource, ", conceptNeuronSource = ", conceptNeuronSource.nodeName)
		simulateBiologicalHFnetworkSequencePropagateForward(networkConceptNodeDict, sentenceIndex, HFconnectionMatrix, HFnumberOfScanIterations)

def simulateBiologicalHFnetworkSequencePropagateForward(networkConceptNodeDict, sentenceIndex, graph, num_time_steps):
	if(vectoriseComputation):
		return simulateBiologicalHFnetworkSequencePropagateForwardParallel(networkConceptNodeDict, sentenceIndex, graph, num_time_steps)
	else:
		return simulateBiologicalHFnetworkSequencePropagateForwardStandard(networkConceptNodeDict, sentenceIndex, graph, num_time_steps)

def simulateBiologicalHFnetworkSequencePropagateForwardParallel(networkConceptNodeDict, sentenceIndex, graph, num_time_steps):
	# Simulate the flow of information (activations) between adjacent neurons for each time step
	for t in range(num_time_steps):
		# Use a vectorized operation to update the activation state of each neuron at time t+1
		source_neurons = graph.edge_index[0]
		target_neurons = graph.edge_index[1]
		activation_state_t1 = graph.activation_state.clone()
		activation_state_t1[target_neurons] += graph.activation_state[source_neurons] * graph.edge_attr
		graph.activation_state = activation_state_t1
		#if(graph.activation_state > HFactivationThreshold):	#incomplete
		
def simulateBiologicalHFnetworkSequencePropagateForwardStandard(networkConceptNodeDict, sentenceIndex, graph, num_time_steps):
	# Simulate the flow of information (activations) between adjacent neurons for each time step
	for t in range(num_time_steps):
		for i in range(graph.edge_index.shape[1]):	#for each edge
			source_neuron, target_neuron = graph.edge_index[:, i]
			activation_state_t1 = graph.activation_state.clone()
			activation_state_t1[target_neuron] += graph.activation_state[source_neuron] * graph.edge_attr[i]
			graph.activation_state = activation_state_t1
			#if(graph.activation_state[source_neuron] > HFactivationThreshold):	#incomplete

