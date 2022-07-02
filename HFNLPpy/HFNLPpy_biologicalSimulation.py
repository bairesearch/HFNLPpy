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
from HFNLPpy_biologicalSimulationNode import *
import HFNLPpy_biologicalSimulationGenerate
import HFNLPpy_hopfieldOperations
if(vectoriseComputation):
	import HFNLPpy_biologicalSimulationPropagateVectorised
else:
	import HFNLPpy_biologicalSimulationPropagateStandard

printVerbose = False


#alwaysAddPredictionInputFromPreviousConcept = False
#if(vectoriseComputation):
#	alwaysAddPredictionInputFromPreviousConcept = True #ensures that simulateBiologicalHFnetworkSequenceNodePropagateParallel:conceptNeuronBatchIndexFound

drawBiologicalSimulation = False	#default: True
if(drawBiologicalSimulation):
	drawBiologicalSimulationDendriticTreeSentence = True	#default: True	#draw graph for sentence neurons and their dendritic tree
	if(drawBiologicalSimulationDendriticTreeSentence):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawSentence
	drawBiologicalSimulationDendriticTreeNetwork = False	#default: True	#draw graph for entire network (not just sentence)
	if(drawBiologicalSimulationDendriticTreeNetwork):
		import HFNLPpy_biologicalSimulationDraw as HFNLPpy_biologicalSimulationDrawNetwork


def simulateBiologicalHFnetwork(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):
	simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)	


#if (!biologicalSimulation:useDependencyParseTree):

def simulateBiologicalHFnetworkSequenceTrain(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):

	#cannot clear now as HFNLPpy_biologicalSimulationDrawSentence/HFNLPpy_biologicalSimulationDrawNetwork memory structure is not independent (diagnose reason for this);
	
	sentenceLength = len(sentenceConceptNodeList)
	
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	
	for wTarget in range(1, sentenceLength):	#wTarget>=1: do not create (recursive) connection from conceptNode to conceptNode branchIndex1=0
		somaActivationFound = simulateBiologicalHFnetworkSequenceNodePropagateWrapper(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, connectionTargetNeuronSet)

		if(somaActivationFound):
			#if(printVerbose):
			print("somaActivationFound")
		else:
			#if(printVerbose):
			print("!somaActivationFound: addPredictiveSequenceToNeuron")
			dendriticBranchMaxW = wTarget-1
			expectFurtherSubbranches = True
			if(wTarget == 1):
				expectFurtherSubbranches = False
			conceptNeuronTarget = sentenceConceptNodeList[wTarget]
			HFNLPpy_biologicalSimulationGenerate.addPredictiveSequenceToNeuron(conceptNeuronTarget, sentenceIndex, sentenceConceptNodeList, conceptNeuronTarget.dendriticTree, dendriticBranchMaxW, 0, expectFurtherSubbranches)
	
	#reset dendritic trees
	if(biologicalSimulationForward):
		for conceptNeuron in connectionTargetNeuronSet:
			resetDendriticTreeActivation(conceptNeuron)

	drawBiologicalSimulationStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList)

			
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
	if(vectoriseComputationCurrentDendriticInput):
		somaActivationFound = HFNLPpy_biologicalSimulationPropagateVectorised.simulateBiologicalHFnetworkSequenceNodePropagateParallel(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)
	else:
		somaActivationFound = HFNLPpy_biologicalSimulationPropagateStandard.simulateBiologicalHFnetworkSequenceNodePropagateStandard(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, activationTime, wSource, conceptNeuronSource, wTarget, conceptNeuronTarget, connectionTargetNeuronSet)			
	return somaActivationFound

def simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
	somaActivationFound = HFNLPpy_biologicalSimulationPropagateStandard.simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget)
	return somaActivationFound
	
#independent method (does not need to be executed in order of wSource)
def simulateBiologicalHFnetworkSequenceNodePropagateForwardFull(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget):
	somaActivationFound = False
	connectionTargetNeuronSet = set()	#for posthoc network deactivation
	
	for wSource, conceptNeuronSource in enumerate(sentenceConceptNodeList):	#support for simulateBiologicalHFnetworkSequenceSyntacticalBranchDPTrain:!biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	#orig for wSource in range(0, wTarget):
		conceptNeuronSource = sentenceConceptNodeList[wSource]
		activationTime = calculateActivationTimeSequence(wSource)
		somaActivationFoundTemp = simulateBiologicalHFnetworkSequenceNodePropagateForward(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList, wTarget, conceptNeuronTarget, activationTime, wSource, conceptNeuronSource, connectionTargetNeuronSet)
		if(wSource == len(sentenceConceptNodeList)-1):
			if(somaActivationFoundTemp):
				somaActivationFound = True
			
	for conceptNeuron in connectionTargetNeuronSet:
		resetDendriticTreeActivation(conceptNeuron)
		
	return somaActivationFound

	
def drawBiologicalSimulationStatic(networkConceptNodeDict, sentenceIndex, sentenceConceptNodeList):
	if(drawBiologicalSimulation):
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
			
			
