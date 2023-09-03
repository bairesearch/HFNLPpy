"""HFNLPpy_DendriticSANIGlobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Scan Global Defs

"""

import numpy as np
from HFNLPpy_globalDefs import *

debugLowActivationThresholds = True	#enable for debugging
debugPyGactivationPropagationImplementationBug = False

#### computation type ####

vectoriseComputation = True	#parallel processing for optimisation	#!vectoriseComputation:simulateBiologicalHFnetworkSequencePropagateForwardStandard does not work because it will keep propagating activations at each new i

#### topk selection ####

if(debugLowActivationThresholds):
	selectActivatedTop = True	#optional
else:
	selectActivatedTop = True	#select activated top k target neurons during propagation test	#recommended (otherwise will overactivate network)
if(selectActivatedTop):
	selectActivatedTopK = 10	#10	#3
	
HFreadSavedConnectionsMatrix = False

HFnumberOfScanIterations = 1	#3	#1	#number of timesteps for scan (per iteration)

#### thresholds ####

HFactivationStateOn = True	
HFactivationLevelOn = 1.0	
HFactivationStateOff = False
HFactivationLevelOff = 0.0

HFresetActivations = True	#default: True (disable for debugging)
if(HFresetActivations):
	HFresetActivationsPrevious = True
	HFactivationStateReset = False
	HFactivationLevelReset = 0.0	#CHECKTHIS
#HFactivationDrain = 0.05	#incomplete #activation level reduction per time step for all target neurons (prediction candidates)

HFactivationFunctionThresholdApply = True
HFactivationFunctionThreshold = 1.0	#0.99	#1.0	#0.5	#TODO: calibrate this
HFactivationThresholdApply = True

if(debugLowActivationThresholds):
	HFconnectionWeightObs = 1.0
	HFactivationThreshold = 1.0	#do not allow activation values to increase beyond this value
else:
	HFconnectionWeightObs = 0.1	#connection weight to add for each observed instance of an adjacent source-target word pair/tuple in a sentence; ie (w, w+1)
	HFactivationThreshold = 10.0
	
#### file i/o ####

HFconnectionMatrixFileName = "HFconnectionGraph.csv"
HFconceptNeuronsFileName = "HFconceptNeurons.csv"

#### test harness ####

HFNLPnonrandomSeed = False	#initialise (dependent var)

#### draw ####

drawBiologicalSimulation = True	#optional
if(drawBiologicalSimulation):
	drawBiologicalSimulationPlot = True	#default: True
	drawBiologicalSimulationSave = False	#default: False	#save to file
	drawBiologicalSimulationSentence = True	#default: True	#draw graph for sentence neurons and their dendritic tree
	drawBiologicalSimulationNetwork = True	#default: False	#draw graph for entire network (not just sentence)
		
#### seed HF network with subsequence ####

seedHFnetworkSubsequence = True #seed/prime HFNLP network with initial few words of a trained sentence and verify that full sentence is sequentially activated (interpret last sentence as target sequence, interpret first seedHFnetworkSubsequenceLength words of target sequence as seed subsequence)
if(seedHFnetworkSubsequence):
	#seedHFnetworkSubsequence currently requires !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	seedHFnetworkSubsequenceLength = 4	#must be < len(targetSentenceConceptNodeList)
	seedHFnetworkSubsequenceBasic = False	#emulate simulateBiologicalHFnetworkSequenceTrain:simulateBiologicalHFnetworkSequenceNodePropagateWrapper method (only propagate those activate neurons that exist in the target sequence); else propagate all active neurons
	seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant = True

	seedHFnetworkSubsequenceSimulateScan = True	#optional #simulate scan during seed as if the graph activations were propagated as normal

enforceMinimumEncodedSequenceLength = True	#do not expect prediction to work if predictive sequence is short
if(enforceMinimumEncodedSequenceLength):
	minimumEncodedSequenceLength = 4

def printe(str):
	print(str)
	exit()
