"""HFNLPpy_MatrixGlobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Matrix Global Defs

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np


#### topk selection ####

selectActivatedTop = True	#mandatory (implied) select activated top k target neurons during propagation test
if(selectActivatedTop):
	matrixPropagateTopK1 = 1	#number of top k elements to save (1)
	matrixPropagateTopK2 = 1	#number of top k elements to save (2)

#### context connections matrix ####

contextSizeMax = 1000 #max sequential context width use for next word prediction
contextMatrixWeightStore = False	#optional	#CHECKTHIS

#### test harness (compare standard/vectorised computation) ####

biologicalSimulationTestHarness = True

HFNLPnonrandomSeed = False	#initialise (dependent var)
if(biologicalSimulationTestHarness):
	HFNLPnonrandomSeed = True	#always generate the same set of random numbers upon execution
	
#### seed HF network with subsequence ####

seedHFnetworkSubsequence = True #seed/prime HFNLP network with initial few words of a trained sentence and verify that full sentence is sequentially activated (interpret last sentence as target sequence, interpret first seedHFnetworkSubsequenceLength words of target sequence as seed subsequence)
if(seedHFnetworkSubsequence):
	#seedHFnetworkSubsequence currently requires !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	seedHFnetworkSubsequenceLength = 4	#must be < len(targetSentenceConceptNodeList)
	seedHFnetworkSubsequenceBasic = False	#emulate simulateBiologicalHFnetworkSequenceTrain:simulateBiologicalHFnetworkSequenceNodePropagateWrapper method (only propagate those activate neurons that exist in the target sequence); else propagate all active neurons
	seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant = True

enforceMinimumEncodedSequenceLength = True	#do not execute addPredictiveSequenceToNeuron if predictive sequence is short (ie does not use up the majority of numberOfBranches1)
if(enforceMinimumEncodedSequenceLength):
	minimumEncodedSequenceLength = 4	#should be high enough to fill a significant proportion of dendrite vertical branch length (numberOfBranches1)	#~seedHFnetworkSubsequenceLength

#### propagation algorithm (source/target activation) ####

biologicalSimulationForward = True	#mandatory (implied)


#### error ####

def printe(str):
	print(str)
	exit()
