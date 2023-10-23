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


debugAlgorithmMatrix = False
debugHFconnectionMatrix = False

#### memory constraints ####
algorithmMatrixSingleTensor = False	#store context size array (and simulated dendritic branches) in pytorch tensor rather than python list	#requires high ram
useHFconnectionMatrixBasicSparse = False		#reduces ram requirements (especially for large HFconnectionMatrixBasicMaxConcepts)
if(debugHFconnectionMatrix):
	HFconnectionMatrixBasicMaxConcepts = 20	 #[Xdataset4PartSmall0000.xml.verifyOldSentenceSomaActivationFound0]
else:
	HFconnectionMatrixBasicMaxConcepts = 1000	#200	#1000	#default:100000	#maximum number of concepts to store	#size of HFconnectionMatrix = HFconnectionMatrixBasicMaxConcepts^2	#CHECKTHIS (should be <= number words in dic)

algorithmMatrixSingleTensorEfficientAdd = False	#initialise (dependent var)
if(algorithmMatrixSingleTensor):
	algorithmMatrixSingleTensorEfficientAdd = False	#incomplete	#efficiently add context to connection matrix (use parallelised algorithm) 

#### simulated dendritic branches ####

simulatedDendriticBranches = False	#independent dendritic branches
HFconnectionMatrixMinValue = 0
if(simulatedDendriticBranches):
	simulatedDendriticBranchesMinMatchStrength = 1.0	#minimum branch match strength for comparison before randomise selection of new branch to write	#CHECKTHIS
	simulatedDendriticBranchesInitialisation = False #incomplete #perform random initialisation to break symmetry (required to select more than 1 dendritic branch)
	if(simulatedDendriticBranchesInitialisation):
		simulatedDendriticBranchesInitialisationWeight = 0.01	#only apply a very small randomisation magnitude to break symmetry
		HFconnectionMatrixMinValue = simulatedDendriticBranchesInitialisationWeight
	numberOfDendriticBranches = 10
else: 
	numberOfDendriticBranches = 1
	simulatedDendriticBranchesInitialisation = False

#### topk selection ####

selectActivatedTop = True	#mandatory (implied) select activated top k target neurons during propagation test
if(selectActivatedTop):
	matrixPropagateTopKconceptNodes = 1	#number of top k elements to save
	matrixPropagateTopKcontextSize = 1	#number of top k elements to save
	matrixPropagateTopKdendriticBranches = 1 	#number of top k elements to save
	
#### context connections matrix ####

if(debugHFconnectionMatrix):
	contextSizeMax = 30 #[Xdataset4PartSmall0000.xml.verifyOldSentenceSomaActivationFound0]
else:
	contextSizeMax = 100 #max sequential context width use for next word prediction
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
