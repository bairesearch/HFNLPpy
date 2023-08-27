"""HFNLPpy_SANIGlobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Biological Simulation Global Defs

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

vectoriseComputation = True	#parallel processing for optimisation

selectActivatedTop = True	#select activated top k target neurons during propagation test	#incomplete
if(selectActivatedTop):
	selectActivatedTopK = 3
	
HFreadSavedConnectionsMatrix = False

HFnumberOfScanIterations = 3	#1	#number of timesteps for scan (burst)

HFactivationStateOn = 1.0	
HFactivationThreshold = 0.5	#TODO: calibrate this	#incomplete
HFconnectionWeightObs = 0.1	#connection weight to add for each observed instance of an adjacent source-target word pair/tuple in a sentence; ie (w, w+1)

HFconnectionMatrixFileName = "HFconnectionMatrix.csv"
HFconceptNeuronsFileName = "HFconceptNeurons.csv"

HFNLPnonrandomSeed = False	#initialise (dependent var)

#### seed HF network with subsequence ####

seedHFnetworkSubsequence = False #seed/prime HFNLP network with initial few words of a trained sentence and verify that full sentence is sequentially activated (interpret last sentence as target sequence, interpret first seedHFnetworkSubsequenceLength words of target sequence as seed subsequence)
if(seedHFnetworkSubsequence):
	#seedHFnetworkSubsequence currently requires !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	seedHFnetworkSubsequenceLength = 4	#must be < len(targetSentenceConceptNodeList)
	seedHFnetworkSubsequenceBasic = False	#emulate simulateBiologicalHFnetworkSequenceTrain:simulateBiologicalHFnetworkSequenceNodePropagateWrapper method (only propagate those activate neurons that exist in the target sequence); else propagate all active neurons
	seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant = True

	seedHFnetworkSubsequenceSimulateScan = True	#optional simulate scan during seed as if the graph activations were propagated as normal
