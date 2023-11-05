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

#### SANI ####
algorithmMatrixSANI = True	#emulate DendriticSANIbiologicalSimulationSimple
if(algorithmMatrixSANI):
	#select algorithmMatrixSANI method (select one);
	algorithmMatrixSANImethodAddActivationAcrossSegments = True	#default: True	#it is not necessary that previous segments be activated, but their activation will bias the selection of a particular dendritic branch
	algorithmMatrixSANImethodEnforceSequentialActivationAcrossSegments = False	#not yet implemented (else sum sequential segment activations)
	#select sequentialSegmentContextEncoding method (select one);
	sequentialSegmentContextEncodingAbsoluteLinear = False
	if(sequentialSegmentContextEncodingAbsoluteLinear):
		sequentialSegmentContextEncodingAbsoluteLinearSize = 1	#number of tokens per segment
	sequentialSegmentContextEncodingRelativeLinear = True
	sequentialSegmentContextEncodingRelativeExponential = False	#sequential segments capture input (past context tokens) at exponentially further distances
	#sequentialSegmentContextEncodingRandom = False #encodings are partially randomised (note dendritic SANI implementation creates multiple semi-random encodings of past context in different dendritic branches) 
	numberOfBranchSequentialSegments = 5
	normaliseConnectionStrengthWrtContextLength = False
else:
	algorithmMatrixSANImethodAddActivationAcrossSegments = False
	algorithmMatrixSANImethodEnforceSequentialActivationAcrossSegments = False
	algorithmMatrixSANImethodTopkActivationAcrossSegments = False
	algorithmMatrixSANImethodAddActivationAcrossSegmentsOld = False
	normaliseConnectionStrengthWrtContextLength = True	#optional	#orig:True	#CHECKTHIS

#### memory constraints ####
#algorithmMatrixTensorDim = 2	#default
algorithmMatrixTensorDim = 4	#optional	#store context size array (and simulated dendritic branches) in pytorch tensor rather than python list	#requires high ram
if(algorithmMatrixTensorDim != 4):
	if(algorithmMatrixSANImethodAddActivationAcrossSegments or algorithmMatrixSANImethodEnforceSequentialActivationAcrossSegments):
		algorithmMatrixTensorDim = 3 #mandatory
HFconnectionMatrixAlgorithmSparse = False		#incomplete: requires hybrid dense-sparse tensor implementation such that tensor can still be indexed #reduces ram requirements (especially for large HFconnectionMatrixBasicMaxConcepts)
HFconnectionMatrixAlgorithmSplit = True		#store each column of connection matrix in separate array (or hard disk file) 	#currently in testing #reduces ram requirements (especially for large HFconnectionMatrixBasicMaxConcepts)	#store each column of connection matrix in separate array (or hard drive file) - bring into memory on demand (depending on the precise activated columns of the contextVector being compared)
if(debugHFconnectionMatrix):
	HFconnectionMatrixBasicMaxConcepts = 200	#200	#20	 #[Xdataset4PartSmall0000.xml.verifyOldSentenceSomaActivationFound0]
else:
	HFconnectionMatrixBasicMaxConcepts = 1000	#1000	#default:100000	#maximum number of concepts to store	#size of HFconnectionMatrix = HFconnectionMatrixBasicMaxConcepts^2	#CHECKTHIS (should be <= number words in dic)

if(HFconnectionMatrixAlgorithmSplit):
	HFconnectionMatrixAlgorithmSplitDatabase = False	#optional	#store each column of connection matrix in separate hard drive file and bring into RAM at start of sentence
	HFconnectionMatrixAlgorithmContextVectorSparse = True	#store context vector as sparse tensor - only index context vector indices of connection matrix (computation or memory efficient)
	HFconnectionMatrixAlgorithmGPU = True	#store connection matrix and context vector in GPU
	HFconnectionMatrixAlgorithmNormaliseStore = False	#False: only store min/max of each target (row) word, for dynamic normalisation (on demand)
	if(HFconnectionMatrixAlgorithmSplitDatabase):
		assert (algorithmMatrixTensorDim == 4)	#HFconnectionMatrixAlgorithmSplitDatabase currently requires algorithmMatrixTensorDim=4 such that the file i/o code can be simplified
		matrixDatabaseFileNameStart = "HFmatrixDatabaseSourceNeuronID"
		matrixDatabaseFileNameEnd = ".csv"
		matrixDatabasePathName = "database/"
		matrixDatabaseFileNameMin = "Min"
		matrixDatabaseFileNameMax = "Max"
else:
	HFconnectionMatrixAlgorithmSplitDatabase = False
	HFconnectionMatrixAlgorithmContextVectorSparse = False
	HFconnectionMatrixAlgorithmGPU = True	#store connection matrix and context vector in GPU
	HFconnectionMatrixAlgorithmNormaliseStore = True	#True: store a normalised connection matrix in RAM (must recalculate entire normalised arrays: not computation or memory efficient)
	
if(HFconnectionMatrixAlgorithmContextVectorSparse):
	HFcontextVectorSparseNull = -1
	
#### simulated dendritic branches ####
simulatedDendriticBranches = False	#independent dendritic branches
HFconnectionMatrixAlgorithmMinValue = 0
if(simulatedDendriticBranches):
	simulatedDendriticBranchesMinMatchStrength = 1.0	#minimum branch match strength for comparison before randomise selection of new branch to write	#CHECKTHIS
	simulatedDendriticBranchesInitialisation = False #incomplete #perform random initialisation to break symmetry (required to select more than 1 dendritic branch)
	if(simulatedDendriticBranchesInitialisation):
		simulatedDendriticBranchesInitialisationWeight = 0.01	#only apply a very small randomisation magnitude to break symmetry
		HFconnectionMatrixAlgorithmMinValue = simulatedDendriticBranchesInitialisationWeight
	numberOfIndependentDendriticBranches = 10
else: 
	numberOfIndependentDendriticBranches = 1
	simulatedDendriticBranchesInitialisation = False

#### topk selection ####
selectActivatedTop = True	#mandatory (implied) select activated top k target neurons during propagation test
if(selectActivatedTop):
	matrixPropagateTopKconceptNodes = 1	#number of top k elements to save
	matrixPropagateTopKcontextSize = 1	#number of top k elements to save
	matrixPropagateTopKsequentialSegments = 1	#number of top k elements to save
	matrixPropagateTopKdendriticBranches = 1 	#number of top k elements to save
	if(algorithmMatrixSANI):
		matrixPropagateTopKsecondIndex = matrixPropagateTopKsequentialSegments
	else:
		matrixPropagateTopKsecondIndex = matrixPropagateTopKcontextSize
	
#### context connections matrix ####
if(debugHFconnectionMatrix):
	contextSizeMax = 30 #[Xdataset4PartSmall0000.xml.verifyOldSentenceSomaActivationFound0]
else:
	contextSizeMax = 100 #max sequential context width use for next word prediction
contextMatrixWeightStore = False	#optional	#CHECKTHIS

	
#### propagation algorithm (source/target activation) ####
biologicalSimulationForward = True	#mandatory (implied)

#### file i/o ####
HFconnectionMatrixAlgorithmMatrixFileName = "HFconnectionGraphAlgorithmMatrix"
HFconceptNeuronsAlgorithmMatrixFileName = "HFconceptNeuronsBasic"	#uses same file as HFconceptNeuronsBasicFileName
HFconnectionMatrixAlgorithmMatrixExtensionName = ".csv"
HFconceptNeuronsAlgorithmMatrixExtensionName = ".csv"

HFconnectionMatrixAlgorithmNormaliseSoftmax = False	#optional	#use softmax function to normalise connections matrix
if(HFconnectionMatrixAlgorithmSplit):
	HFconnectionMatrixAlgorithmNormaliseSoftmax = False	#split does not support softmax function for normalising connections matrix (must dynamically use min/max)
	HFreadSavedConnectionsMatrixAlgorithm = False	#split does not support standard matrix file i/o (only database matrix file i/o)
	HFwriteSavedConnectionsMatrixAlgorithm = False	#split does not support standard matrix file i/o (only database matrix file i/o)
	HFreadSavedConceptListAlgorithm = False		#split does not support standard matrix file i/o (only database matrix file i/o)
	HFwriteSavedConceptListAlgorithm = False	#split does not support standard matrix file i/o (only database matrix file i/o)
else:
	HFreadSavedConnectionsMatrixAlgorithm = False	#optional
	HFwriteSavedConnectionsMatrixAlgorithm = False	#optional
	HFreadSavedConceptListAlgorithm = False	#optional
	HFwriteSavedConceptListAlgorithm = False	#optional
		
useHFconnectionMatrixAlgorithmBool = False
import torch as pt
if(simulatedDendriticBranchesInitialisation):
	HFconnectionsMatrixAlgorithmType = pt.float	
else:
	if(useHFconnectionMatrixAlgorithmBool):
		HFconnectionsMatrixAlgorithmType = pt.bool
	else:
		HFconnectionsMatrixAlgorithmType = pt.long
		#print("HFconnectionsMatrixAlgorithmType = ", HFconnectionsMatrixAlgorithmType)
			
	
#### error ####
def printe(str):
	print(str)
	exit()
