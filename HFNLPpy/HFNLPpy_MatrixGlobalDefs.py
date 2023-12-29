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

#### Propagation order ####
algorithmMatrixPropagationOrder = "propagateReverseLookup"	#select: propagateForward/propagateReverseLookup #propagateForward is required for complete sequentially activated input support (aligns with HFNLPpy_DendriticSANI:useAlgorithmDendriticSANI:!reversePropagationOrder prediction implementation)	#propagateReverseLookup (orig implementation): for each neuron in sequence; complete computation is performed for every next word prediction target neuron candidate 

#### SANI ####
algorithmMatrixSANI = True	#emulate DendriticSANIbiologicalSimulationSimple
if(algorithmMatrixSANI):
	#select algorithmMatrixSANI method (select one);
	if(algorithmMatrixPropagationOrder == "propagateForward"):
		algorithmMatrixSANImethod = "completeSANI"	
		#algorithmMatrixSANImethod = "posthocSANI"	#depreciated #for debug only (propagateReverseLookup:posthocSANI implementation emulation)
	else:
		algorithmMatrixSANImethod = "posthocSANI"	#mandatory
	if(algorithmMatrixSANImethod == "completeSANI"):	#incomplete
		#increase input activation based on presense and proximity of prior segment activations
		algorithmMatrixSANImethodPosthoc = "getLastSequentialSegmentActivation"	#default for completeSANI
		#algorithmMatrixSANImethodPosthoc = "addActivationAcrossSegments"	#for debug only (posthocSANI implementation emulation)
		#activationRepolarisationTime = 1	#calibrate	#in number of sequential segments (propagation distance)
		#activationPropagationTimeMax = 3	#max propagation time between sequential segments
		activationDecayType = "linear"	#select: linear/exponential	#activation decay along segments
	else:
		algorithmMatrixSANImethodPosthoc = "addActivationAcrossSegments"	#default	#it is not necessary that previous segments be activated, but their activation will bias the selection of a particular dendritic branch
		#algorithmMatrixSANImethodPosthoc = "supportSequentialActivationAcrossSegments"	#incomplete	#retain previously unactivated context, to be fed into next segment
		#algorithmMatrixSANImethodPosthoc = "enforceSequentialActivationAcrossSegments"	#incomplete		#previous segments must be activated for current segment to be activated
	#select sequentialSegmentContextEncoding method (select one);
	sequentialSegmentContextEncoding = "linear"	#select: linear/relativeLinear/relativeExponential #relativeExponential: sequential segments capture input (past context tokens) at exponentially further distances
	if(sequentialSegmentContextEncoding=="linear"):
		sequentialSegmentContextEncodingSize = 1	#1	#lower value: engram (prediction) more robust but less generalisable	#number of tokens per segment
		sequentialSegmentContextEncodingMaxLength = 30	#maximum length of engram across all sequential segments
	#sequentialSegmentContextEncodingRandom = False #encodings are partially randomised (note dendritic SANI implementation creates multiple semi-random encodings of past context in different dendritic branches) 
	if(algorithmMatrixSANImethodPosthoc=="supportSequentialActivationAcrossSegments"):
		numberOfBranchSequentialSegments = 10	#SANI supports higher resolution sequential segments (up to max sequence resolution; x=1 [individual tokens])
	elif(algorithmMatrixSANImethodPosthoc=="addActivationAcrossSegments"):
		if(sequentialSegmentContextEncoding=="linear"):
			numberOfBranchSequentialSegments = sequentialSegmentContextEncodingMaxLength//sequentialSegmentContextEncodingSize
		elif(sequentialSegmentContextEncoding=="relativeLinear"):
			numberOfBranchSequentialSegments = 5
		elif(sequentialSegmentContextEncoding=="relativeExponential"):
			numberOfBranchSequentialSegments = 5
	elif(algorithmMatrixSANImethodPosthoc=="getLastSequentialSegmentActivation"):
		numberOfBranchSequentialSegments = 5
	normaliseConnectionStrengthWrtContextLength = False
else:
	algorithmMatrixSANImethodPosthoc = "none"
	algorithmMatrixSANImethodTopkActivationAcrossSegments = False
	algorithmMatrixSANImethodAddActivationAcrossSegmentsOld = False
	normaliseConnectionStrengthWrtContextLength = True	#optional	#orig:True	#CHECKTHIS

#### memory constraints ####
#algorithmMatrixTensorDim = 2	#default
algorithmMatrixTensorDim = 4	#optional	#store context size array (and simulated dendritic branches) in pytorch tensor rather than python list	#requires high ram
if(algorithmMatrixTensorDim != 4):
	if(algorithmMatrixSANImethodPosthoc=="addActivationAcrossSegments" or algorithmMatrixSANImethodPosthoc=="supportSequentialActivationAcrossSegments"):
		algorithmMatrixTensorDim = 3 #mandatory

HFconnectionMatrixAlgorithmSparse = False		#incomplete: requires hybrid dense-sparse tensor implementation such that tensor can still be indexed #reduces ram requirements (especially for large HFconnectionMatrixBasicMaxConcepts)
HFconnectionMatrixAlgorithmSplit = True		#store each column of connection matrix in separate array (or hard disk file) 	#currently in testing #reduces ram requirements (especially for large HFconnectionMatrixBasicMaxConcepts)	#store each column of connection matrix in separate array (or hard drive file) - bring into memory on demand (depending on the precise activated columns of the contextVector being compared)

if(algorithmMatrixSANImethod == "completeSANI"):
	assert (algorithmMatrixTensorDim == 4)	#algorithmMatrixSANImethod==completeSANI currently requires algorithmMatrixTensorDim=4 such that the code can be simplified

if(HFconnectionMatrixAlgorithmSplit):
	HFconnectionMatrixAlgorithmSplitDatabase = False	#optional	#store each column of connection matrix in separate hard drive file and bring into RAM at start of sentence
	HFconnectionMatrixAlgorithmContextVectorSparse = True	#store context vector as sparse tensor - only index context vector indices of connection matrix (computation or memory efficient)
	HFconnectionMatrixAlgorithmGPU = True	#store connection matrix and context vector in GPU
	HFconnectionMatrixAlgorithmNormaliseStore = False	#False: only store min/max of each target (row) word, for dynamic normalisation (on demand)
	if(HFconnectionMatrixAlgorithmSplitDatabase):
		assert (algorithmMatrixTensorDim == 4)	#HFconnectionMatrixAlgorithmSplitDatabase currently requires algorithmMatrixTensorDim=4 such that the file i/o code can be simplified
		matrixDatabaseFileNameStart = "HFmatrixDatabaseSourceNeuronID"
		matrixDatabaseFileNameEnd = ".csv"
		matrixDatabasePathName = "database"
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

#### max database size selection ####
if(debugHFconnectionMatrix):
	if(HFconnectionMatrixAlgorithmSplit):
		HFconnectionMatrixBasicMaxConcepts = 200	#200	#20	 #[Xdataset4PartSmall0000.xml.verifyOldSentenceSomaActivationFound0]
	HFconnectionMatrixBasicMaxConceptsInArticle = 200
else:
	from HFNLPpy_globalDefs import debugUseSmallSequentialInputDataset
	if(debugUseSmallSequentialInputDataset):
		if(HFconnectionMatrixAlgorithmSplit):
			HFconnectionMatrixBasicMaxConcepts = 10000	#default:100000	#maximum number of concepts to store	#size of HFconnectionMatrix = HFconnectionMatrixBasicMaxConcepts^2	#CHECKTHIS (should be <= number words in dic)
		HFconnectionMatrixBasicMaxConceptsInArticle = 10000
	else:
		if(HFconnectionMatrixAlgorithmSplitDatabase):
			if(HFconnectionMatrixAlgorithmSplit):
				HFconnectionMatrixBasicMaxConcepts = 100000	#default:100000	#maximum number of concepts to store	#size of HFconnectionMatrix = HFconnectionMatrixBasicMaxConcepts^2	#CHECKTHIS (should be <= number words in dic)
			HFconnectionMatrixBasicMaxConceptsInArticle = 100000
		else:
			print("error: !debugUseSmallSequentialInputDataset requires HFconnectionMatrixAlgorithmSplitDatabase")
	
	
#### topk selection ####
selectActivatedTop = True	#mandatory (implied) select activated top k target neurons during propagation test
if(selectActivatedTop):
	if(algorithmMatrixTensorDim==4):
		matrixPropagateTopCommonSegmentPredictions = False	#optional	#for each sequential segment, calculate top x (eg 10) predictions - take the predictions that are common across all segments and highest weighted
	else:
		matrixPropagateTopCommonSegmentPredictions = False #not yet implemented
	matrixPropagateTopKcontextSize = 1	#number of top k elements to save
	if(matrixPropagateTopCommonSegmentPredictions):
		matrixPropagateTopCommonSegmentPredictionsVectorised = False	#requires more GPU memory (*matrixPropagateTopKconceptNodes)
		matrixPropagateTopKconceptNodes = 10	#100	#number of top k elements to save
		matrixPropagateTopKsequentialSegments = 1	#number of top k elements to save
		matrixPropagateTopCommonSegmentPredictionsRequired = numberOfBranchSequentialSegments-0	#default: numberOfBranchSequentialSegments
	else:
		matrixPropagateTopKconceptNodes = 1	#number of top k elements to save
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

HFconnectionMatrixAlgorithmNormalise = "linear"
if(HFconnectionMatrixAlgorithmSplit):
	HFconnectionMatrixAlgorithmNormalise = "tanh"	#"tanh"	#select: linear/tanh	#split does not support softmax function for normalising connections matrix (must dynamically use min/max; ie "linear")
	HFreadSavedConnectionsMatrixAlgorithm = False	#split does not support standard matrix file i/o (only database matrix file i/o)
	HFwriteSavedConnectionsMatrixAlgorithm = False	#split does not support standard matrix file i/o (only database matrix file i/o)
	if(HFconnectionMatrixAlgorithmSplitDatabase):
		HFreadSavedConceptList = True
		HFwriteSavedConceptList = True
	else:
		HFreadSavedConceptList = False		#split does not support standard matrix file i/o (only database matrix file i/o)
		HFwriteSavedConceptList = False	#split does not support standard matrix file i/o (only database matrix file i/o)
else:
	HFconnectionMatrixAlgorithmNormalise = "tanh"	#"tanh"	#select: linear/softmax/tanh
	HFreadSavedConnectionsMatrixAlgorithm = False	#optional
	HFwriteSavedConnectionsMatrixAlgorithm = False	#optional
	HFreadSavedConceptList = False	#optional
	HFwriteSavedConceptList = False	#optional
				
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
	exite

def getHFconnectionMatrixBasicMaxConcepts(HFconnectionGraphObject):
	return HFconnectionGraphObject.connectionMatrixMaxConcepts
		
