"""HFNLPpy_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP - global defs

"""

printVerbose = True

#select SANIHFNLP algorithm;
useAlgorithmLayeredSANI = False
#select HFNLP algorithm;
useAlgorithmMatrix = True
useAlgorithmDendriticSANI = False	#simulate sequential activation of dendritic input 
useAlgorithmScan = False
useAlgorithmArtificial = False	#default
useAlgorithmDendriticPrototype = False	#optional	#add contextual connections to emulate primary connection spatiotemporal index restriction (visualise biological connections without simulation)
	

#### tokenise subwords ####
convertLemmasToLowercase = True #required to ensure that capitalised start of sentence words are always converted to the same lemmas (irrespective of inconsistent propernoun detection)
convertWordsToLowercase = True
if(useAlgorithmMatrix):	
	tokeniseSubwords = False	#optional
	storeConceptNodesByLemma = False	#default: False	#False: enable prediction across grammatical forms - store by word (morphology included)
else:
	tokeniseSubwords = False	
	storeConceptNodesByLemma = True	#default: True	#False: enable prediction across grammatical forms - store by word (morphology included)
if(useAlgorithmMatrix):
	from HFNLPpy_MatrixGlobalDefs import contextSizeMax
else:
	contextSizeMax =  512	#max number tokens per sentence
if(tokeniseSubwords):
	stateTrainTokeniser = False	#only required to be executed once	#should enable trainMultipleFiles with high numberOfDatafiles to perform comprehensive tokenizer train
	sequenceMaxNumTokens =  contextSizeMax	#max number tokens per sentence
	useFullwordTokenizer = False
	useFullwordTokenizerClass = True	#required #legacy config; always uses tokenizer class even with full word tokenizer
	usePreprocessedDataset = True	#required #legacy config; ensures trainTokeniserFromDataFiles
	specialTokens = ['<s>', '<pad>', '</s>', '<mask>']	#'<unk>'
	dataset4FileNameXstartTokenise = "Xdataset4Part"
	modelPathName = 'tokeniser'
	tokeniserOnlyTrainOnDictionary = False
	vocabularySize = 30522	#default: 30522	#number of independent tokens identified by HFNLPpy_dataTokeniser.trainTokeniserSubwords
	
	
#### concept connections/connectionMatrix ####

#initialise dependent vars;
linkSimilarConceptNodes = False
linkSimilarConceptNodesWordnet = False
linkSimilarConceptNodesBagOfWords = False
useHFconnectionMatrix = False
useHFconnectionMatrixPyG = False	#useHFconnectionMatrixScan
useHFconnectionMatrixBasic = False 
useHFconnectionMatrixAlgorithm = False 
tokenWordnetSynonyms = False
HFconnectionMatrixAlgorithmSplit = False
HFconnectionMatrixAlgorithmSplitDatabase = False

if(useAlgorithmDendriticSANI):
	linkSimilarConceptNodes = False	#optional
elif(useAlgorithmMatrix):
	linkSimilarConceptNodes = False	#optional

if(linkSimilarConceptNodes):
	if(not tokeniseSubwords):
		linkSimilarConceptNodesWordnet = False
	linkSimilarConceptNodesBagOfWords = True
	if(linkSimilarConceptNodesWordnet):
		tokenWordnetSynonyms = True	#requires spacy nltk:wordnet
		if(tokenWordnetSynonyms):
			tokenWordnetSynonymsFromLemma = False
	elif(linkSimilarConceptNodesBagOfWords):
		linkSimilarConceptNodesBagOfWordsWeightStore = False	#optional		#recommended - weight matrix storage calculation by distance of current sentence context word
		linkSimilarConceptNodesBagOfWordsWeightRetrieval = False	#optional	#recommended - weight matrix lookup calculation by distance of current sentence context word
		linkSimilarConceptNodesBagOfWordsContextual = True	#optional #uses incontext knowledge of previous words to find synonyms
		linkSimilarConceptNodesBagOfWordsBidirectional = False	#mandatory - !bidirectional lookup is required for next word prediction (causal) 
		if(not linkSimilarConceptNodesBagOfWordsContextual):
			linkSimilarConceptNodesBagOfWordsWeightStore = True	#required for next word prediction with !linkSimilarConceptNodesBagOfWordsContextual
			linkSimilarConceptNodesBagOfWordsWeightRetrieval = True
		linkSimilarConceptNodesBagOfWordsDistanceMax = 5 #max distance of context word
		linkSimilarConceptNodesBagOfWordsTopK = 3	#CHECKTHIS
		useHFconnectionMatrix = True
		useHFconnectionMatrixBasic = True
		useHFconnectionMatrixBasicBool = False #initialise (dependent var)
		if(not linkSimilarConceptNodesBagOfWordsWeightStore and not linkSimilarConceptNodesBagOfWordsWeightRetrieval):
			if(not useAlgorithmMatrix):	#useHFconnectionMatrixBasicBool is not supported by useAlgorithmMatrix
				useHFconnectionMatrixBasicBool = True	
			
useDependencyParseTree = False	#initialise (dependent var)
if(useAlgorithmLayeredSANI):
	useDependencyParseTree = False
else:
	if(useAlgorithmMatrix):
		useDependencyParseTree = False
	elif(useAlgorithmScan):
		useDependencyParseTree = False
	elif(useAlgorithmDendriticSANI):
		from HFNLPpy_DendriticSANIGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
		if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
			useDependencyParseTree = True
		else:
			useDependencyParseTree = False
	else:
		useDependencyParseTree = True
		biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False

if(useAlgorithmMatrix):
	useHFconnectionMatrix = True
	useHFconnectionMatrixAlgorithm = True
elif(useAlgorithmScan):
	useHFconnectionMatrix = True
	useHFconnectionMatrixPyG = True


#### file i/o ####
HFconnectionMatrixBasicFileName = "HFconnectionGraphBasic"
HFconceptNeuronsBasicFileName = "HFconceptNeuronsBasic"
HFconnectionMatriBasicExtensionName = ".csv"
HFconceptNeuronsBasicExtensionName = ".csv"

usePytorch = False
if(useHFconnectionMatrixBasic):
	HFreadSavedConnectionsMatrixBasic = False	#not available
	HFwriteSavedConnectionsMatrixBasic = False	#not available
	HFconnectionMatrixBasicNormaliseSoftmax = False	#use softmax function to normalise connections matrix
	usePytorch = True
	HFconnectionMatrixBasicMaxConcepts = 1000	#200	#1000	#default:100000	#maximum number of concepts to store	#size of HFconnectionMatrix = HFconnectionMatrixBasicMaxConcepts^2	#CHECKTHIS (should be <= number words in dic)
	HFconnectionMatrixBasicGPU = True
	HFconnectionMatrixBasicMinValue = 0
HFreadSavedConceptListBasic = False
HFwriteSavedConceptListBasic = False
		
if(usePytorch):
	import torch as pt
	from HFNLPpy_MatrixGlobalDefs import simulatedDendriticBranchesInitialisation
	if(simulatedDendriticBranchesInitialisation):
		HFconnectionsMatrixBasicType = pt.float	
	else:
		if(useHFconnectionMatrixBasicBool):
			HFconnectionsMatrixBasicType = pt.bool
		else:
			HFconnectionsMatrixBasicType = pt.long
			#print("HFconnectionsMatrixBasicType = ", HFconnectionsMatrixBasicType)
	useLovelyTensors = False
	if(useLovelyTensors):
		import lovely_tensors as lt
		lt.monkey_patch()
	else:
		pt.set_printoptions(profile="full")

if(useDependencyParseTree):
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		identifySyntacticalDependencyRelations = True	#optional
		#configuration notes:
		#some constituency parse trees are binary trees eg useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphConstituencyParserWordVectors (or Stanford constituency parser with binarize option etc), other constituency parsers are non-binary trees; eg !useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphConstituencyParserFormal (Berkeley neural parser)
		#most dependency parse trees are non-binary trees eg useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphDependencyParserWordVectors / !useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphDependencyParserWordVectors (spacy dependency parser)
		#if identifySyntacticalDependencyRelations False (use constituency parser), synapses are created in most distal branch segments only - requires dendritic tree propagation algorithm mod	
		#if supportForNonBinarySubbranchSize True, dendriticTree will support 2+ subbranches, with inputs adjusted by weight depending on number of subbranches expected to be activated
		#if supportForNonBinarySubbranchSize False, constituency/dependency parser must produce a binary parse tree (or disable biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect)
		if(not identifySyntacticalDependencyRelations):
			print("useAlgorithmDendriticSANI constituency parse tree support has not yet been implemented: synapses are created in most distal branch segments only - requires dendritic tree propagation algorithm mod")
			exit()
	else:
		identifySyntacticalDependencyRelations = True	#mandatory 	#standard hopfield NLP graph requires words are connected (no intermediary constituency parse tree syntax nodes) 

drawHopfieldGraph = True
if(useAlgorithmLayeredSANI):
	drawHopfieldGraph = False	#default: False
else:
	if(useAlgorithmMatrix):
		drawHopfieldGraph = False	#default: False
	elif(useAlgorithmScan):
		drawHopfieldGraph = False	#default: False
	elif(useAlgorithmDendriticSANI):
		drawHopfieldGraph = False	#default: False
	else:
		drawHopfieldGraph = True	#default: True
	
if(drawHopfieldGraph):
	drawHopfieldGraphPlot = True
	drawHopfieldGraphSave = False
	drawHopfieldGraphSentence = False
	drawHopfieldGraphNetwork = True	#default: True	#draw graph for entire network (not just sentence)



#### test harness (compare standard/vectorised computation) ####
from HFNLPpy_DendriticSANIGlobalDefs import biologicalSimulationTestHarness
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
enforceMinimumEncodedSequenceLength = True	#do not execute addPredictiveSequenceToNeuron if predictive sequence is short (eg does not use up the majority of numberOfBranches1)	#do not expect prediction to work if predictive sequence is short
if(enforceMinimumEncodedSequenceLength):
	minimumEncodedSequenceLength = 4	#should be high enough to fill a significant proportion of dendrite vertical branch length (numberOfBranches1)	#~seedHFnetworkSubsequenceLength


#### data loading (main) #### 

#HFNLP algorithm selection;
algorithmHFNLP = "generateHopfieldNetwork"

#debug parameters
debugUseSmallSequentialInputDataset = True


NLPsequentialInputTypeTokeniseWords = False	#perform spacy tokenization later in pipeline

NLPsequentialInputTypeMinWordVectors = True
NLPsequentialInputTypeMaxWordVectors = True
limitSentenceLengthsSize = None
limitSentenceLengths = False
NLPsequentialInputTypeTrainWordVectors = False
wordVectorLibraryNumDimensions = 300	#https://spacy.io/models/en#en_core_web_md (300 dimensions)

datasetFolderRelative = "datasets"
datasetFileNameIndexDigits = 4

trainMultipleFiles = False	#can set to true for production (after testing algorithm)
if(trainMultipleFiles):
	numberOfDatafiles = 10
else:
	numberOfDatafiles = 1
fileIndexFirst = 0
fileIndexLast = numberOfDatafiles
	
numEpochs = 1
if(numEpochs > 1):
	randomiseFileIndexParse = True
else:
	randomiseFileIndexParse = False
	
#code from ANNtf;
dataset = "wikiXmlDataset"
#if(NLPsequentialInputTypeMinWordVectors):
#	numberOfFeaturesPerWord = 1000	#used by wordToVec
paddingTagIndex = 0.0	#not used
if(debugUseSmallSequentialInputDataset):
	dataset4FileNameXstart = "Xdataset4PartSmall"
else:
	dataset4FileNameXstart = "Xdataset4Part"
xmlDatasetFileNameEnd = ".xml"

if(tokeniseSubwords):
	datasetNumberOfDataFiles = numberOfDatafiles
	trainTokenizerNumberOfFilesToUseSmall = numberOfDatafiles	#CHECKTHIS
	useSmallTokenizerTrainNumberOfFiles = False
	
def printe(str):
	print(str)
	exit()
