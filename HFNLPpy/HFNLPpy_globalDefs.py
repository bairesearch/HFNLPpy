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
	storeConceptNodesByLemma = False	#enable prediction across grammatical forms #False: store by word (morphology included)
else:
	tokeniseSubwords = False	
	storeConceptNodesByLemma = True	#False: store by word (morphology included)
if(tokeniseSubwords):
	stateTrainTokeniser = False	#only required to be executed once	#should enable trainMultipleFiles with high numberOfDatafiles to perform comprehensive tokenizer train
	if(useAlgorithmMatrix):
		from HFNLPpy_MatrixGlobalDefs import contextSizeMax
		sequenceMaxNumTokens = contextSizeMax	#max number tokens per sentence
	else:
		sequenceMaxNumTokens =  512	#max number tokens per sentence
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
useHFconnectionMatrixPyG = False
useHFconnectionMatrixBasic = False 
tokenWordnetSynonyms = False

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
		if(not bagOfWordsWeightStore and not linkSimilarConceptNodesBagOfWordsWeightRetrieval):
			assert (not useAlgorithmMatrix)	#useHFconnectionMatrixBasicBool is not supported by useAlgorithmMatrix
			useHFconnectionMatrixBasicBool = True	
			
useDependencyParseTree = False	#initialise (dependent var)
if(useAlgorithmLayeredSANI):
	useDependencyParseTree = False
elif(useAlgorithmMatrix):
	useHFconnectionMatrix = True
	useHFconnectionMatrixBasic = True
	useHFconnectionMatrixBasicBool = False
elif(useAlgorithmScan):
	useDependencyParseTree = False
	useHFconnectionMatrix = True
	useHFconnectionMatrixPyG = True
elif(useAlgorithmDendriticSANI):
	from HFNLPpy_DendriticSANIGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		useDependencyParseTree = True
	else:
		useDependencyParseTree = False
else:
	useDependencyParseTree = True
	biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False

usePytorch = False
if(useHFconnectionMatrix):
	HFreadSavedConnectionsMatrixPyG = False	#currently requires useAlgorithmScan
	HFreadSavedConnectionsMatrixBasic = False	#not available
	HFwriteSavedConnectionsMatrixPyG = False	#currently requires useAlgorithmScan
	HFwriteSavedConnectionsMatrixBasic = False	#not available
	useHFconnectionMatrixNormaliseSoftmax = False	#use softmax function to normalise connections matrix
	usePytorch = True
	if(useAlgorithmMatrix):
		from HFNLPpy_MatrixGlobalDefs import useHFconnectionMatrixBasicSparse
		from HFNLPpy_MatrixGlobalDefs import HFconnectionMatrixBasicMaxConcepts
	else:
		useHFconnectionMatrixBasicSparse = False
		HFconnectionMatrixBasicMaxConcepts = 1000	#200	#1000	#default:100000	#maximum number of concepts to store	#size of HFconnectionMatrix = HFconnectionMatrixBasicMaxConcepts^2	#CHECKTHIS (should be <= number words in dic)
	
if(usePytorch):
	import torch as pt
	from HFNLPpy_MatrixGlobalDefs import simulatedDendriticBranchesInitialisation
	if(simulatedDendriticBranchesInitialisation):
		HFconnectionsMatrixType = pt.float	
	else:
		if(useHFconnectionMatrixBasicBool):
			HFconnectionsMatrixType = pt.bool
		else:
			HFconnectionsMatrixType = pt.long
			#print("HFconnectionsMatrixType = ", HFconnectionsMatrixType)
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
elif(useAlgorithmMatrix):
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


#initialise (dependent var)
seedHFnetworkSubsequence = False
HFNLPnonrandomSeed = False


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
