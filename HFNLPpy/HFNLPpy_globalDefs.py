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
useAlgorithmMatrix = False
useAlgorithmDendriticSANI = True	#simulate sequential activation of dendritic input 
useAlgorithmScan = False
useAlgorithmArtificial = False	#default
useAlgorithmDendriticPrototype = False	#optional	#add contextual connections to emulate primary connection spatiotemporal index restriction (visualise biological connections without simulation)

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

if(useHFconnectionMatrix):
	HFreadSavedConnectionsMatrixPyG = False	#currently requires useAlgorithmScan
	HFreadSavedConnectionsMatrixBasic = False	#not available
	HFwriteSavedConnectionsMatrixPyG = False	#currently requires useAlgorithmScan
	HFwriteSavedConnectionsMatrixBasic = False	#not available
	HFconnectionMatrixBasicMaxConcepts = 1000	#maximum number of concepts to store	#size of HFconnectionMatrix = HFconnectionMatrixBasicMaxConcepts^2	#CHECKTHIS (should be <= number words in dic)
	useHFconnectionMatrixNormaliseSoftmax = False	#use softmax function to normalise connections matrix
	import torch as pt
	if(useHFconnectionMatrixBasicBool):
		HFconnectionsMatrixType = pt.bool
	else:
		HFconnectionsMatrixType = pt.long
		#print("HFconnectionsMatrixType = ", HFconnectionsMatrixType)

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

def printe(str):
	print(str)
	exit()
