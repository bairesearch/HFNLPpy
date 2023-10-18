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

#select HFNLP algorithm;
useAlgorithmLayeredSANIbiologicalSimulation = True
useAlgorithmDendriticSANIbiologicalSimulation = True	#simulate sequential activation of dendritic input 
useAlgorithmScanBiologicalSimulation = False
useAlgorithmArtificial = False	#default
useAlgorithmDendriticSANIbiologicalPrototype = False	#optional	#add contextual connections to emulate primary connection spatiotemporal index restriction (visualise biological connections without simulation)

#### concept connections ####
linkSimilarConceptNodes = False #initialise (dependent var)
linkSimilarConceptNodesWordnet = False #initialise (dependent var)
linkSimilarConceptNodesBagOfWords = False #initialise (dependent var)
useHFconnectionMatrix = False
useHFconnectionMatrixPyG = False	#initialise (dependent var)
useHFconnectionMatrixBasic = False 	#initialise (dependent var)
tokenWordnetSynonyms = False 	#initialise (dependent var)
if(useAlgorithmDendriticSANIbiologicalSimulation):
	linkSimilarConceptNodes = False
	if(linkSimilarConceptNodes):
		linkSimilarConceptNodesWordnet = False
		linkSimilarConceptNodesBagOfWords = True
		if(linkSimilarConceptNodesWordnet):
			tokenWordnetSynonyms = True	#requires spacy nltk:wordnet
			if(tokenWordnetSynonyms):
				tokenWordnetSynonymsFromLemma = False
		elif(linkSimilarConceptNodesBagOfWords):
			linkSimilarConceptNodesBagOfWordsWeightStore = True	#optional	#required for next word prediction (ie !linkSimilarConceptNodesBagOfWordsFullContextAvailableUponRetrieval)
			linkSimilarConceptNodesBagOfWordsWeightRetrieval = True	#optional	#recommended - weight matrix lookup calculation by distance of current sentence context word
			linkSimilarConceptNodesBagOfWordsFullContextAvailableUponRetrieval = False	#has not yet been coded #retrieveSimilarConcepts:linkSimilarConceptNodesBagOfWordsFullContextAvailableUponRetrieval requires sentenceConceptNodeList and w	#requires requires incontext knowledge of previous and next words to find synonyms
			useHFconnectionMatrix = True
			useHFconnectionMatrixBasic = True
			useHFconnectionMatrixBasicBool = False #initialise (dependent var)
			if(not linkSimilarConceptNodesBagOfWordsWeightStore and not linkSimilarConceptNodesBagOfWordsWeightRetrieval):
				useHFconnectionMatrixBasicBool = True
			linkSimilarConceptNodesBagOfWordsDistanceMax = 5 #max distance of context word
			linkSimilarConceptNodesBagOfWordsTopK = 3	#CHECKTHIS
if(useHFconnectionMatrix):
	HFreadSavedConnectionsMatrixPyG = False
	HFreadSavedConnectionsMatrixBasic = False
	HFconnectionMatrixBasicMaxConcepts = 1000	#size of HFconnectionMatrix [^2]	#maximum number of concepts to store
	
useDependencyParseTree = False	#initialise (dependent var)
if(useAlgorithmLayeredSANIbiologicalSimulation):
	useDependencyParseTree = False
elif(useAlgorithmScanBiologicalSimulation):
	useDependencyParseTree = False
	useHFconnectionMatrix = True
	useHFconnectionMatrixPyG = True
elif(useAlgorithmDendriticSANIbiologicalSimulation):
	from HFNLPpy_DendriticSANIGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		useDependencyParseTree = True
	else:
		useDependencyParseTree = False
else:
	useDependencyParseTree = True
	biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False
	
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
			print("useAlgorithmDendriticSANIbiologicalSimulation constituency parse tree support has not yet been implemented: synapses are created in most distal branch segments only - requires dendritic tree propagation algorithm mod")
			exit()
	else:
		identifySyntacticalDependencyRelations = True	#mandatory 	#standard hopfield NLP graph requires words are connected (no intermediary constituency parse tree syntax nodes) 

drawHopfieldGraph = False
if(useAlgorithmLayeredSANIbiologicalSimulation):
	drawHopfieldGraph = False
elif(useAlgorithmScanBiologicalSimulation):
	drawHopfieldGraph = False	#default: False - typically use drawBiologicalSimulation only
elif(useAlgorithmDendriticSANIbiologicalSimulation):
	drawHopfieldGraph = False	#default: False - typically use drawBiologicalSimulation only
else:
	drawHopfieldGraph = True
	
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
