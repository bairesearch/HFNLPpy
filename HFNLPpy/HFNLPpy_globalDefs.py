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
useAlgorithmLayeredSANIbiologicalSimulation = False
useAlgorithmDendriticSANIbiologicalSimulation = True	#simulate sequential activation of dendritic input 
useAlgorithmScanBiologicalSimulation = False
useAlgorithmArtificial = False	#default
useAlgorithmDendriticSANIbiologicalPrototype = False	#optional	#add contextual connections to emulate primary connection spatiotemporal index restriction (visualise biological connections without simulation)

tokenWordnetSynonyms = True	#requires spacy nltk:wordnet
if(tokenWordnetSynonyms):
	tokenWordnetSynonymsFromLemma = False
	
useDependencyParseTree = False	#initialise (dependent var)
if(useAlgorithmLayeredSANIbiologicalSimulation):
	useDependencyParseTree = False
elif(useAlgorithmScanBiologicalSimulation):
	useDependencyParseTree = False
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
