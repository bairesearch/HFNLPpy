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
ScanBiologicalSimulation = True
SANIbiologicalSimulation = False	#simulate sequential activation of dendritic input 
#useAlgorithmArtificial = False	#default

SANIbiologicalPrototype = False	#optional	#add contextual connections to emulate primary connection spatiotemporal index restriction (visualise biological connections without simulation)

useDependencyParseTree = False	#initialise (dependent var)
if(ScanBiologicalSimulation):
	useDependencyParseTree = False
elif(SANIbiologicalSimulation):
	from HFNLPpy_SANIGlobalDefs import biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		useDependencyParseTree = True
	else:
		useDependencyParseTree = False
else:
	useDependencyParseTree = True
		
if(useDependencyParseTree):
	import SPNLPpy_globalDefs
	if(not SPNLPpy_globalDefs.useSPNLPcustomSyntacticalParser):
		SPNLPpy_syntacticalGraph.SPNLPpy_syntacticalGraphConstituencyParserFormal.initalise(spacyWordVectorGenerator)
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
		identifySyntacticalDependencyRelations = True	#optional
		#configuration notes:
		#some constituency parse trees are binary trees eg useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphConstituencyParserWordVectors (or Stanford constituency parser with binarize option etc), other constituency parsers are non-binary trees; eg !useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphConstituencyParserFormal (Berkeley neural parser)
		#most dependency parse trees are non-binary trees eg useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphDependencyParserWordVectors / !useSPNLPcustomSyntacticalParser:SPNLPpy_syntacticalGraphDependencyParserWordVectors (spacy dependency parser)
		#if identifySyntacticalDependencyRelations False (use constituency parser), synapses are created in most distal branch segments only - requires dendritic tree propagation algorithm mod	
		#if supportForNonBinarySubbranchSize True, dendriticTree will support 2+ subbranches, with inputs adjusted by weight depending on number of subbranches expected to be activated
		#if supportForNonBinarySubbranchSize False, constituency/dependency parser must produce a binary parse tree (or disable biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect)
		if(not identifySyntacticalDependencyRelations):
			print("SANIbiologicalSimulation constituency parse tree support has not yet been implemented: synapses are created in most distal branch segments only - requires dendritic tree propagation algorithm mod")
			exit()
	else:
		identifySyntacticalDependencyRelations = True	#mandatory 	#standard hopfield NLP graph requires words are connected (no intermediary constituency parse tree syntax nodes) 

drawHopfieldGraph = False
if(ScanBiologicalSimulation):
	drawHopfieldGraph = False	#default: False - typically use drawBiologicalSimulation only
elif(SANIbiologicalSimulation):
	drawHopfieldGraph = False	#default: False - typically use drawBiologicalSimulation only
else:
	drawHopfieldGraph = True
	
if(drawHopfieldGraph):
	drawHopfieldGraphPlot = True
	drawHopfieldGraphSave = False
	drawHopfieldGraphSentence = False
	drawHopfieldGraphNetwork = True	#default: True	#draw graph for entire network (not just sentence)

def printe(str):
	print(str)
	exit()
