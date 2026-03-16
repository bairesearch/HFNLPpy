"""HFNLPpy_LayeredSANIGlobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Layered SANI Global Defs

"""

import numpy as np
from HFNLPpy_globalDefs import *

debugLowActivationThresholds = True

#### SANI word vectors ####

if(useAlgorithmMatrix):
	LayeredSANIwordVectors = False	#default:True	#SANI nodes use compound word vectors instead of compound words	#requires HFconnectionMatrixAlgorithm
else:
	LayeredSANIwordVectors = False	#mandatory
if(LayeredSANIwordVectors):
	LayeredSANIwordVectorsTopK = 1	#number of top k compound word vectors to identify	#currently required to = 1
	#currently requires algorithmMatrixTensorDim == 4
	
#### SANI hierarchy ####

if(debugLowActivationThresholds):
	LayeredSANInumberOfLayersMax = 3
else:
	LayeredSANInumberOfLayersMax = 6	#max number of SANI layers to generate
enableSkipLayerConnectivity = True	#keep lower layer associated nodes in the buffer so that they can still be used to generate new SANI nodes

	
#### topk selection ####

if(debugLowActivationThresholds):
	selectActivatedTop = True	#optional
else:
	selectActivatedTop = True	#select activated top k target neurons during propagation 	#top k selection provides functionality similar to attention/transformer layer (softmax)
if(selectActivatedTop):
	selectActivatedTopK = 1	#3	#10	
	
	
#### HF association strengths ####

if(debugLowActivationThresholds):
	if(LayeredSANIwordVectors):
		LayeredSANInodeGenerationHFassociationThresholdWordVector = 1.0	#minimum SANI/compound node association threshold with candidate compound words
	LayeredSANInodeGenerationHFassociationThreshold = 2	
else:
	if(LayeredSANIwordVectors):
		LayeredSANInodeGenerationHFassociationThresholdWordVector = 1.0	#minimum SANI/compound node association threshold with candidate compound words
	LayeredSANInodeGenerationHFassociationThreshold = 10	#minimum association strength before generating SANI node
	
useAlgorithmCausalNextWordPrediction = True	#required for useAlgorithmDendriticSANI/useAlgorithmDendriticMatrix/etc
if(useAlgorithmCausalNextWordPrediction):
	HFassociationPermutationInvariance = False	#non-contiguous SANI node input (limited enableBasicNextWordCausalPredictions implementation with t-intersection causal connections) is not currently supported by useAlgorithmDendriticSANI
else:
	HFassociationPermutationInvariance = True	#permutation invariance provides functionality similar to attention/transformer layer (k.q measurement)	#else assume wContiguityEnforced (orig SANINLPc++/SANINLPtf algorithm)
HFassociationStrengthProximityBias = True	#favour closer sentence nodes for generating SANI association (e.g. adjacent/contiguous)
if(HFassociationStrengthProximityBias):
	HFassociationStrengthProximityBiasLevel = 1.0
HFassociationStrengthAtrophy = False
if(HFassociationStrengthAtrophy):
	HFassociationStrengthAtrophy = 0.1

#initialise (dependent vars)
enableBasicNextWordCausalPredictions = False 
enableNextWordCausalPredictionsPermutationInvariance = False

if(enableSkipLayerConnectivity):
	if(useAlgorithmCausalNextWordPrediction):
		enableBasicNextWordCausalPredictions = False	#not necessary (alternate/complex next word prediction algorithm is used by useAlgorithmDendriticSANI that takes into account context)
	else:
		enableBasicNextWordCausalPredictions = True		#enableBasicNextWordCausalPredictions does not properly support HFassociationPermutationInvariance (non-contiguous SANI node input)	#limited support for HFassociationPermutationInvariance
	if(HFassociationPermutationInvariance):
		enableNextWordCausalPredictionsPermutationInvariance = True	#restrict associations of central contents of non-contiguous-input SANI node to central contents

#### computation type ####

vectoriseComputation = False	#parallel processing for optimisation	#incomplete (requires pytorch implementation for graph traversal/update)
outputFileNameComputationType = False

#### draw ####

drawBiologicalSimulation = True	#optional	#draws Layered SANI hierarchical graph
if(drawBiologicalSimulation):
	drawBiologicalSimulationPlot = True	#default: True
	drawBiologicalSimulationSave = False	#default: False	#save to file
	drawBiologicalSimulationSentence = True	#default: True	#draw graph for sentence neurons and their dendritic tree
	drawBiologicalSimulationNetwork = True	#default: False	#draw graph for entire network (not just sentence)
	LayeredSANIHierarchicalGraphOnlyShowParentChildLinks = True	#default: False	#optional	#only draw parent-child links between SANI nodes; hide same-layer HF context/causal links

highlightPartialActivations = False

#### error ####

def printe(str):
	print(str)
	exit()
