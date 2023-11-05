"""HFNLPpy_ConceptsMatrix.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Concepts Matrix

"""
import numpy as np
import torch as pt
from HFNLPpy_globalDefs import *

import HFNLPpy_ConnectionMatrixBasic
import HFNLPpy_ConceptsMatrixOperations

def initialiseHFconnectionMatrixBasicWrapper(HFconnectionGraphObject):
	if(linkSimilarConceptNodesBagOfWords):
		HFconnectionGraphObject.HFconnectionGraphBasic = HFNLPpy_ConnectionMatrixBasic.initialiseHFconnectionMatrixBasic()
	HFNLPpy_ConnectionMatrixBasic.initialiseNeuronNameList(HFconnectionGraphObject, HFreadSavedConceptListBasic)

def addContextWordsToConnectionGraphLinkConcepts(tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject):
	for w1, token1 in enumerate(tokenisedSentence):
		conceptNode = sentenceConceptNodeList[w1]
		neuronID = HFconnectionGraphObject.neuronIDdict[conceptNode.nodeName]
		HFconnectionGraphObject.HFconnectionGraphBasic, HFconnectionGraphObject.HFconnectionGraphBasicNormalised = addContextWordsToConnectionGraphBasic(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraphObject.HFconnectionGraphBasic, linkSimilarConceptNodesBagOfWordsDistanceMax, linkSimilarConceptNodesBagOfWordsWeightStore, linkSimilarConceptNodesBagOfWordsBidirectional)

def addContextWordsToConnectionGraphBasic(w1, neuronID, tokenisedSentence, sentenceConceptNodeList, HFconnectionGraphObject, HFconnectionGraph, contextSizeIndex, weightStore, bidirectionalContext):
	contextConnectionVector = HFNLPpy_ConceptsMatrixOperations.createContextVectorBasic(w1, sentenceConceptNodeList, HFconnectionGraphObject, contextSizeIndex, weightStore, bidirectionalContext)
	HFconnectionGraph[neuronID] = HFNLPpy_ConnectionMatrixBasic.addContextConnectionsToGraphBasic(HFconnectionGraph[neuronID], contextConnectionVector)
	HFconnectionGraphNormalised = HFNLPpy_ConnectionMatrixBasic.normaliseBatchedTensor(HFconnectionGraph)
	return HFconnectionGraph, HFconnectionGraphNormalised

def retrieveSimilarConcepts(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject=None):
	if(linkSimilarConceptNodesWordnet):
		for conceptNeuron in connectionTargetNeuronSet:
			connectionTargetNeuronSetExtended.append(conceptNeuron)
			for synonym in conceptNeuron.synonymsList:
				synonymConcept, conceptInDict = convertLemmaToConcept(networkConceptNodeDict, synonym)
				if(conceptInDict):
					#print("conceptInDict: ", synonymConcept.nodeName)
					connectionTargetNeuronSetExtended.append(synonymConcept)
	elif(linkSimilarConceptNodesBagOfWords):
		connectionTargetNeuronSetExtended = HFNLPpy_ConceptsMatrixOperations.retrieveSimilarConceptsBagOfWords(wSource, sentenceConceptNodeList, networkConceptNodeDict, connectionTargetNeuronSet, HFconnectionGraphObject)

	return connectionTargetNeuronSetExtended
