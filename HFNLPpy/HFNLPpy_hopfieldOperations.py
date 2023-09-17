"""HFNLPpy_hopfieldOperations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Hopfield Operations

Contains shared HFNLP operations on HFNLPpy_hopfieldNodeClass/HFNLPpy_hopfieldConnectionClass

"""

import numpy as np
from HFNLPpy_globalDefs import *
from HFNLPpy_hopfieldNodeClass import *
from HFNLPpy_hopfieldConnectionClass import *
from nltk.corpus import wordnet

def addConnectionToNode(nodeSource, nodeTarget, activationTime=-1, spatioTemporalIndex=-1, useAlgorithmDendriticSANIbiologicalPrototype=False, weight=1.0, subsequenceConnection=False, contextConnection=False, contextConnectionSANIindex=0, useAlgorithmDendriticSANIbiologicalSimulation=False, nodeTargetSequentialSegmentInput=None):
	connection = HopfieldConnection(nodeSource, nodeTarget, spatioTemporalIndex, activationTime, useAlgorithmDendriticSANIbiologicalPrototype)
	connection.contextConnection = contextConnection
	
	if(contextConnection):
		nodeSource.HFcontextTargetConnectionDict[nodeTarget.nodeName] = connection
		nodeTarget.HFcontextSourceConnectionDict[nodeSource.nodeName] = connection
		if(useAlgorithmLayeredSANIbiologicalSimulation):
			nodeSource.HFcontextTargetConnectionLayeredDict[nodeTarget.SANIlayerNeuronID] = connection
			nodeTarget.HFcontextSourceConnectionLayeredDict[nodeSource.SANIlayerNeuronID] = connection
		if(useAlgorithmDendriticSANIbiologicalSimulation):
			createConnectionKeyIfNonExistant(nodeSource.HFcontextTargetConnectionMultiDict, nodeTarget.nodeName)
			createConnectionKeyIfNonExistant(nodeTarget.HFcontextSourceConnectionMultiDict, nodeSource.nodeName)
			nodeSource.HFcontextTargetConnectionMultiDict[nodeTarget.nodeName].append(connection)
			nodeTarget.HFcontextSourceConnectionMultiDict[nodeSource.nodeName].append(connection)
			#connection.subsequenceConnection = subsequenceConnection
	else:
		nodeSource.HFcausalTargetConnectionDict[nodeTarget.nodeName] = connection
		nodeTarget.HFcausalSourceConnectionDict[nodeSource.nodeName] = connection
		if(useAlgorithmLayeredSANIbiologicalSimulation):
			nodeSource.HFcausalTargetConnectionLayeredDict[nodeTarget.SANIlayerNeuronID] = connection
			nodeTarget.HFcausalSourceConnectionLayeredDict[nodeSource.SANIlayerNeuronID] = connection
		if(useAlgorithmDendriticSANIbiologicalSimulation):
			createConnectionKeyIfNonExistant(nodeSource.HFcausalTargetConnectionMultiDict, nodeTarget.nodeName)
			createConnectionKeyIfNonExistant(nodeTarget.HFcausalSourceConnectionMultiDict, nodeSource.nodeName)
			nodeSource.HFcausalTargetConnectionMultiDict[nodeTarget.nodeName].append(connection)
			nodeTarget.HFcausalSourceConnectionMultiDict[nodeSource.nodeName].append(connection)
			#connection.subsequenceConnection = subsequenceConnection
		
	if(useAlgorithmDendriticSANIbiologicalPrototype):
		connection.useAlgorithmDendriticSANIbiologicalPrototype = useAlgorithmDendriticSANIbiologicalPrototype
		connection.weight = weight
		connection.contextConnectionSANIindex = contextConnectionSANIindex
	if(useAlgorithmDendriticSANIbiologicalSimulation):
		connection.useAlgorithmDendriticSANIbiologicalSimulation = useAlgorithmDendriticSANIbiologicalSimulation
		connection.nodeTargetSequentialSegmentInput = nodeTargetSequentialSegmentInput
		connection.weight = weight
	return connection

def retrieveSimilarConcepts(networkConceptNodeDict, connectionTargetNeuronSet):
	connectionTargetNeuronSetExtended = []
	for conceptNeuron in connectionTargetNeuronSet:
		connectionTargetNeuronSetExtended.append(conceptNeuron)
		for synonym in conceptNeuron.synonymsList:
			synonymConcept, conceptInDict = convertLemmaToConcept(networkConceptNodeDict, synonym)
			if(conceptInDict):
				#print("conceptInDict: ", synonymConcept.nodeName)
				connectionTargetNeuronSetExtended.append(synonymConcept)
	return connectionTargetNeuronSetExtended
		
def convertLemmaToConcept(networkConceptNodeDict, synonym):
	synonymConcept = None
	conceptInDict = False
	if(synonym in networkConceptNodeDict):
		synonymConcept = networkConceptNodeDict[synonym]
		conceptInDict = True
	return synonymConcept, conceptInDict
	
def getTokenSynonyms(token):
	synonymsList = []

	# Use spaCy to get the part of speech (POS) of the word
	pos = token.pos_

	# Map spaCy POS tags to WordNet POS tags
	if pos.startswith("N"):
		pos_tag = wordnet.NOUN
	elif pos.startswith("V"):
		pos_tag = wordnet.VERB
	elif pos.startswith("R"):
		pos_tag = wordnet.ADV
	elif pos.startswith("J"):
		pos_tag = wordnet.ADJ
	else:
		pos_tag = wordnet.NOUN  # Default to noun if POS is unknown

	# Get synsets for the word using WordNet
	synsets = wordnet.synsets(token.text, pos=pos_tag)

	# Extract synonyms from synsets
	for synset in synsets:
		synonymsList.extend(synset.lemma_names())
 
	if(token.text in synonymsList):
		#print("token.text itself is in synonymsList")
		synonymsList.remove(token.text)
		#index = animals.index(token.text)
		
	#print("synonymsList = ", synonymsList)
	
	return synonymsList



