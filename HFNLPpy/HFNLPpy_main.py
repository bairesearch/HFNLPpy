"""HFNLPpy_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n anntf2 python=3.7
source activate anntf2
pip install tensorflow [required for HFNLPpy_DendriticSANIPropagateVectorised]
pip install networkx [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_DendriticSANIDraw]
pip install matplotlib==2.2.3 [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_DendriticSANIDraw]
pip install yattag [required for HFNLPpy_DendriticSANIXML]
pip install torch [required for HFNLPpy_Scan/SANIHFNLPpy_LayeredSANI:vectoriseComputation]
pip install torch_geometric [required for HFNLPpy_Scan]
pip install nltk spacy==2.3.7
python3 -m spacy download en_core_web_md
pip install benepar [required for SPNLPpy_syntacticalGraphConstituencyParserFormal]

# Usage:
source activate anntf2
python3 HFNLPpy_main.py

# Description:
HFNLP - hopfield natural language processing


"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

import random
import ANNtf2_loadDataset
import HFNLPpy_hopfieldGraph


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

trainMultipleFiles = False	#can set to true for production (after testing algorithm)
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
def loadDataset(fileIndex, textualDatasetLoadPerformProcessing=True):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameXstart + fileIndexStr + xmlDatasetFileNameEnd

	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY, addOnlyPriorUnidirectionalPOSinputToTrain)
		if(trainDataIncludesSentenceOutOfBoundsIndex):
			datasetNumClasses = datasetNumClasses + 1
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, limitSentenceLengthsSize)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, limitSentenceLengths, limitSentenceLengthsSize, NLPsequentialInputTypeTrainWordVectors, NLPsequentialInputTypeTokeniseWords)
		if(textualDatasetLoadPerformProcessing):
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.convertArticlesTreeToSentencesWordVectors(articles, limitSentenceLengthsSize)

	if((dataset == "wikiXmlDataset") and not textualDatasetLoadPerformProcessing):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y


def trainSequentialInput(trainMultipleFiles=False):
	
	#if(algorithmHFNLP == "generateHopfieldNetwork"):
	#	HFNLPpy_hopfieldNetwork.constructPOSdictionary()	#required for HFNLPpy_hopfieldNetwork:HFNLPtf_getAllPossiblePosTags.getAllPossiblePosTags(word)
					
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False

	#configure optional parameters;
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = ANNtf2_loadDataset.generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f

			#NLP specific code;
							
			articles = loadDataset(fileIndex, textualDatasetLoadPerformProcessing=False)	#do not perform processing of textual dataset during load (word vector extraction)
								
			#print("articles = ", articles)
			#print("listDimensions(articles) = ", listDimensions(articles))

			processingSimple(articles)
					
						
def processingSimple(articles):
	if(NLPsequentialInputTypeMaxWordVectors):
		#flatten any higher level abstractions defined in NLPsequentialInputTypeMax down to word vector lists (sentences);
		articles = ANNtf2_loadDataset.flattenNestedListToSentences(articles)

	if(algorithmHFNLP == "generateHopfieldNetwork"):
		articles = HFNLPpy_hopfieldGraph.generateHopfieldGraphNetwork(articles)
	

if __name__ == "__main__":
	trainSequentialInput(trainMultipleFiles=trainMultipleFiles)

