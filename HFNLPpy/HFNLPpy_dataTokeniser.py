"""HFNLPpy_dataTokeniser.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLPpy data tokeniser

"""

import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizer	#CHECKTHIS (requires generalisation) 
from HFNLPpy_globalDefs import *
if(useFullwordTokenizer):	#not supported
	import HFNLPpt_dataTokeniserFullword
if(tokeniserOnlyTrainOnDictionary):	#not supported
	from nltk.corpus import words

def initialiseTokeniser():
	if(tokeniseSubwords):
		dataElements = prepareDataElementsWikiXmlDataset()
		if(stateTrainTokeniser):
			tokenizer = trainTokeniser(dataElements, vocabularySize)
		else:
			tokenizer = loadTokeniser()
	else:
		tokenizer = None
	return tokenizer

def prepareDataElementsWikiXmlDataset():
	paths = []
	for fileIndex in range(fileIndexFirst, fileIndexLast):
		fileIndexStr = str(fileIndex).zfill(datasetFileNameIndexDigits)
		datasetType4FileName = datasetFolderRelative + "/" + dataset4FileNameXstartTokenise + fileIndexStr + xmlDatasetFileNameEnd
		paths.append(datasetType4FileName)
	dataElements = paths
	return dataElements
		
def tokenise(lines, tokenizer, maxLength):
	if(useFullwordTokenizerClass):
		if(maxLength is None):
			sample = tokenizer(lines, return_tensors='pt')
		else:
			sample = tokenizer(lines, max_length=maxLength, padding='max_length', truncation=True, return_tensors='pt')
	else:
		sample = SBNLPpt_dataTokeniserFullword.tokenizeBasic(lines, tokenizer)
	return sample

def trainTokeniser(dataElements, vocabSize):	
	if(useFullwordTokenizer):
		SBNLPpt_dataTokeniserFullword.trainTokenizerFullwords(dataElements, vocabularySize)	#default method (vocabSize used by GIA word2vec model will be greater than numberOfTokens in tokenizer)
	else:
		tokenizer = trainTokeniserSubwords(dataElements, vocabularySize)
	return tokenizer
	
def trainTokeniserSubwords(dataElements, vocabSize):	
	trainTokeniserFromDataFiles = usePreprocessedDataset
	
	if(tokeniserOnlyTrainOnDictionary):
		min_frequency = 1
		trainTokenizerNumberOfFilesToUse = 1
		path = createDictionaryFile()
		paths = []
		paths.append(path)
		trainTokeniserFromDataFiles = True
	else:
		min_frequency = 2
		if(useSmallTokenizerTrainNumberOfFiles):
			trainTokenizerNumberOfFilesToUse = trainTokenizerNumberOfFilesToUseSmall
		else:
			trainTokenizerNumberOfFilesToUse = datasetNumberOfDataFiles

	tokenizer = ByteLevelBPETokenizer()

	if(trainTokeniserFromDataFiles):
		tokenizer.train(files=dataElements[:trainTokenizerNumberOfFilesToUse], vocab_size=vocabSize, min_frequency=1, special_tokens=specialTokens)
	else:
		tokenizer.train_from_iterator(dataset, length=trainTokenizerNumberOfFilesToUse, vocab_size=vocabSize, min_frequency=1, special_tokens=specialTokens)
	
	#os.mkdir(modelPathName)

	tokenizer.save_model(modelPathName)
		
	return tokenizer

def createDictionaryFile():
	dictionaryList = words.words() 
	print("len(dictionaryList) = ", len(dictionaryList))
	fileName = modelPathName + "/dictionary.txt"
	with open(fileName, 'w', encoding='utf-8') as fp:
		fp.write(' '.join(dictionaryList))
	return fileName
					
def loadTokeniser():
	if(useFullwordTokenizer):
		tokenizer = SBNLPpt_dataTokeniserFullword.loadTokenizerFullwords()
	else:
		tokenizer = loadTokenizerSubwords()
	return tokenizer
		
def loadTokenizerSubwords():	
	tokenizer = RobertaTokenizer.from_pretrained(modelPathName, max_len=sequenceMaxNumTokens)
	return tokenizer

def getTokenizerLength(tokenizer):
	return len(tokenizer)	#Size of the full vocabulary with the added token	#https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils.py

def printSpecialTokenIDs(tokenizer):
	#pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
	print("tokenizer.cls_token_id = ", tokenizer.cls_token_id)	#0 [CLS]
	print("tokenizer.pad_token_id = ", tokenizer.pad_token_id)	#1 [PAD]
	print("tokenizer.sep_token_id = ", tokenizer.sep_token_id)	#2 [SEP]
	print("tokenizer.unk_token_id = ", tokenizer.unk_token_id)	#3 [UNK]

	

#common data loader functions:

def getSampleEncodings(useMLM, input_ids, attention_mask, batched):
	#print("input_ids = ", input_ids)
	#print("attention_mask = ", attention_mask)
	inputIDs = []
	mask = []
	labels = []
	if(legacyDataloaderCode2):
		labels.append(input_ids)
	else:
		labels.append(addLabelsPredictionMaskTokens(input_ids))
	mask.append(attention_mask)
	sampleInputIDs = (input_ids).detach().clone()
	if(useMaskedLM):
		if(batched):
			sampleInputIDsMasked = addMaskTokensBatch(useMLM, sampleInputIDs)
		else:
			sampleInputIDsMasked = addMaskTokensSample(useMLM, sampleInputIDs)
	else:
		sampleInputIDsMasked = sampleInputIDs
	inputIDs.append(sampleInputIDsMasked)
	inputIDs = torch.cat(inputIDs)
	mask = torch.cat(mask)
	labels = torch.cat(labels)
	encodings = {'inputIDs': inputIDs, 'attentionMask': mask, 'labels': labels}
	return encodings

def addLabelsPredictionMaskTokens(input_ids):
	mask_arr = (input_ids == paddingTokenID)
	mask_arr = mask_arr*(labelPredictionMaskTokenID-paddingTokenID)
	labels = input_ids + mask_arr
	#print("labels = ", labels)
	return labels
	
def addMaskTokensBatch(useMLM, inputIDs):
	for i in range(inputIDs.shape[0]):
		inputIDs[i] = addMaskTokensSample(useMLM, inputIDs[i])
	return inputIDs

def addMaskTokensSample(useMLM, inputIDs):
	if(useMLM):
		rand = torch.rand(inputIDs.shape)
		mask_arr = (rand < fractionOfMaskedTokens) * notSpecialTokensIDs(inputIDs)
	else:	
		mask_arr = notSpecialTokensIDs(inputIDs)
	selection = torch.flatten(mask_arr.nonzero()).tolist()
	inputIDs[selection] = customMaskTokenID
	return inputIDs
	
def generateAttentionMask(tokenizer, inputIDs):
	attention_mask = notSpecialTokensIDs(inputIDs).float()
	return attention_mask

def notSpecialTokensIDs(inputIDs):
	inputIDsNotSpecialTokens = (inputIDs > 2) #or (inputIDs != 0) * (inputIDs != 1) * (inputIDs != 2)	#inputIDs are not in [tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id]
	return inputIDsNotSpecialTokens
	
def preprocessDocumentText(documentText):
	if(preprocessRemoveNewLineCharacters):
		documentText = documentText.replace('\n', '')
	return documentText
	
def getNextDocument(datasetIterator):
	document = next(datasetIterator)
	'''
	reachedEndOfDataset = False
	try:
		document = next(datasetIterator)
	except StopIteration:
		reachedEndOfDataset = True
	'''
	if(usePreprocessedDataset):
		documentText = document
	else:
		documentText = document['text']
	return documentText
	
def generateDataFileName(fileIndex):
	fileName = dataPathName + dataPreprocessedFileNameStart + str(fileIndex) + dataPreprocessedFileNameEnd
	return fileName
