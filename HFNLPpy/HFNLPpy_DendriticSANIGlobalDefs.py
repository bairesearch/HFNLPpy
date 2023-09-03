"""HFNLPpy_DendriticSANIGlobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Dendritic SANI Global Defs

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np


#### dendritic structure type ####

DendriticSANIbiologicalSimulationSimple = True	#single dendritic branch with multiple overloaded segments	#does not support resetConnectionTargetNeuronDendriteDuringActivation

#### topk selection ####

selectActivatedTop = True	#select activated top k target neurons during propagation test
if(selectActivatedTop):
	selectActivatedTopK = 3
	
#### computation type ####

vectoriseComputation = True	#parallel processing for optimisation
if(vectoriseComputation):
	updateNeuronObjectActivationLevels = True	#default: False #optional	#only required for drawBiologicalSimulationDynamic (slows down processing)	#activation levels are required to be stored in denditicTree object structure (HopfieldNode/DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic
else:
	updateNeuronObjectActivationLevels = True	#mandatory (typically implied true)
 
 
#### test harness (compare standard/vectorised computation) ####

biologicalSimulationTestHarness = True

writeBiologicalSimulation = False	#initialise (dependent var)
writeBiologicalSimulationActivationStates = False	#initialise (dependent var)
outputFileNameComputationType = False	#initialise (dependent var)
outputBiologicalSimulationNetworkLastSentenceOnly = False	#initialise (dependent var)

emulateVectorisedComputationOrder = False	#initialise (dependent var)
emulateVectorisedComputationOrderVerifyTargetConnectionFound = True	#initialise (dependent var)
emulateVectorisedComputationOrderPreactivateAxonsAndTargetInputs = False	#initialise (dependent var)
vectorisedComputationActivateSomaAfterFinishingPropagation = True	#initialise (dependent var)	#orig: True
emulateVectorisedComputationOrderActivateSomaAfterFinishingPropagation = False	#initialise (dependent var)	#orig: False	#requires storeBranchActivationState

HFNLPnonrandomSeed = False	#initialise (dependent var)

if(biologicalSimulationTestHarness):
	writeBiologicalSimulation = False	#default: True	#write biological simulation to xml file
	if(writeBiologicalSimulation):
		writeBiologicalSimulationActivationStates = True	#print final activation states of network (only valid with writeBiologicalSimulationDynamic)
		outputFileNameComputationType = False	#optional
	if(vectoriseComputation):
		vectorisedComputationActivateSomaAfterFinishingPropagation = True	#sync (True or False)
	else:	
		emulateVectorisedComputationOrderActivateSomaAfterFinishingPropagation = True	#sync (True or False)
		#emulateVectorisedComputationOrder requires biologicalSimulationForward, !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
		emulateVectorisedComputationOrder = True	#change standard computation to execute in order of vectorised computation (for comparison)
		if(emulateVectorisedComputationOrder):
			emulateVectorisedComputationOrderVerifyTargetConnectionFound = True
			emulateVectorisedComputationOrderPreactivateAxonsAndTargetInputs = True			
	HFNLPnonrandomSeed = True	#always generate the same set of random numbers upon execution
	

#### write (xml) ####

writeBiologicalSimulationDynamic = False	#initialise (dependent var)
if(writeBiologicalSimulation):
	writeBiologicalSimulationNetwork = True	#default: True
	writeBiologicalSimulationSentence = False	#default: False
	if(writeBiologicalSimulationNetwork):
		outputBiologicalSimulationNetworkLastSentenceOnly = True	#only write network (xml/graph) at last sentence	#optional
	if(updateNeuronObjectActivationLevels or not vectoriseComputation):
		writeBiologicalSimulationDynamic = True	#default: False	#write dynamic activation levels of biological simulation	#optional
		if(writeBiologicalSimulationDynamic):
			writeBiologicalSimulationSentenceDynamic = True	#default: True	#write graph for sentence neurons and their dendritic tree
			writeBiologicalSimulationNetworkDynamic = False	#default: True	#write graph for entire network (not just sentence)
	else:
		print("HFNLPpy_DendriticSANIPropagateVectorised warning: updateNeuronObjectActivationLevels is required for vectoriseComputation:writeBiologicalSimulationDynamic (if writeBiologicalSimulationDynamic is required; either enable updateNeuronObjectActivationLevels or disable vectoriseComputation)")
		writeBiologicalSimulationDynamic = False	#mandatory: False


#### draw ####

drawBiologicalSimulationDynamicHighlightNewActivations = True	#useful with resetConnectionTargetNeuronDendriteAfterSequence/resetConnectionTargetNeuronDendriteDuringActivation to visually distinguish between new activations (at current time) and prior activations	#if debugCalculateNeuronActivation*: incompatible with first call of draw since activationStateNew is reset by getActivationColor
if(drawBiologicalSimulationDynamicHighlightNewActivations):
	highlightNewActivationColor = 'magenta'	#'black'

drawBiologicalSimulation = True	#optional
if(drawBiologicalSimulation):
	drawBiologicalSimulationPlot = True	#default: True
	drawBiologicalSimulationSave = False	#default: False	#save to file
	drawBiologicalSimulationSentence = True	#default: True	#draw graph for sentence neurons and their dendritic tree
	drawBiologicalSimulationNetwork = False	#default: False	#draw graph for entire network (not just sentence)

#if(biologicalSimulationForward):	#required for drawBiologicalSimulationSentenceDynamic/drawBiologicalSimulationNetworkDynamic
if(updateNeuronObjectActivationLevels or not vectoriseComputation):
	drawBiologicalSimulationDynamic = False	#draw dynamic activation levels of biological simulation	#optional
	if(drawBiologicalSimulationDynamic):
		drawBiologicalSimulationDynamicPlot = True	#default: True
		drawBiologicalSimulationDynamicSave = False	#default: False	#save to file
		drawBiologicalSimulationSentenceDynamic = True	#default: True	#draw graph for sentence neurons and their dendritic tree
		drawBiologicalSimulationNetworkDynamic = False	#default: False	#draw graph for entire network (not just sentence)
else:
	print("HFNLPpy_DendriticSANIPropagateVectorised warning: updateNeuronObjectActivationLevels is required for vectoriseComputation:drawBiologicalSimulationDynamic (if drawBiologicalSimulationDynamic is required; either enable updateNeuronObjectActivationLevels or disable vectoriseComputation)")
	drawBiologicalSimulationDynamic = False	#mandatory: False

debugCalculateNeuronActivation = False	#default: False
if(debugCalculateNeuronActivation):
	sentenceIndexDebug = 3	#10	#208	#397	#1	#10	#397
	wSourceDebug = 9	#3
	wTargetDebug = 15	#4
else:
	wTargetDebug = None	


#### propagation algorithm (dendrite activation) ####

reversePropagationOrder = True		#optional	#True: original implementation
if(emulateVectorisedComputationOrder):
	emulateVectorisedComputationOrderReversed = reversePropagationOrder	#initialise (dependent var)


#### seed HF network with subsequence ####

seedHFnetworkSubsequence = True #seed/prime HFNLP network with initial few words of a trained sentence and verify that full sentence is sequentially activated (interpret last sentence as target sequence, interpret first seedHFnetworkSubsequenceLength words of target sequence as seed subsequence)
if(seedHFnetworkSubsequence):
	#seedHFnetworkSubsequence currently requires !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	seedHFnetworkSubsequenceLength = 4	#must be < len(targetSentenceConceptNodeList)
	seedHFnetworkSubsequenceBasic = False	#emulate simulateBiologicalHFnetworkSequenceTrain:simulateBiologicalHFnetworkSequenceNodePropagateWrapper method (only propagate those activate neurons that exist in the target sequence); else propagate all active neurons
	seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant = True


#### dendritic encoding calibration ####

reduceCompletenessOfEncodingWithPreviousContextDistance = True	#the more proximal the previous context, the more likely to form a synapse
reduceCompletenessOfEncodingWithSequenceLength = False	#when a predictive sequence is short ensure almost every previous word is encoded in branch structure	#the shorter the sequence, the more likely to form a synapse
reduceCompletenessOfEncodingCalibration = False		#initialise (dependent var)
if(reduceCompletenessOfEncodingWithPreviousContextDistance):
	reduceCompletenessOfEncodingCalibration = True
if(reduceCompletenessOfEncodingWithSequenceLength):
	reduceCompletenessOfEncodingWithSequenceLengthCalibration = 10.0	#subsequence encoding length modifier = predictiveSequenceLength/reduceCompletenessOfEncodingWithSequenceLengthCalibration
	reduceCompletenessOfEncodingCalibration = True

subsequenceLengthRandExponential = False	#orig: True
if(subsequenceLengthRandExponential):
	subsequenceLengthRandCalibration = 3.0	#5.0
else:
	#subsequenceLengthRandLinear
	subsequenceLengthRandCalibration = 2.0	#3.0	#2.0
subsequenceLengthCalibration = 1.0*subsequenceLengthRandCalibration	#CONSIDER: reduce proportional to number of vertical branches

if(reduceCompletenessOfEncodingCalibration):
	averageSentenceLength = 10.0
	subsequenceLengthCalibration = subsequenceLengthCalibration/averageSentenceLength	

#probabilityOfSubsequenceThreshold = 0.01	#FUTURE: calibrate depending on number of branches/sequentialSegments etc
	
enforceMinimumEncodedSequenceLength = True	#do not execute addPredictiveSequenceToNeuron if predictive sequence is short (ie does not use up the majority of numberOfBranches1)
if(enforceMinimumEncodedSequenceLength):
	minimumEncodedSequenceLength = 4	#should be high enough to fill a significant proportion of dendrite vertical branch length (numberOfBranches1)	#~seedHFnetworkSubsequenceLength
	
preventGenerationOfDuplicateConnections = True	#note sequentialSegment inputs will be stored as a dictionary indexed by source node name (else indexed by sequentialSegmentInputIndex)
	

#### encode syntax in dendritic branch structure ####

supportForNonBinarySubbranchSize = False	#initialise (dependent var)
performSummationOfSequentialSegmentInputsAcrossBranch = False	#initialise (dependent var)
weightedSequentialSegmentInputs = False	#initialise (dependent var)
allowNegativeActivationTimes = False	#initialise (dependent var)	#calculateNeuronActivationSyntacticalBranchDPlinear current implementation does not require allowNegativeActivationTimes
expectFirstBranchSequentialSegmentConnection = True	#initialise (dependent var)	#True:default	#False: orig implementation

biologicalSimulationEncodeSyntaxInDendriticBranchStructure = False	#determines HFNLPpy_hopfieldGraph:useDependencyParseTree	#speculative: use precalculated syntactical structure to generate dendritic branch connections (rather than deriving syntax from commonly used dendritic subsequence encodings)
if(biologicalSimulationEncodeSyntaxInDendriticBranchStructure):
	biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect = True	#speculative: directly encode precalculated syntactical structure into dendritic branches
	if(biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect):
		expectFirstBranchSequentialSegmentConnection = False
		supportForNonBinarySubbranchSize = True	#required by useDependencyParseTree:biologicalSimulationEncodeSyntaxInDendriticBranchStructureDirect with non-binary dependency/constituency parse trees
		allowNegativeActivationTimes = True
	else:
		#implied biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinear = True	#speculative: convert precalculated syntactical structure to linear subsequences before encoding into dendritic branches
		biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearHierarchical = False	#incomplete #adds the most distant nodes to the start of a linear contextConceptNodesList - will still perform propagate/predict in reverse order of tree crawl
		#if(not biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearHierarchical):
		#	implied biologicalSimulationEncodeSyntaxInDendriticBranchStructureLinearCrawl = True: adds the nodes in reverse order of tree crawl to a linear contextConceptNodesList) - will also perform propagate/predict in reverse order of tree crawl

expectFirstBranchSequentialSegmentConnectionStrictNumBranches1 = False
if(expectFirstBranchSequentialSegmentConnection):
	expectFirstBranchSequentialSegmentConnectionStrictNumBranches1 = True	#optional	#ensure numberOfBranches1=numberOfVerticalBranches

#non binary/consistent branch encoding;
if(supportForNonBinarySubbranchSize):
	performSummationOfSequentialSegmentInputsAcrossBranch = True
	debugBiologicalSimulationEncodeSyntaxInDendriticBranchStructure = False	#reduce number of subbranches to support and draw simpler dependency tree (use with drawBiologicalSimulationDynamic)
	if(performSummationOfSequentialSegmentInputsAcrossBranch):
		weightedSequentialSegmentInputs = True
		#performSummationOfSequentialSegmentInputsAcrossBranch does not support numberOfBranchSequentialSegments>1 as branch summation occurs using the final sequentialSegment activationLevel of each subbranch
		#performSummationOfSequentialSegmentInputsAcrossBranch does not support performSummationOfSequentialSegmentInputs (or multiple simultaneous inputs at a sequential segment) as this will arbitrarily overwrite the precise sequentialSegment activationLevel of a subbranch

if(allowNegativeActivationTimes):
	minimumActivationTime = -1000
else:
	minimumActivationTime = 0	#alternatively set -1;	#initial activation time of dendritic sequence set artificially low such that passSegmentActivationTimeTests automatically pass (not required (as passSegmentActivationTimeTests are ignored for currentSequentialSegmentInput.firstInputInSequence)
	

#### standard computation ####

standardComputationOptimised = False	#initialise (dependent var)
if(not vectoriseComputation):
	standardComputationOptimised = True	#optional	#False: original implementation	#only perform local propagation at active connections


#### vectorised computation ####

if(vectoriseComputation):
	import tensorflow as tf
	vectoriseComputationCurrentDendriticInput = True	#mandatory - default behaviour
	if(vectoriseComputationCurrentDendriticInput):
		vectoriseComputationIndependentBranches = True	#mandatory - default behaviour
	batchSizeDefault = 100	#high batch size allowed since parallel processing simple/small scalar operations (on effective boolean synaptic inputs), lowered proportional to max (most distal) numberOfHorizontalBranches	#not used (createDendriticTreeVectorised is never called with batched=True)
	
	if(updateNeuronObjectActivationLevels):
		recordVectorisedBranchObjectList = True	#vectorisedBranchObjectList is required to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels (as HFNLPpy_DendriticSANIDraw currently only supports drawing of denditicTree object structure activations)  
	else:
		recordVectorisedBranchObjectList = False	#vectorisedBranchObjectList is not required as it is not necessary to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput); activation levels are not required to be stored in denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput)
else:
	vectoriseComputationCurrentDendriticInput = False

onlyPropagateIfConceptNeuronTargetActivatedByConceptNeuronSourceVectorised = False	#True: orig implementation
if(vectoriseComputation):
	if(enforceMinimumEncodedSequenceLength):
		onlyPropagateIfConceptNeuronTargetActivatedByConceptNeuronSourceVectorised = False	#mandatory
	
	
#### dendritic branch/sequential segment activation level cache ####

storeBranchActivationState = True	#True: orig implementation	#False: storeBranchActivationLevel

performSummationOfSequentialSegmentInputs = False #allows sequential segment activation to be dependent on summation of individual local inputs #support multiple source neurons fired simultaneously	#consider renaming to performSummationOfSequentialSegmentInputsLocal
if(performSummationOfSequentialSegmentInputs):
	weightedSequentialSegmentInputs = True
	#summationOfSequentialSegmentInputsFirstInputInSequenceOverride = True	#mandatory (only implementation coded) #True: orig HFNLPpy_DendriticSANIPropagateStandard method	 #False: orig HFNLPpy_DendriticSANIPropagateVectorised method
if(weightedSequentialSegmentInputs):
	sequentialSegmentMinActivationLevel = 1.0	#requirement: greater or equal to sequentialSegmentMinActivationLevel
else:
	sequentialSegmentMinActivationLevel = 1	#always 1 (not used)

#default activation levels;
#key:
#"object" = neuron/dendritic tree class structure
#"local" = activation level for synaptic inputs/sequential segments
#"area" = activation level for dendritic branches/somas/axons
objectAreaActivationLevelOff = False
objectAreaActivationLevelOn = True
if(weightedSequentialSegmentInputs):
	#numeric (sequential segment only consider depolarised if sequential requirements met and summed activationLevel of sequential inputs passes threshold)
	objectLocalActivationLevelOff = 0.0
	objectLocalActivationLevelOn = sequentialSegmentMinActivationLevel		
else:
	#bool (sequential segment only consider depolarised if sequential requirements met)
	objectLocalActivationLevelOff = False
	objectLocalActivationLevelOn = True	
vectorisedActivationLevelOff = 0.0
vectorisedActivationLevelOn = 1.0
vectorisedActivationTimeFlagDefault = 0	#boolean flag (not a numeric activation time)
vectorisedActivationTimeFlagFirstInputInSequence = 1	#boolean flag (not a numeric activation time)


#### propagation algorithm (source/target activation) ####

if(vectoriseComputation):
	biologicalSimulationForward = True	#mandatory (only implementation) #required for drawBiologicalSimulationSentenceDynamic/drawBiologicalSimulationNetworkDynamic
else:
	biologicalSimulationForward = True	#optional	#True: default (mandatory for many configurations)	#orig implementation; False (simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup)


#### activation reset ####

resetSourceNeuronAxonAfterActivation = True	#mandatory

if(biologicalSimulationForward):
	#dendrite activations reset mode selection (typically select one only):
	if(DendriticSANIbiologicalSimulationSimple):
		resetConnectionTargetNeuronDendriteAfterSequence = False	#optional	#does not reset sequential segment activations during sequence propagation (overwrites them)
		resetConnectionTargetNeuronDendriteDuringActivation = False 	#optional #reset previous sequential segments of newly activated sequential segments  #requires !overwriteSequentialSegments, !performSummationOfSequentialSegmentInputs
		resetSourceNeuronDendriteAfterActivation = False	#optional	#True: orig implementation	#not compatible with recursive connections (ie nodeX -> nodeX; connection created from repeated words in sentence)	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
		resetConnectionTargetNeuronDendriteAfterActivation = False	#optional	#reset all connection target neuron dendrites after activation	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
		resetTargetNeuronDendriteAfterActivation = False	#optional	#only reset expected target neuron dendrites after activation	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
	else:
		resetConnectionTargetNeuronDendriteAfterSequence = False	#optional	#does not reset sequential segment activations during sequence propagation (overwrites them)
		resetConnectionTargetNeuronDendriteDuringActivation = True 	#optional #reset previous sequential segments of newly activated sequential segments  #requires !overwriteSequentialSegments, !performSummationOfSequentialSegmentInputs
		resetSourceNeuronDendriteAfterActivation = False	#optional	#True: orig implementation	#not compatible with recursive connections (ie nodeX -> nodeX; connection created from repeated words in sentence)	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
		resetConnectionTargetNeuronDendriteAfterActivation = False	#optional	#reset all connection target neuron dendrites after activation	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
		resetTargetNeuronDendriteAfterActivation = False	#optional	#only reset expected target neuron dendrites after activation	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
else:
	resetTargetNeuronDendriteAfterActivation = True	#optional

resetConnectionTargetNeuronDendriteDuringActivationFreezeUntilRoundCompletion = False	#initialise (dependent var)
if(resetConnectionTargetNeuronDendriteDuringActivation):
	if(vectoriseComputation):
		reversePropagationOrder = False	#mandatory #required to prevent dynamic reset from overwriting active connections of more distal segment before they are propagated
	else:
		#reversePropagationOrder = True	#optional
		emulateVectorisedComputationOrder = True	#mandatory
		if(emulateVectorisedComputationOrder):
			emulateVectorisedComputationOrderReversed = False	#mandatory	#required to prevent dynamic reset from overwriting active connections of more distal segment before they are propagated
		standardComputationOptimised = True	#optional	#only perform local propagation at active connections
	#depreciated implementation (for reversePropagationOrder);
	#if(not vectoriseComputation):
	#	if(not emulateVectorisedComputationOrder):
	#		resetConnectionTargetNeuronDendriteDuringActivationFreezeUntilRoundCompletion = False	#incomplete	#note for HFNLPpy_DendriticSANIPropagateVectorised this is implied True because entire source propagation round is executed simultaneously in parallel

verifyRepolarisationTime = False	#initialise (dependent var)
overwriteSequentialSegmentsAfterPropagatingSignal = False	#initialise (dependent var)
if(DendriticSANIbiologicalSimulationSimple):
	overwriteSequentialSegments = True	#optional
	if(overwriteSequentialSegments):
		overwriteSequentialSegmentsAfterPropagatingSignal = True
		verifyRepolarisationTime = False	
else:
	if(resetConnectionTargetNeuronDendriteAfterSequence):
		overwriteSequentialSegments = True
		if(overwriteSequentialSegments):
			overwriteSequentialSegmentsAfterPropagatingSignal = True	#only overwrite activation if a more proximal branch has been activated	#current implementation is insufficient (too lenient wrt repeated concepts); may need to only consider a sequential segment reactivated if its subbranch activation times have increased (similar to resetConnectionTargetNeuronDendriteDuringActivation)
			verifyRepolarisationTime = True
		#resetConnectionTargetNeuronDendriteAfterSequence does not currently support !expectFirstBranchSequentialSegmentConnection
	else:
		overwriteSequentialSegments = False	#orig: False	#False: prevent reactivation of sequential segments (equates to a long repolarisation time of ~= sentenceLength)	#False: algorithmTimingWorkaround2
if(vectoriseComputation):
	verifyReactivationTime2 = verifyRepolarisationTime	#TODO: variables could be merged

deactivateSequentialSegmentsIfAllConnectionInputsOff = False	#initialise (dependent var)
deactivateSequentialSegmentsIfTimeTestsFail = False	#initialise (dependent var)
drawBiologicalSimulationDynamicFrozenActivations = False	#initialise (dependent var)
if(overwriteSequentialSegments):
	drawBiologicalSimulationDynamicFrozenActivations = True
	if(drawBiologicalSimulationDynamicFrozenActivations):
		frozenActivationColor = 'blue'
	if(vectoriseComputation):
		deactivateSequentialSegmentsIfTimeTestsFail = True	#mandatory implied True (only coded implementation)
	else:
		deactivateSequentialSegmentsIfTimeTestsFail = True	#Default/mandatory: True		#deactivates sequential segments upon reencountering insufficient prior dendritic activation conditions (level/time)	#deactivates only if connectionInputActivationFound but passSegmentActivationTimeTests fail
	if(vectoriseComputation):
		deactivateSequentialSegmentsIfAllConnectionInputsOff = False	#Default/mandatory: False	#deactivates deactivates if !connectionInputActivationFound 
	else:
		deactivateSequentialSegmentsIfAllConnectionInputsOff = False	#mandatory implied False (only coded implementation)

deactivateConnectionTargetIfSomaActivationNotFound = True	#default:True #True: orig simulateBiologicalHFnetworkSequenceNodesPropagateParallel:calculateNeuronActivationParallel method, False: orig simulateBiologicalHFnetworkSequenceNodePropagateStandard method
if(standardComputationOptimised):
	if(not emulateVectorisedComputationOrderActivateSomaAfterFinishingPropagation):
		deactivateConnectionTargetIfSomaActivationNotFound = False	#mandatory	#False: required because calculateNeuronActivationStandardWrapper returns valid somaActivationFound only if((currentBranchIndex1 == branchIndex1MostProximal) and (currentSequentialSegmentIndex == sequentialSegmentIndexMostProximal))
		 
algorithmTimingWorkaround1 = False	#insufficient workaround


#### input/synaptic activation level cache ####

if(vectoriseComputation):
	recordSequentialSegmentInputActivationLevels = True	#optional
	if(updateNeuronObjectActivationLevels):
		recordSequentialSegmentInputActivationLevels = True	#required for draw of active simulation - required by drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels	
else:
	recordSequentialSegmentInputActivationLevels = True	#optional (not required by HFNLPpy_DendriticSANIPropagateStandard processing, and dynamic draw is not supported)
if(vectoriseComputation):
	if(recordSequentialSegmentInputActivationLevels):
		vectoriseComputionUseSequentialSegmentInputActivationLevels	= False	#not yet implemented	#not required as local segment inputs must fire simultaneously; so they can be stored as a segment scalar value	#only ever used in buffer processing
		if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
			numberOfSequentialSegmentInputs = 100	#max number available

storeSequentialSegmentInputIndexValues = False	#not required	#index record value not robust if inputs are removed (synaptic atrophy)	#HFNLPpy_DendriticSANIDraw can use currentSequentialSegmentInputIndexDynamic instead


#### dendritic structure ####

if(DendriticSANIbiologicalSimulationSimple):
	numberOfBranches1 = 1	#number of vertical branches
	numberOfBranches2 = 1
	numberOfBranchSequentialSegments = 5
else:
	numberOfBranches1 = 5	#5	#3	#numberOfVerticalBranches (w expectFirstBranchSequentialSegmentConnectionStrictNumBranches1) or numberOfVerticalBranches-1 (w !expectFirstBranchSequentialSegmentConnectionStrictNumBranches1)
	if(supportForNonBinarySubbranchSize):
		if(debugBiologicalSimulationEncodeSyntaxInDendriticBranchStructure):
			numberOfBranches2 = 4
		else:
			numberOfBranches2 = 20	#8	#number of new horizontal branches created at each vertical branch	#must sync with max number subbrances of constituency/dependency parser	#if dependencyParser: maximum number of dependents per governor
	else:
		numberOfBranches2 = 2	#number of new horizontal branches created at each vertical branch
		#[1,2,4,8]	#number of new horizontal branches created at each vertical branch
	numberOfBranchSequentialSegments = 1	#1+	#sequential inputs (FUTURE: if > 1: each branch segment may require sequential inputs)
	#numberOfBranchSequentialSegmentInputs = 1	#1+	#nonSequentialInputs	#in current implementation (non-parallel generative network) number of inputs at sequential segment is dynamically increased on demand #not used; currently encode infinite number of

sequentialSegmentIndexMostProximal = 0
branchIndex1MostProximal = 0


#### dendritic encoding/propagation ####

#dendritic structure (encoding requirements):
numberOfHorizontalSubBranchesTrained = numberOfBranches2

#dendritic structure (propagation requirements):
numberOfHorizontalSubBranchesRequiredForActivation = 2	#calibrate

trainSubsetOfHorizontalSubbranches = False	#optional
if(trainSubsetOfHorizontalSubbranches):
	numberOfHorizontalSubBranchesTrained = 1
	
requireSubbranchOrSequentialSegmentForActivation = False	#initialise (dependent var)
if(numberOfHorizontalSubBranchesTrained == 2 and numberOfHorizontalSubBranchesRequiredForActivation == 2 and numberOfBranchSequentialSegments == 1):
	requireSubbranchOrSequentialSegmentForActivation = False	#optional
	if(requireSubbranchOrSequentialSegmentForActivation):
		#does not support performSummationOfSequentialSegmentInputsAcrossBranch
		numberOfHorizontalSubBranchesRequiredForActivation = 1
		numberOfHorizontalSubBranchesOrSequentialSegmentsRequiredForActivation = 2
		#resetConnectionTargetNeuronDendriteAfterSequence:vectorisedBranchActivationStateBatchSequentialSegmentFinalNew/newActivationFoundFinalSequentialSegment not supported (most proximal sequential segment in dendritic tree must be active)

if(verifyRepolarisationTime):	
	activationRepolarisationTime = 1	#calibrate

verifyPropagationTime = False	#default: False	#early activation of sequential segments (from out of context connections) may prevent verifyPropagationTime time requirements from being met
if(verifyPropagationTime):
	activationPropagationTimeMax = 3	#max propagation time between sequential segments

def printe(str):
	print(str)
	exit()
