"""HFNLPpy_biologicalSimulationGlobalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
ATNLP Biological Simulation Global Defs

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np


vectoriseComputation = True	#parallel processing for optimisation
if(vectoriseComputation):
	updateNeuronObjectActivationLevels = True	#optional	#only required for drawBiologicalSimulationDynamic (slows down processing)	#activation levels are required to be stored in denditicTree object structure (HopfieldNode/DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic


biologicalSimulationTestHarness = False
HFNLPnonrandomSeed = False	#initialise (dependent var)
emulateVectorisedComputationOrder = False	#initialise (dependent var)
if(biologicalSimulationTestHarness):
	if(not vectoriseComputation):
		#emulateVectorisedComputationOrder requires biologicalSimulationForward, !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
		emulateVectorisedComputationOrder = True	#change standard computation to execute in order of vectorised computation (for comparison)
	HFNLPnonrandomSeed = True	#always generate the same set of random numbers upon execution

enforceMinimumEncodedSequenceLength = False	#do not execute addPredictiveSequenceToNeuron if predictive sequence is short (ie does not use up the majority of numberOfBranches1)
if(enforceMinimumEncodedSequenceLength):
	minimumEncodedSequenceLength = 4	#should be high enough to fill a significant proportion of dendrite vertical branch length (numberOfBranches1)	#~seedHFnetworkSubsequenceLength
	
seedHFnetworkSubsequence = False #seed/prime HFNLP network with initial few words of a trained sentence and verify that full sentence is sequentially activated (interpret last sentence as target sequence, interpret first seedHFnetworkSubsequenceLength words of target sequence as seed subsequence)
if(seedHFnetworkSubsequence):
	#seedHFnetworkSubsequence currently requires !biologicalSimulationEncodeSyntaxInDendriticBranchStructure
	seedHFnetworkSubsequenceLength = 4	#must be < len(targetSentenceConceptNodeList)
	seedHFnetworkSubsequenceBasic = False	#emulate simulateBiologicalHFnetworkSequenceTrain:simulateBiologicalHFnetworkSequenceNodePropagateWrapper method (only propagate those activate neurons that exist in the target sequence); else propagate all active neurons
	seedHFnetworkSubsequenceVerifySeedSentenceIsReplicant = True


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
	
	
preventGenerationOfDuplicateConnections = True	#note sequentialSegment inputs will be stored as a dictionary indexed by source node name (else indexed by sequentialSegmentInputIndex)

storeSequentialSegmentInputIndexValues = False	#not required	#index record value not robust if inputs are removed (synaptic atrophy)	#HFNLPpy_biologicalSimulationDraw can use currentSequentialSegmentInputIndexDynamic instead


deactivateConnectionTargetIfSomaActivationNotFound = True	#default:True #True: orig simulateBiologicalHFnetworkSequenceNodesPropagateParallel:calculateNeuronActivationParallel method, False: orig simulateBiologicalHFnetworkSequenceNodePropagateStandard method

overwriteSequentialSegments = False	#default: False	#False: prevent reactivation of sequential segments (equates to a long repolarisation time of ~= sentenceLength)	#algorithmTimingWorkaround2
deactivateSequentialSegmentsIfAllConnectionInputsOff = False	#initialise (dependent var)
deactivateSequentialSegmentsIfTimeTestsFail = False	#initialise (dependent var)
if(overwriteSequentialSegments):
	if(vectoriseComputation):
		deactivateSequentialSegmentsIfTimeTestsFail = True	#mandatory implied True (only coded implementation)
	else:
		deactivateSequentialSegmentsIfTimeTestsFail = True	#Default/mandatory: True		#deactivates sequential segments upon reencountering insufficient prior dendritic activation conditions (level/time)	#deactivates only if connectionInputActivationFound but passSegmentActivationTimeTests fail
	if(vectoriseComputation):
		deactivateSequentialSegmentsIfAllConnectionInputsOff = False	#Default/mandatory: False	#deactivates deactivates if !connectionInputActivationFound 
	else:
		deactivateSequentialSegmentsIfAllConnectionInputsOff = False	#mandatory implied False (only coded implementation)
		  
algorithmTimingWorkaround1 = False	#insufficient workaround


performSummationOfSequentialSegmentInputs = False #allows sequential segment activation to be dependent on summation of individual local inputs #support multiple source neurons fired simultaneously	#consider renaming to performSummationOfSequentialSegmentInputsLocal
if(performSummationOfSequentialSegmentInputs):
	weightedSequentialSegmentInputs = True
	#summationOfSequentialSegmentInputsFirstInputInSequenceOverride = True	#mandatory (only implementation coded) #True: orig HFNLPpy_biologicalSimulationPropagateStandard method	 #False: orig HFNLPpy_biologicalSimulationPropagateVectorised method
if(weightedSequentialSegmentInputs):
	sequentialSegmentMinActivationLevel = 1.0	#requirement: greater or equal to sequentialSegmentMinActivationLevel
else:
	sequentialSegmentMinActivationLevel = 1	#always 1 (not used)

if(vectoriseComputation):
	import tensorflow as tf
	vectoriseComputationCurrentDendriticInput = True	#mandatory - default behaviour
	if(vectoriseComputationCurrentDendriticInput):
		vectoriseComputationIndependentBranches = True	#mandatory - default behaviour
	batchSizeDefault = 100	#high batch size allowed since parallel processing simple/small scalar operations (on effective boolean synaptic inputs), lowered proportional to max (most distal) numberOfHorizontalBranches	#not used (createDendriticTreeVectorised is never called with batched=True)
	
	if(updateNeuronObjectActivationLevels):
		recordVectorisedBranchObjectList = True	#vectorisedBranchObjectList is required to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput) for drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels (as HFNLPpy_biologicalSimulationDraw currently only supports drawing of denditicTree object structure activations)  
	else:
		recordVectorisedBranchObjectList = False	#vectorisedBranchObjectList is not required as it is not necessary to convert vectorised activations back to denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput); activation levels are not required to be stored in denditicTree object structure (DendriticBranch/SequentialSegment/SequentialSegmentInput)
else:
	vectoriseComputationCurrentDendriticInput = False

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
	

if(vectoriseComputation):
	biologicalSimulationForward = True	#mandatory (only implementation) #required for drawBiologicalSimulationDendriticTreeSentenceDynamic/drawBiologicalSimulationDendriticTreeNetworkDynamic
else:
	biologicalSimulationForward = True	#optional	#orig implementation; False (simulateBiologicalHFnetworkSequenceNodePropagateReverseLookup)


resetSourceNeuronAxonAfterActivation = True	#mandatory

resetConnectionTargetNeuronDendriteDuringActivation = False	#requires !overwriteSequentialSegments, !performSummationOfSequentialSegmentInputs
resetConnectionTargetNeuronDendriteDuringActivationFreezeUntilRoundCompletion = False	#initialise (dependent var)
if(resetConnectionTargetNeuronDendriteDuringActivation):
	if(not vectoriseComputation):
		resetConnectionTargetNeuronDendriteDuringActivationFreezeUntilRoundCompletion = False	#incomplete
	
resetSourceNeuronDendriteAfterActivation = False	#initialise (dependent var)
resetConnectionTargetNeuronDendriteAfterActivation = False	#initialise (dependent var)
resetTargetNeuronDendriteAfterActivation = False	#initialise (dependent var)
if(not resetConnectionTargetNeuronDendriteDuringActivation):
	if(biologicalSimulationForward):
		resetSourceNeuronDendriteAfterActivation = True	#optional	#True: orig implementation	#not compatible with recursive connections (ie nodeX -> nodeX; connection created from repeated words in sentence)	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
		resetConnectionTargetNeuronDendriteAfterActivation = False	#optional	#reset all connection target neuron dendrites after activation	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
		resetTargetNeuronDendriteAfterActivation = False	#optional	#only reset expected target neuron dendrites after activation	#not compatible with repeated concepts; consider the sequence of words: Q(1) A(2) R(3) A(4)
	else:
		resetTargetNeuronDendriteAfterActivation = False	#optional


if(vectoriseComputation):
	recordSequentialSegmentInputActivationLevels = True	#optional
	if(updateNeuronObjectActivationLevels):
		recordSequentialSegmentInputActivationLevels = True	#required for draw of active simulation - required by drawBiologicalSimulationDynamic:updateNeuronObjectActivationLevels	
else:
	recordSequentialSegmentInputActivationLevels = True	#optional (not required by HFNLPpy_biologicalSimulationPropagateStandard processing, and dynamic draw is not supported)
if(vectoriseComputation):
	if(recordSequentialSegmentInputActivationLevels):
		vectoriseComputionUseSequentialSegmentInputActivationLevels	= False	#not yet implemented	#not required as local segment inputs must fire simultaneously; so they can be stored as a segment scalar value	#only ever used in buffer processing
		if(vectoriseComputionUseSequentialSegmentInputActivationLevels):
			numberOfSequentialSegmentInputs = 100	#max number available


numberOfBranches1 = 3	#number of vertical branches -1
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

#probabilityOfSubsequenceThreshold = 0.01	#FUTURE: calibrate depending on number of branches/sequentialSegments etc

subsequenceLengthCalibration = 1.0

numberOfHorizontalSubBranchesRequiredForActivation = 2	#calibrate
activationRepolarisationTime = 1	#calibrate

sequentialSegmentIndexMostProximal = 0
branchIndex1MostProximal = 0