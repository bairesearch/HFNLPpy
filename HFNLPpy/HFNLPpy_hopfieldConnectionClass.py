"""HFNLPpy_hopfieldConnectionClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Hopfield Connection Class

"""

import numpy as np
from HFNLPpy_globalDefs import *

objectTypeConnection = 5

class HopfieldConnection:
	def __init__(self, nodeSource, nodeTarget, activationTime=-1, spatioTemporalIndex=-1, useAlgorithmDendriticPrototype=False):
		#primary vars;
		self.nodeSource = nodeSource
		self.nodeTarget = nodeTarget	#for useAlgorithmDendriticPrototype: interpret as axon synapse
		self.contextConnection = False	#else causalConnection (next word prediction)
		self.weight = 1.0	
		
		if(useAlgorithmLayeredSANI):
			self.isSANIcompoundNode = False
			self.SANIactivationState = False
			self.weight = 0	#SANIassociationStrength
			self.SANInodeAssigned = False
			self.SANInode = None
			self.activationStatePartial = False	#always false
			self.SANIoptionalCausalConnection = False
		self.useAlgorithmDendriticPrototype = False
		if(useAlgorithmDendriticPrototype):
			#for useAlgorithmDendriticPrototype: interpret connection as unique synapse
			self.useAlgorithmDendriticPrototype = True
			self.spatioTemporalIndex = spatioTemporalIndex	#creation time (not used by biological implementation)		#for useAlgorithmDendriticPrototype: e.g. 1) interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex/clause); or 2) store spatiotemporal index synapses on separate dendritic branch
			self.contextConnectionSANIindex = 0
		if(useAlgorithmDendriticSANI):
			#for useAlgorithmDendriticSANI: interpret connection as unique synapse
			self.activationLevel = False	#currently only used by drawBiologicalSimulationDynamic
			self.nodeTargetSequentialSegmentInput = None
			#weight for weightedSequentialSegmentInputs only
			self.objectType = objectTypeConnection
