"""HFNLPpy_hopfieldConnectionClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

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

objectTypeConnection = 5

class HopfieldConnection:
	def __init__(self, nodeSource, nodeTarget, activationTime, spatioTemporalIndex, SANIbiologicalPrototype, SANIbiologicalSimulation):
		#primary vars;
		self.nodeSource = nodeSource
		self.nodeTarget = nodeTarget	#for SANIbiologicalPrototype: interpret as axon synapse
		self.activationTime = activationTime	#last activation time (used to calculate recency)	#not currently used
		self.activationLevel = False	#currently only used by drawBiologicalSimulationDynamic
		self.spatioTemporalIndex = spatioTemporalIndex	#creation time (not used by biological implementation)		#for SANIbiologicalPrototype: e.g. 1) interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex/clause); or 2) store spatiotemporal index synapses on separate dendritic branch
		
		self.SANIbiologicalPrototype = False
		if(SANIbiologicalPrototype):
			#for SANIbiologicalPrototype: interpret connection as unique synapse
			self.SANIbiologicalPrototype = True
			self.weight = 1.0	
			self.contextConnection = False
			self.contextConnectionSANIindex = 0
		self.SANIbiologicalSimulation = False
		if(SANIbiologicalSimulation):
			#for SANIbiologicalSimulation: interpret connection as unique synapse
			self.SANIbiologicalSimulation = True
			self.nodeTargetSequentialSegmentInput = None
			self.weight = 1.0	#for weightedSequentialSegmentInputs only
			self.objectType = objectTypeConnection
