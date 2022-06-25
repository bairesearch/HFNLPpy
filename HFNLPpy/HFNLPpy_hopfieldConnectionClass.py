"""HFNLPpy_hopfieldConnectionClass.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
ATNLP Hopfield Connection Class

"""

import numpy as np


class HopfieldConnection:
	def __init__(self, nodeSource, nodeTarget, activationTime, spatioTemporalIndex, biologicalPrototype, biologicalSimulation):
		#primary vars;
		self.nodeSource = nodeSource
		self.nodeTarget = nodeTarget	#for biologicalPrototype: interpret as axon synapse
		self.activationTime = activationTime	#last activation time (used to calculate recency)	#not currently used
		self.activationLevel = False	#currently only used by drawBiologicalSimulationDynamic
		self.spatioTemporalIndex = spatioTemporalIndex	#creation time (not used by biological implementation)		#for biologicalPrototype: e.g. 1) interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex/clause); or 2) store spatiotemporal index synapses on separate dendritic branch
		
		self.biologicalPrototype = False
		if(biologicalPrototype):
			#for biologicalPrototype: interpret connection as unique synapse
			self.biologicalPrototype = True
			self.weight = 1.0	
			self.contextConnection = False
			self.contextConnectionSANIindex = 0
		self.biologicalSimulation = False
		if(biologicalSimulation):
			#for biologicalSimulation: interpret connection as unique synapse
			self.biologicalSimulation = True
			self.nodeTargetSequentialSegmentInput = None
