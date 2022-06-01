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
	def __init__(self, nodeSource, nodeTarget, activationTime, spatioTemporalIndex, weight=1.0):
		#primary vars;
		self.nodeSource = nodeSource
		self.nodeTarget = nodeTarget
		self.activationTime = activationTime	#last activation time (used to calculate recency)	#not currently used
		self.spatioTemporalIndex = spatioTemporalIndex	#creation time (not used by biological implementation)		#for biologicalImplementation: interpret as dendriticDistance - generate a unique dendritic distance for the synapse (to ensure the spikes from previousConceptNodes refer to this particular spatioTemporalIndex)
		#if(biologicalImplementation):
		self.weight = weight	#for biologicalImplementation: interpret connection as unique synapse

