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
	def __init__(self, nodeSource, nodeTarget, spatioTemporalIndex, activationTime):
		#primary vars;
		self.nodeSource = nodeSource
		self.nodeTarget = nodeTarget
		self.spatioTemporalIndex = spatioTemporalIndex	#creation time
		self.activationTime = activationTime	#last activation time (used to calculate recency)	#not currently used
		
