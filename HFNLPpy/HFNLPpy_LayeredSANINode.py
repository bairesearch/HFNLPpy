"""HFNLPpy_LayeredSANINode.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Layered SANI Node Classes

"""


import numpy as np
import random

from HFNLPpy_LayeredSANIGlobalDefs import *

def layeredSANINodePropertiesInitialisation(node):
	node.isSANIcompoundNode = False
	node.SANIlayerNeuronID = 0
	node.SANIlayerIndex = 0
	node.SANIactivationState = False
	node.activationStatePartial = False	#highlightPartialActivations only (incomplete)
	#node.SANIsequentialInputActiveList = []	#records if sequential inputs are active
	node.SANIinputNodeList = []	#input nodes (lower layer)
	node.SANIoutputNodeList = []	#output nodes (higher layer)
	node.SANIcontiguousInput = False
	
	#sentence artificial vars (for sentence graph only, do not generalise to network graph);
	node.wMin = node.w
	node.wMax = node.w
	node.noncontinguousInputNodeAboveCentralContentsStart = False
	node.noncontinguousInputNodeAboveCentralContentsEnd = False
	node.noncontinguousInputNodeAbove = None
							

class SANIlayer:
	def __init__(self, layerIndex):
		self.layerIndex = layerIndex
		self.sentenceSANINodeList = []
		self.networkSANINodeList = []
		if(vectoriseComputation):
			#layerBufferSANIconnectionTensor
			self.networkConnectionStrengthDict = {}
		
