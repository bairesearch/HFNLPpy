"""HFNLPpy_LayeredSANIGraph.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see HFNLPpy_main.py

# Usage:
see HFNLPpy_main.py

# Description:
HFNLP Layered SANI Graph - generate layered SANI graph/network based on textual input

"""

import HFNLPpy_LayeredSANI
import HFNLPpy_LayeredSANINode
from HFNLPpy_LayeredSANIGlobalDefs import LayeredSANInumberOfLayersMax

SANIlayerList = []
for layerIndex in range(LayeredSANInumberOfLayersMax):
	SANIlayerList.append(HFNLPpy_LayeredSANINode.SANIlayer(layerIndex))

def generateLayeredSANIGraphSentence(HFconnectionGraphObject, sentenceIndex, tokenisedSentence, sentenceConceptNodeList, networkConceptNodeDict):	
	
	for conceptNode in sentenceConceptNodeList:	
		conceptNode.SANIlayerNeuronID = conceptNode.networkIndex
		conceptNode.SANIlayerIndex = 0

	SANIlayerList[0].networkSANINodeList = list(networkConceptNodeDict.values())
	SANIlayerList[0].sentenceSANINodeList = sentenceConceptNodeList
		
	sentenceSANINodeList = HFNLPpy_LayeredSANI.updateLayeredSANIgraph(HFconnectionGraphObject, networkConceptNodeDict, SANIlayerList, sentenceIndex)
	
	return sentenceSANINodeList
