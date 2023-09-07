# HFNLPpy

### Author

Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

### Description

Hopfield Natural Language Processing (HFNLP) for Python - experimental

### License

MIT License

### Installation
```
conda create -n anntf2 python=3.7
source activate anntf2
pip install tensorflow [required for HFNLPpy_DendriticSANIPropagateVectorised]
pip install networkx [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_DendriticSANIDraw]
pip install matplotlib==2.2.3 [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_DendriticSANIDraw]
pip install yattag [required for HFNLPpy_DendriticSANIXML]
pip install torch [required for HFNLPpy_Scan/SANIHFNLPpy_LayeredSANI:vectoriseComputation]
pip install torch_geometric [required for HFNLPpy_Scan]
pip install nltk spacy==2.3.7
python3 -m spacy download en_core_web_md
pip install benepar [required for SPNLPpy_syntacticalGraphConstituencyParserFormal]
```

### Execution
```
source activate anntf2
python3 HFNLPpy_main.py
```
