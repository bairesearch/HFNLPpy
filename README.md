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
conda install -c tensorflow tensorflow=2.3 [required for HFNLPpy_SANIPropagateVectorised]
conda install nltk
conda install spacy
python3 -m spacy download en_core_web_md
conda install networkx [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_SANIDraw]
pip install matplotlib==2.2.3 [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_SANIDraw]
pip install yattag [required for HFNLPpy_SANIXML]
pip install torch_geometric [required for HFNLPpy_Scan]
```

### Execution
```
source activate anntf2
python3 HFNLPpy_main.py
```
