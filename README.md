# HFNLPpy

### Author

Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

### Description

Hopfield Natural Language Processing (HFNLP) for Python - experimental

### License

MIT License

### Installation
```
conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3 [required for HFNLPpy_biologicalSimulationPropagateVectorised]
conda install nltk
conda install spacy
python3 -m spacy download en_core_web_md
conda install networkx [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_biologicalSimulationDraw]
pip install matplotlib==2.2.3 [required for HFNLPpy_hopfieldGraphDraw/HFNLPpy_biologicalSimulationDraw]
pip install yattag [required for HFNLPpy_biologicalSimulationXML]
```

### Execution
```
source activate anntf2
python3 HFNLPpy_main.py
```
