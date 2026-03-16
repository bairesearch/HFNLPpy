# HFNLPpy

### Author

Richard Bruce Baxter - Copyright (c) 2022-2024 Baxter AI (baxterai.com)

### Description

Hopfield Natural Language Processing (HFNLP) for Python - experimental

### License

MIT License

### Installation
```
Minimal Dependencies:
source activate pytorchsenv
python ANNpt_main.py
pip install networkx
pip install matplotlib
pip install yattag
pip install torch
pip install torch_geometric
pip install nltk spacy
python3 -m spacy download en_core_web_md
python3 -c "import nltk; nltk.download('punkt_tab')"

useAlgorithmDendriticSANI+vectoriseComputation=True:
conda create -n anntf2 python=3.7
source activate anntf2
pip install tensorflow
pip install networkx
pip install matplotlib==2.2.3
pip install yattag
pip install torch
pip install torch_geometric
pip install nltk spacy==2.3.7
	python -m pip install --upgrade "pip<24" "setuptools<68" wheel
	python -m pip install --only-binary=:all: "spacy==2.3.7" "nltk==3.8.1"
python3 -m spacy download en_core_web_md
python3 -c "import nltk; nltk.download('punkt_tab')"
```

### Execution
```
source activate pytorchsenv
python3 HFNLPpy_main.py

source activate anntf2
python3 HFNLPpy_main.py
```
