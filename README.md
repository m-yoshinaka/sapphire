# SAPPHIRE: Simple Aligner for Phrasal Paraphrase with Hierarchical Representation (Beta)

**SAPPHIRE** is a simple monolingual phrase aligner based on word embeddings.


## Description

SAPPHIRE depends only on a pre-trained word embeddings. 
Therefore, it is easily transferable to specific domains and different languages.  
This library is designed for a pre-trained model of [fastText](https://fasttext.cc/).
But it is easy to replace the model.


## Requirements

- Python 3.5 or newer
- NumPy & SciPy
- fasttext


## Installation (for fastText version)

1. Install requirements  
After cloning this repository, go to the root directory and install requirements.
```
$ pip install -r requirements.txt
```

2. Install SAPPHIRE  
Installation with `develop` option allows you to add scripts for other word representations.
```
$ python setup.py develop
```


3. Download the pre-trained model of fastText (or prepare your model of fastText) and move the model to *model* directory.
```
$ curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip  
$ unzip wiki-news-300d-1M-subword.bin.zip
$ mkdir model
$ mv wiki-news-300d-1M-subword.bin model/
```


## Usage

### Interactive mode
```
$ python run_sapphire.py
```
To stop SAPPHIRE, enter `exit` when inputting a sentence.

### Usage of the SAPPHIRE module
```
>>> from sapphire import Sapphire
>>> aligner = Sapphire()
```
After preparing a tokenized sentence pair (`tokenized_sentence1: list` and `tokenized_sentence2: list`),
```
>>> alignment = aligner.align(tokenized_sentence1, tokenized_sentence2)

>>> print(alignment)
1,2,3-2,3 8,9-5,6 13-8 27-9
```
Output format: 1-indexed alignment
