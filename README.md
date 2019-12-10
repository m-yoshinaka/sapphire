# SAPPHIRE: Simple Aligner for Phrasal Paraphrase with Hierarchical Representation (Beta)

**SAPPHIRE** is a simple monolingual phrase aligner based on word embeddings.


## Description

SAPPHIRE depends only on a pre-trained word embeddings. 
Therefore, it is easily transferable to specific domains and different languages.  
This library is designed for a pre-trained model of [fastText](https://fasttext.cc/).
But it is easy to replace the model.


## Requirement
- Python 3.5 or newer
- NumPy & SciPy
- fasttext


## Installation (for fastText version)

1. Install requirements  
After cloning this repository, go to the root directory and `pip install -r requirements.txt` or `pipenv install`.

2. Download pre-trained model of fastText or prepare your model of fastText.
```
$ curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip

$ unzip wiki-news-300d-1M-subword.bin.zip
```

3. Move the pre-trained model to *model* directory.


## Usage
```
import sa
```

