# SAPPHIRE: Simple Aligner for Phrasal Paraphrase with Hierarchical Representation (Beta)

**SAPPHIRE** is a simple monolingual phrase aligner based on word embeddings.

We explain the details of SAPPHIRE in the following paper.
```
@inproceedings{yoshinaka-etal-2020,
    author      = {Yoshinaka, Masato and Kajiwara, Tomoyuki and Arase, Yuki},
    title       = {SAPPHIRE: Simple Aligner for Phrasal Paraphrase with Hierarchical Representaion},
    booktitle   = {Proceedings of the 12th International Conference on Language Resources and Evaluation (LREC 2020)},
    year        = {2020},
    pages       = {6861--6867},
    url         = {https://www.aclweb.org/anthology/2020.lrec-1.847/}
}
```


## Description

SAPPHIRE depends only on a pre-trained word embedding.
Therefore, it is easily transferable to specific domains and different languages.
This library is designed for a pre-trained model of [fastText](https://fasttext.cc/).
But it is easy to replace the model.


## Requirements

- Python 3.7 or newer
- NumPy & SciPy
- fasttext


## Installation (for fastText version)

1. Install requirements
After cloning this repository, go to the root directory and install requirements.
```
$ pip install -r requirements.txt
```

2. Install SAPPHIRE
Installation with `develop` option allows you to change the parameters and add scripts for other word representations.
```
$ python setup.py develop
```


3. Download the pre-trained model of fastText (or prepare your model of fastText) and move it to *model* directory.
```
$ curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip
$ unzip wiki-news-300d-1M-subword.bin.zip
$ mkdir model
$ mv wiki-news-300d-1M-subword.bin model/
```


## Usage

### Interactive mode
```
$ python run_sapphire.py model/wiki-news-300d-1M-subword.bin
```
To stop SAPPHIRE, enter `EXIT` when inputting a sentence.

### Usage of the SAPPHIRE module
```
>>> from sapphire import Sapphire
>>> aligner = Sapphire()
```
After preparing a **tokenized** sentence pair (`tokenized_sentence_a: list` and `tokenized_sentence_b: list`),
```
>>> result = aligner.align(tokenized_sentence_a, tokenized_sentence_b)
>>> alignment = result.top_alignment[0][0]
>>> print(alignment)
[(1, 3, 2, 3), (8, 9, 5, 6), (13, 13, 8, 8), (27, 27, 9, 9)]
```
phrase pair <img src="https://render.githubusercontent.com/render/math?math={(x, y)}"> : 
<img src="https://render.githubusercontent.com/render/math?math={(x_\text{start}, x_\text{end}, y_\text{start}, y_\text{end})}">
  \# 1-indexed alignment
