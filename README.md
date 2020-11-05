# SAPPHIRE: Simple Aligner for Phrasal Paraphrase with Hierarchical Representation (Beta)

**SAPPHIRE** is a simple monolingual phrase aligner based on word embeddings.

We explain the details of SAPPHIRE in the following paper.
[[PDF]](https://www.aclweb.org/anthology/2020.lrec-1.847.pdf)
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
This tool is designed for a pre-trained model of [fastText](https://fasttext.cc/).
(Of course, it is easy to replace the word embedding.)


## Requirements

- Python 3.6 or newer
- NumPy & SciPy
- fasttext


## Installation

1. Download the pre-trained model of fastText
(or prepare your model of fastText) and move it to *model* directory.
```
$ curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip
$ unzip wiki-news-300d-1M-subword.bin.zip
$ mv wiki-news-300d-1M-subword.bin model/
```

### Docker
1. Build the Docker image:
```
$ docker build -t sapphire .
```
2. Run a container:
```
$ docker run -it --rm -v ${PWD}/model:/work/model sapphire:latest /bin/bash
# python
>>> from sapphire import Sapphire
```

### Local installation
1. Install requirements:
```
$ pip install -r requirements.txt
```
2. Install SAPPHIRE using `develop` option
(that allows you to add scripts for other word representations):
```
$ python setup.py develop
```


## Usage

### Interactive mode
```
$ python run_sapphire.py model/wiki-news-300d-1M-subword.bin
```
To stop SAPPHIRE, enter `Ctrl-C` when inputting a sentence.

### Usage of the SAPPHIRE module
```
>>> import fasttext
>>> from sapphire import Sapphire
>>> model = fasttext.FastText.load_model(path_to_your_model)
>>> aligner = Sapphire(model)
```
If you change the hyper-parameters,
```
>>> aligner.set_params(lambda_=0.6, delta=0.6, alpha=0.01, hungarian=False)
```
After preparing a **tokenized** sentence pair
(`tokenized_sentence_a: list` and `tokenized_sentence_b: list`),
```
>>> _, alignment = aligner(tokenized_sentence_a, tokenized_sentence_b)
>>> alignment
[(1, 3, 2, 3), (8, 9, 5, 6), (13, 13, 8, 8), (27, 27, 9, 9)]
```

- Phrase pair <img src="https://render.githubusercontent.com/render/math?math={(x,y)}">
is represented as
<img src="https://render.githubusercontent.com/render/math?math={(x_\text{start},x_\text{end},y_\text{start},y_\text{end})}">.
- Outputs of SAPPHIRE are 1-indexed alignments.
