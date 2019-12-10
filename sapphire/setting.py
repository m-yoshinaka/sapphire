import os
import sys

FASTTEXT_PATH = "model/wiki-news-300d-1M-subword.bin"

HUGARIAN = False  # word alignment option (default: grow-diag-final)

GAMMA = 0.6  # threshold of word alignment candidate score
DELTA = 0.6  # threshold of phrase alignment candidate score
ALPHA = 0.05  # bias for length of phrase

if os.path.exists(FASTTEXT_PATH):
    MODEL_PATH = "model/wiki-news-300d-1M-subword.bin"  # path of pre-trained word embedding model (default: fastText)
else:
    print("Input the path of your pre-trained word embedding model.")
    MODEL_PATH = input("> ")
