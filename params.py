FASTTEXT_PATH = "wiki-news-300d-1M-subword.bin" # path of pre-trained word embedding model (default: fastText)

HUGARIAN = False # word alignment option (default: grow-diag-final)

GAMMA = 0.5 # threshold of word alignment candidate score
DELTA = 0.5 # threshold of phrase alignment candidate score
ALPHA = 0.05 # bias for length of phrase