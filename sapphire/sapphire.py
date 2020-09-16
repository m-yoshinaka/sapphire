from sapphire.word_alignment import (
    FastTextVectorize, WordAlign, get_similarity_matrix
)
from sapphire.phrase_alignment import PhraseExtract, PhraseAlign


class Sapphire(object):

    def __init__(self, model):
        self.vectorizer = FastTextVectorize(model)
        self.set_params()
        self.word_aligner = WordAlign(self.lambda_, self.use_hungarian)
        self.extractor = PhraseExtract()
        self.phrase_aligner = PhraseAlign()

    def set_params(self, lambda_=0.6, delta=0.6, alpha=0.01, hungarian=False):
        self.lambda_ = lambda_
        self.delta = delta
        self.alpha = alpha
        self.use_hungarian = hungarian

    def align(self, tokens_src: list, tokens_trg: list):
        len_src = len(tokens_src)
        len_trg = len(tokens_trg)

        vectors_src = self.vectorizer(tokens_src)
        vectors_trg = self.vectorizer(tokens_trg)

        sim_matrix = get_similarity_matrix(vectors_src, vectors_trg)
        word_alignment = self.word_aligner(sim_matrix)

        phrase_pairs = self.extractor.extract(word_alignment,
                                              vectors_src,
                                              vectors_trg,
                                              self.delta,
                                              self.alpha)
        phrase_alignment = self.phrase_aligner.create_lattice(phrase_pairs,
                                                              len_src,
                                                              len_trg)
        return word_alignment, phrase_alignment
