from .word_alignment import FastTextVectorize, WordAlign
from .phrase_alignment import PhraseExtract, PhraseAlign


class Sapphire(object):

    def __init__(self, model):
        self.vectorizer = FastTextVectorize(model)
        self.word_aligner = WordAlign()
        self.extractor = PhraseExtract()
        self.phrase_aligner = PhraseAlign()
        self.set_params()

    def set_params(self, lambda_=0.6, delta=0.6, alpha=0.01, hungarian=False):
        self.lambda_ = lambda_
        self.delta = delta
        self.alpha = alpha
        self.hungarian = hungarian

    def align(self, tokens_src: list, tokens_trg: list):
        len_src = len(tokens_src)
        len_trg = len(tokens_trg)

        vectors_src = self.vectorizer.vectorize(tokens_src)
        vectors_trg = self.vectorizer.vectorize(tokens_trg)

        sim_matrix = self.word_aligner.similarity_matrix(vectors_src,
                                                         vectors_trg)
        word_alignment = self.word_aligner.align(sim_matrix,
                                                 self.lambda_,
                                                 self.hungarian)

        phrase_pairs = self.extractor.extract(word_alignment,
                                              vectors_src,
                                              vectors_trg,
                                              self.delta,
                                              self.alpha)
        phrase_alignment = self.phrase_aligner.create_lattice(phrase_pairs,
                                                              len_src,
                                                              len_trg)
        return word_alignment, phrase_alignment
