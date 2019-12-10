from . import setting
from .word_alignment import FastTextVectorize, WordAlign
from .phrase_alignment import PhraseExtract, PhraseAlign


class Sapphire(object):

    def __init__(self):
        self.name = ""

        self.model_path = setting.MODEL_PATH
        self.hungarian = setting.HUGARIAN
        self.gamma = setting.GAMMA
        self.delta = setting.DELTA
        self.alpha = setting.ALPHA

        self.vectorizer = FastTextVectorize(self.model_path)
        self.word_aligner = WordAlign()
        self.extractor = PhraseExtract()
        self.phrase_aligner = PhraseAlign()

    def align(self, tokens_src: list, tokens_trg: list):
        len_src = len(tokens_src)
        len_trg = len(tokens_trg)

        vectors_src = self.vectorizer.vectorize(tokens_src)
        vectors_trg = self.vectorizer.vectorize(tokens_trg)

        sim_matrix = self.word_aligner.similarity_matrix(vectors_src, vectors_trg)
        word_alignment = self.word_aligner.align(sim_matrix, self.gamma, self.hungarian)

        phrase_pairs = self.extractor.extract(word_alignment, vectors_src, vectors_trg, self.delta, self.alpha)
        alignment_lattice = self.phrase_aligner.create_lattice(phrase_pairs, len_src, len_trg)

        return alignment_lattice[0][0]
