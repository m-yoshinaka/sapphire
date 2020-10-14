from .word_alignment import (
    FastTextVectorize, WordAlign, get_similarity_matrix
)
from .phrase_alignment import PhraseExtract, PhraseAlign


class Sapphire(object):
    """
    SAPPHIRE : monolingual phrase aligner

    Attributes
    ----------
    vectorizer : FastTextVectorize
        Vectorize words using fastText (Bojanowski et al., 2017).
    word_aligner : WordAlign
        Align words in two sentences.
    extractor : PhraseExtract
        Extract phrase pairs in two sentences based on word alignment and
        calculate alignment scores of phrase pairs.
    phrase_aligner : PhraseAlign
        Search for a phrase alignment with the highest total alignment score.

    Methods
    -------
    set_params(lambda_=0.6, delta=0.6, alpha=0.01, hungarian=False)
        Set hyper-parameters of SAPPHIRE.
    align(tokens_src, tokens_trg)
        Get word alignment and phrase alignment.
    """

    def __init__(self, model):
        self.vectorizer = FastTextVectorize(model)
        self.lambda_ = 0.6
        self.delta = 0.6
        self.alpha = 0.01
        self.use_hungarian = False
        self.prune_k = -1
        self.get_score = False
        self.word_aligner = WordAlign(self.lambda_, self.use_hungarian)
        self.extractor = PhraseExtract(self.delta, self.alpha)
        self.phrase_aligner = PhraseAlign()

    def __call__(self, tokens_src, tokens_trg):
        return self.align(tokens_src, tokens_trg)

    def set_params(self, lambda_=0.6, delta=0.6, alpha=0.01, hungarian=False,
                   prune_k=-1, get_score=False):
        """
        Set hyper-parameters of SAPPHIRE.

        Details are discussed in the following paper:
        https://www.aclweb.org/anthology/2020.lrec-1.847/ .

        Parameters
        ----------
        lambda_ : float
            Prunes word alignment candidates.
        delta : float
            Prunes phrase alignment candidates.
        alpha : float
            Biases the phrase alignment score based on the lengths of phrases.
        hungarian : bool
            Whether to use the extended Hangarian method to get word alignment.
        """
        self.lambda_ = lambda_
        self.delta = delta
        self.alpha = alpha
        self.use_hungarian = hungarian
        self.prune_k = prune_k
        self.get_score = get_score
        self.word_aligner.set_params(self.lambda_, self.use_hungarian)
        self.extractor.set_params(self.delta, self.alpha)

    def align(self, tokens_src: list, tokens_trg: list):
        """
        Align phrase pairs in two sentences.

        Parameters
        ----------
        tokens_src, tokens_trg : list
            A tokenized sentence represented by a list of words.

        Returns
        -------
        tuple
            (word_alignment, phrase_alignment)
        """
        len_src = len(tokens_src)
        len_trg = len(tokens_trg)

        vectors_src = self.vectorizer(tokens_src)
        vectors_trg = self.vectorizer(tokens_trg)

        sim_matrix = get_similarity_matrix(vectors_src, vectors_trg)
        word_alignment = self.word_aligner(sim_matrix)

        phrase_pairs = self.extractor(word_alignment, vectors_src, vectors_trg)
        phrase_alignment = self.phrase_aligner(phrase_pairs, len_src, len_trg,
                                               prune_k=self.prune_k,
                                               get_score=self.get_score)

        return word_alignment, phrase_alignment
