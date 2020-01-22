import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


class WordEmbedding(object):

    def __init__(self):
        pass

    def vectorize(self, word: list) -> np.array:
        pass


class FastTextVectorize(WordEmbedding):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def vectorize(self, words: list) -> np.array:
        vector = []

        if words:
            for word in words:
                # vector.append(self.model.get_word_vector(word.lower()))
                vector.append(self.model.get_word_vector(word))
        else:
            vector.append(np.zeros(300))

        return np.array(vector)


class WordAlign(object):

    def __init__(self):
        self.name = ''

    @staticmethod
    def similarity_matrix(vectors_src: np.array, vectors_trg: np.array) -> np.ndarray:
        len_src = len(vectors_src)
        len_trg = len(vectors_trg)

        sim_matrix = np.zeros((len_src, len_trg))

        for (id_src, vec_src), (id_trg, vec_trg) in itertools.product(enumerate(vectors_src), enumerate(vectors_trg)):
            sim_matrix[id_src][id_trg] = 1 - distance.cosine(vec_src, vec_trg)

        return sim_matrix

    @staticmethod
    def _hungarian_assign(sim_matrix):

        len_src, len_trg = sim_matrix.shape
        cost_matrix = np.ones((len_src, len_trg))

        for id_src, id_trg in itertools.product(range(len_src), range(len_trg)):
            cost_matrix[id_src][id_trg] -= sim_matrix[id_src][id_trg]

        aligned_src, aligned_trg = linear_sum_assignment(cost_matrix)
        alignments = [(s, t) for s, t in zip(aligned_src, aligned_trg)]

        return alignments

    @staticmethod
    def _grow_diag_final(sim_matrix):

        def _grow_diag():
            point_added = False
            for src, trg in itertools.product(range(len_src), range(len_trg)):
                if not align_matrix[src][trg]:
                    continue
                for ns, nt in neighbors:
                    if src + ns < 0 or src + ns >= len_src or trg + nt < 0 or trg + nt >= len_trg:
                        continue
                    if (not np.any(align_matrix[src + ns, :]) or not np.any(align_matrix[:, trg + nt])) \
                            and union_matrix[src + ns][trg + nt]:
                        align_matrix[src + ns][trg + nt] = 1
                        point_added = True
            if point_added:
                _grow_diag()

        def _final(matrix):
            for src, trg in itertools.product(range(len_src), range(len_trg)):
                if (not np.any(align_matrix[src, :]) or not np.any(align_matrix[:, trg])) \
                        and matrix[src][trg]:
                    align_matrix[src][trg] = 1

        len_src, len_trg = sim_matrix.shape

        src2trg = np.zeros((len_src, len_trg))
        for s, t in enumerate(np.argmax(sim_matrix, axis=1)):
            src2trg[s][t] = 1

        trg2src = np.zeros((len_src, len_trg))
        for t, s in enumerate(np.argmax(sim_matrix, axis=0)):
            trg2src[s][t] = 1

        align_matrix = np.logical_and(src2trg, trg2src)
        union_matrix = np.logical_or(src2trg, trg2src)
        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        _grow_diag()
        _final(src2trg)
        _final(trg2src)

        alignments = []
        for s, t in itertools.product(range(len_src), range(len_trg)):
            if align_matrix[s][t]:
                alignments.append((s, t))

        return alignments

    def align(self, sim_matrix: np.ndarray, gamma, hungarian) -> list:
        ### 1-index alignment ###
        if hungarian:
            alignments = self._hungarian_assign(sim_matrix)
        else:
            alignments = self._grow_diag_final(sim_matrix)

        return [(s + 1, t + 1) for s, t in alignments if sim_matrix[s][t] >= gamma]
