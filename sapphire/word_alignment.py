import numpy as np
from itertools import product
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment


class WordEmbedding(object):

    def __init__(self):
        pass

    def __call__(self, words):
        pass


class FastTextVectorize(WordEmbedding):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dim = model.get_dimension()

    def __call__(self, words):
        vector = []

        if words:
            for word in words:
                vector.append(self.model.get_word_vector(word.lower()))
        else:
            vector.append(np.zeros(self.dim))

        return np.array(vector)


def get_similarity_matrix(src_vectors, trg_vectors):
    len_src = len(src_vectors)
    len_trg = len(trg_vectors)
    sim_matrix = np.zeros((len_src, len_trg))

    for (src_idx, src_vec), (trg_idx, trg_vec) in product(
        enumerate(src_vectors), enumerate(trg_vectors)
    ):
        sim_matrix[src_idx][trg_idx] = 1 - cosine(src_vec, trg_vec)

    return sim_matrix


class WordAlign(object):

    def __init__(self, lambda_, use_hungarian):
        self.lambda_ = lambda_
        self.use_hungarian = use_hungarian

    def __call__(self, sim_matrix):
        return self.align(sim_matrix=sim_matrix)

    def set_params(self, lambda_, use_hungarian):
        self.lambda_ = lambda_
        self.use_hungarian = use_hungarian

    @staticmethod
    def _hungarian_assign(sim_matrix):
        cost_matrix = - sim_matrix
        aligned_src, aligned_trg = linear_sum_assignment(cost_matrix)
        alignments = [(s, t) for s, t in zip(aligned_src, aligned_trg)]

        return alignments

    @staticmethod
    def _grow_diag_final(sim_matrix):

        def _grow_diag():
            point_added = False
            for src, trg in product(range(len_src), range(len_trg)):
                if not align_matrix[src][trg]:
                    continue
                for ns, nt in neighbors:
                    if src + ns < 0 or src + ns >= len_src \
                            or trg + nt < 0 or trg + nt >= len_trg:
                        continue
                    if (not np.any(align_matrix[src + ns, :])
                        or not np.any(align_matrix[:, trg + nt])) \
                            and union_matrix[src + ns][trg + nt]:
                        align_matrix[src + ns][trg + nt] = 1
                        point_added = True
            if point_added:
                _grow_diag()

        def _final(matrix):
            for src, trg in product(range(len_src), range(len_trg)):
                if (not np.any(align_matrix[src, :])
                    or not np.any(align_matrix[:, trg])) \
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
        neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        _grow_diag()
        _final(src2trg)
        _final(trg2src)

        alignments = []
        for s, t in product(range(len_src), range(len_trg)):
            if align_matrix[s][t]:
                alignments.append((s, t))

        return alignments

    def align(self, sim_matrix):
        if self.use_hungarian:
            alignments = self._hungarian_assign(sim_matrix)
        else:
            alignments = self._grow_diag_final(sim_matrix)

        return [(s + 1, t + 1) for s, t in alignments
                if sim_matrix[s][t] >= self.lambda_]  # 1-index alignment
