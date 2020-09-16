import itertools
from collections import defaultdict
import numpy as np
from scipy.spatial import distance


class PhraseExtract(object):

    def __init__(self):
        self.name = ''

    @staticmethod
    def _no_adpoint(s_start, s_end, t_start, t_end, matrix):
        len_src, len_trg = matrix.shape

        if s_start - 1 >= 0 and np.any(matrix[s_start - 1, :][t_start:t_end + 1]):
            return False
        elif s_end + 1 < len_src and np.any(matrix[s_end + 1, :][t_start:t_end + 1]):
            return False
        elif t_start - 1 >= 0 and np.any(matrix[:, t_start - 1][s_start:s_end + 1]):
            return False
        elif t_end + 1 < len_trg and np.any(matrix[:, t_end + 1][s_start:s_end + 1]):
            return False
        else:
            return True

    def extract(self, word_alignments: list, vectors_src: np.array, vectors_trg: np.array,
                delta, alpha) -> list:
        phrase_dict = {}
        len_src = len(vectors_src)
        len_trg = len(vectors_trg)

        matrix = np.zeros((len_src, len_trg))
        for s, t in word_alignments:
            matrix[s - 1][t - 1] = 1

        for (src1, trg1), (src2, trg2) in itertools.product(word_alignments, word_alignments):
            ### s_start, s_end, t_start and t_end are 0-index ###
            s_start, s_end = min(src1 - 1, src2 - 1), max(src1 - 1, src2 - 1)
            t_start, t_end = min(trg1 - 1, trg2 - 1), max(trg1 - 1, trg2 - 1)

            while True:
                if s_start - 1 >= 0 and np.any(matrix[s_start - 1, :][t_start:t_end + 1]):
                    s_start -= 1
                if s_end + 1 < len_src and np.any(matrix[s_end + 1, :][t_start:t_end + 1]):
                    s_end += 1
                if t_start - 1 >= 0 and np.any(matrix[:, t_start - 1][s_start:s_end + 1]):
                    t_start -= 1
                if t_end + 1 < len_trg and np.any(matrix[:, t_end + 1][s_start:s_end + 1]):
                    t_end += 1

                if self._no_adpoint(s_start, s_end, t_start, t_end, matrix):
                    break

            if (s_start + 1, s_end + 1, t_start + 1, t_end + 1) not in phrase_dict:
                phrase_vec_src = np.array(vectors_src[s_start:s_end + 1]).mean(axis=0)
                phrase_vec_trg = np.array(vectors_trg[t_start:t_end + 1]).mean(axis=0)
                sim = 1 - distance.cosine(phrase_vec_src, phrase_vec_trg)

                sim -= alpha / (s_end - s_start + t_end - t_start + 2)

                phrase_dict[(s_start + 1, s_end + 1, t_start + 1, t_end + 1)] = sim

        phrase_pairs = [(k[0], k[1], k[2], k[3], v) for k, v in phrase_dict.items() if v >= delta]
        phrase_pairs.sort(key=lambda x: (x[0], x[2], x[1], x[3]))

        return phrase_pairs


class PhraseAlign(object):

    def __init__(self):
        self.name = ''

    @staticmethod
    def create_lattice(phrase_pairs: list, len_src: int, len_trg: int) -> list:
        node_list = defaultdict(lambda: defaultdict(list))

        bos_node = {'index': (0, 0, 0, 0), 'sim': 0, 'next': []}
        eos_node = {'index': (len_src + 1, len_src + 1, len_trg + 1, len_trg + 1), 'sim': 0, 'next': []}
        node_list[0][0].append(bos_node)
        node_list[len_src + 1][len_trg + 1].append(eos_node)

        def _forward(s, t, start_node, end_node, pairs):
            path = []

            if start_node == end_node or not pairs:
                return [[sum(similarity)]]

            min_s, _, min_t, _, _ = min(pairs, key=lambda x: ((x[0] - s) ** 2 + (x[2] - t) ** 2))
            min_dist = (min_s - s) ** 2 + (min_t - t) ** 2
            nearest_pairs = [p for p in pairs if (p[0] - s) ** 2 + (p[2] - t) ** 2 == min_dist]

            for pair in pairs[len(nearest_pairs):]:
                nearer = False
                for nearest_pair in nearest_pairs:
                    if pair[0] > nearest_pair[1] and pair[2] > nearest_pair[3]:
                        nearer = True
                        break
                if not nearer:
                    nearest_pairs.append(pair)

            for next_pair in nearest_pairs:
                s_start, s_end, t_start, t_end, sim = next_pair
                next_node = {'index': (s_start, s_end, t_start, t_end), 'sim': sim, 'next': []}
                rest_pairs = [p for p in pairs if p[0] > s_end and p[2] > t_end]

                checked = False
                for checked_node in node_list[s_start][t_start]:
                    if next_node['index'] == checked_node['index']:
                        next_node = checked_node
                        checked = True
                        break
                if not checked:
                    node_list[s_start][t_start].append(next_node)

                if next_node != end_node:
                    similarity.append(next_node['sim'])

                for solution in _forward(s_end + 1, t_end + 1, next_node, end_node, rest_pairs):
                    ids = start_node['index']
                    path.append([(ids)] + solution)

                if next_node != end_node:
                    similarity.pop()

            return path

        if not phrase_pairs:
            return [([], 0)]

        _s_start, _s_end, _t_start, _t_end, _sim = sorted(phrase_pairs, key=lambda x: x[4], reverse=True)[0]
        top_node = {'index': (_s_start, _s_end, _t_start, _t_end), 'sim': _sim, 'next': []}
        node_list[_s_start][_t_start].append(top_node)

        top_index = [top_node['index']]

        prev_pairs = [p for p in phrase_pairs if p[1] < _s_start and p[3] < _t_start]
        prev_pairs.append((_s_start, _s_end, _t_start, _t_end, _sim))

        next_pairs = [p for p in phrase_pairs if p[0] > _s_end and p[2] > _t_end]
        next_pairs.append((len_src + 1, len_src + 1, len_trg + 1, len_trg + 1, 0))

        similarity = []
        prev_align = [(sol[1:-1], sol[-1]) for sol in
                        _forward(1, 1, bos_node, top_node, prev_pairs)]

        similarity = []
        next_align = [(sol[1:-1], sol[-1]) for sol in
                        _forward(_s_end + 1, _t_end + 1, top_node, eos_node, next_pairs)]

        alignments = []
        for prev_path, next_path in itertools.product(prev_align, next_align):
            concat_path = prev_path[0] + top_index + next_path[0]
            length = len(concat_path)
            score = (prev_path[1] + next_path[1] + _sim) / length if length != 0 else 0
            alignments.append((concat_path, str(score)))

        alignments.sort(key=lambda x: float(x[1]), reverse=True)

        return alignments[0][0]
