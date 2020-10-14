import itertools
from collections import defaultdict
import numpy as np
from scipy.spatial import distance


class PhraseExtract(object):

    def __init__(self, delta, alpha):
        self.delta = delta
        self.alpha = alpha

    def __call__(self, word_alignments, vectors_src, vectors_trg):
        return self.extract(word_alignments, vectors_src, vectors_trg)

    def set_params(self, delta, alpha):
        self.delta = delta
        self.alpha = alpha

    @staticmethod
    def _no_additional_point(ss, se, ts, te, matrix):
        """
        Check if there are any more adjacent points to be added.

        Parameters
        ----------
        ss : int
            the index of the start of source phrase
        se : int
            the index of the end of source phrase
        ts : int
            the index of the start of target phrase
        te : int
            the index of the end of target phrase
        """
        len_src, len_trg = matrix.shape

        if ss - 1 >= 0 and np.any(matrix[ss - 1, :][ts:te + 1]):
            return False
        elif se + 1 < len_src and np.any(matrix[se + 1, :][ts:te + 1]):
            return False
        elif ts - 1 >= 0 and np.any(matrix[:, ts - 1][ss:se + 1]):
            return False
        elif te + 1 < len_trg and np.any(matrix[:, te + 1][ss:se + 1]):
            return False
        else:
            return True

    def extract(
        self, word_alignments, vectors_src: np.array, vectors_trg: np.array
    ):
        """
        Extract phrase pairs using the hueristic of phrase-based SMT.

        Parameters
        ----------
        word_alignments : list
            A return value of 'align' method in WordAlign class.
        vectors_src, vectors_trg : np.array
            Matrix of similarities of word embeddings.

        Returns
        -------
        list
            All candidates of phrase alignment.
        """

        phrase_dict = {}
        len_src = len(vectors_src)
        len_trg = len(vectors_trg)

        matrix = np.zeros((len_src, len_trg))
        for s, t in word_alignments:
            matrix[s - 1][t - 1] = 1

        for (src1, trg1), (src2, trg2) in itertools.product(word_alignments,
                                                            word_alignments):
            ss, se = min(src1 - 1, src2 - 1), max(src1 - 1, src2 - 1)
            ts, te = min(trg1 - 1, trg2 - 1), max(trg1 - 1, trg2 - 1)
            # ss, se, ts and te are 0-index at this time

            while True:
                if ss - 1 >= 0 and np.any(matrix[ss - 1, :][ts:te + 1]):
                    ss -= 1
                if se + 1 < len_src and np.any(matrix[se + 1, :][ts:te + 1]):
                    se += 1
                if ts - 1 >= 0 and np.any(matrix[:, ts - 1][ss:se + 1]):
                    ts -= 1
                if te + 1 < len_trg and np.any(matrix[:, te + 1][ss:se + 1]):
                    te += 1

                if self._no_additional_point(ss, se, ts, te, matrix):
                    break

            if (ss + 1, se + 1, ts + 1, te + 1) not in phrase_dict:
                phrase_vec_src = np.array(
                    vectors_src[ss:se + 1]).mean(axis=0)
                phrase_vec_trg = np.array(
                    vectors_trg[ts:te + 1]).mean(axis=0)

                sim = 1 - distance.cosine(phrase_vec_src, phrase_vec_trg)
                sim -= self.alpha / (se - ss + te - ts + 2)
                phrase_dict[(ss + 1, se + 1, ts + 1, te + 1)] = sim

        phrase_pairs = [(k[0], k[1], k[2], k[3], v)
                        for k, v in phrase_dict.items() if v >= self.delta]
        phrase_pairs.sort(key=lambda x: (x[0], x[2], x[1], x[3]))

        return phrase_pairs


class PhraseAlign(object):

    def __init__(self):
        self.name = ''

    def __call__(self, phrase_pairs, len_src, len_trg,
                 prune_k=-1, get_score=False):
        return self.search_for_lattice(phrase_pairs, len_src, len_trg,
                                       prune_k=prune_k, get_score=get_score)

    @staticmethod
    def search_for_lattice(phrase_pairs, len_src: int, len_trg: int,
                           prune_k=-1, get_score=False):
        """
        Construct a lattice of phrase pairs and depth-first search for the
        path with the highest total alignment score.

        Parameters
        ----------
        phrase_pairs : list
            A return value of 'extract' method in PhraseExtract class.
        len_src, len_trg : int
            Length of sentence.

        Returns
        -------
        list
            List of tuples consisting of indexes of phrase pairs
            = one of the phrase alignments
            = the path of the lattice with the highest total alignment score.
        """

        node_list = defaultdict(lambda: defaultdict(list))
        bos_node = {'index': (0, 0, 0, 0),
                    'score': 0, 'next': []}
        eos_node = {'index': (len_src + 1, len_src + 1,
                              len_trg + 1, len_trg + 1),
                    'score': 0, 'next': []}
        node_list[0][0].append(bos_node)
        node_list[len_src + 1][len_trg + 1].append(eos_node)

        def _forward(s, t, start_node, end_node, pairs):
            """Depth-first search for a lattice."""

            path = []
            if start_node == end_node or not pairs:
                return [[sum(alignment_scores)]]

            min_s, _, min_t, _, _ = min(pairs, key=lambda x: (
                (x[0] - s) ** 2 + (x[2] - t) ** 2))
            min_dist = (min_s - s) ** 2 + (min_t - t) ** 2
            nearest_pairs = [p for p in pairs
                             if (p[0] - s) ** 2 + (p[2] - t) ** 2 == min_dist]

            for pair in pairs[len(nearest_pairs):]:
                nearer = False
                for nearest_pair in nearest_pairs:
                    if pair[0] > nearest_pair[1] and pair[2] > nearest_pair[3]:
                        nearer = True
                        break
                if not nearer:
                    nearest_pairs.append(pair)

            if prune_k != -1:
                nearest_pairs = nearest_pairs[:prune_k]

            for next_pair in nearest_pairs:
                ss, se, ts, te, __score = next_pair
                next_node = {'index': (ss, se, ts, te),
                             'score': __score, 'next': []}
                rest_pairs = [p for p in pairs
                              if p[0] > se and p[2] > te]

                checked = False
                for checked_node in node_list[ss][ts]:
                    if next_node['index'] == checked_node['index']:
                        next_node = checked_node
                        checked = True
                        break

                if not checked:
                    node_list[ss][ts].append(next_node)
                if next_node != end_node:
                    alignment_scores.append(next_node['score'])

                for solution in _forward(se + 1, te + 1,
                                         next_node, end_node, rest_pairs):
                    ids = start_node['index']
                    path.append([(ids)] + solution)

                if next_node != end_node:
                    alignment_scores.pop()

            return path

        if not phrase_pairs:
            return ([], 0) if get_score else []

        s_start, s_end, t_start, t_end, score = sorted(
            phrase_pairs, key=lambda x: x[4], reverse=True)[0]
        top_node = {'index': (s_start, s_end, t_start,
                              t_end), 'score': score, 'next': []}
        node_list[s_start][t_start].append(top_node)

        top_index = [top_node['index']]

        prev_pairs = [p for p in phrase_pairs
                      if p[1] < s_start and p[3] < t_start]
        prev_pairs.append((s_start, s_end, t_start, t_end, score))
        next_pairs = [p for p in phrase_pairs
                      if p[0] > s_end and p[2] > t_end]
        next_pairs.append((len_src + 1, len_src + 1,
                           len_trg + 1, len_trg + 1, 0))

        alignment_scores = []  # Initialize the stack of alignment scores
        prev_align = [
            (sol[1:-1], sol[-1]) for sol
            in _forward(1, 1, bos_node, top_node, prev_pairs)
        ]

        alignment_scores = []  # Re-initialize the stack of alignment scores
        next_align = [
            (sol[1:-1], sol[-1]) for sol
            in _forward(s_end + 1, t_end + 1, top_node, eos_node, next_pairs)
        ]

        alignments = []
        for prev_path, next_path in itertools.product(prev_align, next_align):
            concat_path = prev_path[0] + top_index + next_path[0]
            length = len(concat_path)
            score = (prev_path[1] + next_path[1] + score) / length \
                if length != 0 else 0
            alignments.append((concat_path, score))

        alignments.sort(key=lambda x: float(x[1]), reverse=True)

        if get_score:
            return alignments[0]

        return alignments[0][0]  # Return only the top one of phrase alignments
