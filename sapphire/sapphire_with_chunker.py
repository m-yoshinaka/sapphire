import re
from nltk_opennlp.taggers import OpenNLPTagger
from nltk_opennlp.chunkers import OpenNLPChunker

from .sapphire import Sapphire
from .word_alignment import get_similarity_matrix


class SapphireWithChunker(Sapphire):

    def __init__(self, model, pyopennlp_dir):
        super().__init__(model)

        path_to_bin = pyopennlp_dir + '/apache-opennlp/bin'
        path_to_tagger = pyopennlp_dir + '/opennlp_models/en-pos-maxent.bin'
        path_to_chunker = pyopennlp_dir + '/opennlp_models/en-chunker.bin'
        self.chunker = Chunker(path_to_bin=path_to_bin,
                               path_to_tagger=path_to_tagger,
                               path_to_chunker=path_to_chunker)

    def align(self, tokens_src, tokens_trg):
        len_src = len(tokens_src)
        len_trg = len(tokens_trg)

        tokens_src, chunks_src = self.chunker(tokens_src)
        tokens_trg, chunks_trg = self.chunker(tokens_trg)

        if tokens_src is None or tokens_trg is None:
            if self.get_score:
                return [], ([], 0)
            else:
                return [], []

        vectors_src = self.vectorizer(tokens_src)
        vectors_trg = self.vectorizer(tokens_trg)

        sim_matrix = get_similarity_matrix(vectors_src, vectors_trg)
        word_alignment = self.word_aligner(sim_matrix)

        phrase_pairs = self.extractor(word_alignment, vectors_src, vectors_trg)
        phrase_pairs = self._filter_out_by_chunk(chunks_src, chunks_trg,
                                                 phrase_pairs)
        phrase_alignment = self.phrase_aligner(phrase_pairs, len_src, len_trg)

        return word_alignment, phrase_alignment

    def _filter_out_by_chunk(self, chunks_src, chunks_trg, phrase_pairs):
        new_phrase_paris = []
        for ss, se, ts, te, score in phrase_pairs:
            is_ok = True
            for cnk in chunks_src:
                if self._conflict_with_chunk(ss, se, cnk):
                    is_ok = False
                    break
            if is_ok:
                for cnk in chunks_trg:
                    if self._conflict_with_chunk(ts, te, cnk):
                        is_ok = False
                        break
            if is_ok:
                new_phrase_paris.append((ss, se, ts, te, score))

        return new_phrase_paris

    @staticmethod
    def _conflict_with_chunk(start, end, chunk):
        if len(chunk) < 1:
            return False
        elif start == end and chunk[0] <= start <= chunk[-1]:
            return True
        elif chunk[0] < start <= chunk[-1]:
            return True
        elif chunk[0] <= end < chunk[-1]:
            return True
        return False


class Chunker(object):

    def __init__(self, path_to_bin, path_to_tagger, path_to_chunker):
        self.tagger = OpenNLPTagger(path_to_bin=path_to_bin,
                                    path_to_model=path_to_tagger)
        self.chunker = OpenNLPChunker(path_to_bin=path_to_bin,
                                      path_to_chunker=path_to_chunker,
                                      use_punc_tag=True)

    def __call__(self, sentence: list):
        return self.chunking(sentence)

    def chunking(self, sentence: list):
        sentence = ' '.join(sentence)
        tokens = self.tagger.tag(sentence)

        try:
            tree = self.chunker.parse(tokens)
        except (TypeError, IndexError, AttributeError):
            return None, None

        chunks = self._tree2chunk(tree)
        chunks = self._chunk2index(chunks)
        tokens = [t[0] for t in tokens]
        return tokens, chunks

    @ staticmethod
    def _tree2chunk(tree):
        tree_text = str(tree).split('\n')
        tree_text = [c.strip() for c in tree_text][1:]
        chunks = []
        for cnk in tree_text[:]:
            if cnk[0] == '(':
                cnk = cnk[1:]
            if cnk[-1] == ')':
                cnk = cnk[:-1]
            cnk = cnk.split()
            cnk = tuple(c for c in cnk if re.fullmatch(r'[A-Z]+', c) is None)
            chunks.append(cnk)
        return chunks

    @ staticmethod
    def _chunk2index(chunks):
        chunk_index = []
        i = 1
        for cnk in chunks:
            c_len = len(cnk)
            index = tuple(j for j in range(i, i + c_len))
            chunk_index.append(index)
            i += c_len
        return chunk_index
