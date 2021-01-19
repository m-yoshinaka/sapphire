import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from .word_alignment import WordAlign, WordEmbedding, get_similarity_matrix
from .phrase_alignment import PhraseExtract, PhraseAlign


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertEmbedding(WordEmbedding):

    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)

    def __call__(self, words):
        return self.vectorize(words)

    @staticmethod
    def _to_word_vectors(words, subwords, subword_vectors):
        tokenized = []
        word_vectors = []
        i = 0
        for subword, vector in zip(subwords, subword_vectors):
            if '#' not in subword:
                tokenized.append(subword)
                word_vectors.append(vector)
            else:
                tokenized[i - 1] += subword.replace('#', '')
                word_vectors[i - 1] += vector

        if len(words) == len(word_vectors):
            return np.array(word_vectors)

        _words = [w.lower() for w in words]
        tmp = []
        j = 0
        for i, token in enumerate(tokenized):
            if j >= len(_words):
                break
            if token == _words[j]:
                tmp.append(word_vectors[i])
                j += 1
            else:
                for k in range(1, len(tokenized) - i + 1):
                    cand = ''.join(tokenized[i:i + k + 1])
                    if cand == _words[j]:
                        new_vectors = np.array(word_vectors[i:i + k + 1])
                        new_vector = np.mean(new_vectors, axis=0)
                        tmp.append(new_vector)
                        j += 1
                        break

        return np.array(tmp)

    def vectorize(self, words):
        self.model.eval()

        text = '[CLS] ' + ' '.join(words) + ' [SEP]'
        tokenized = self.tokenizer.tokenize(text)

        index_tokens = self.tokenizer.convert_tokens_to_ids(tokenized)
        tokens_tensor = torch.tensor([index_tokens]).to(device)

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor)

        encoded_layers = encoded_layers[0][1:-1].detach().cpu().numpy()
        vectors = self._to_word_vectors(words, tokenized[1:-1],
                                        np.array(encoded_layers))

        return vectors


class SapphireBert(object):

    def __init__(self, model_name='bert-base-uncased'):
        self.vectorizer = BertEmbedding(model_name)
        self.lambda_ = 0.6
        self.delta = 0.6
        self.alpha = 0.01
        self.use_hungarian = False
        self.prune_k = -1
        self.get_score = False
        self.epsilon = None
        self.word_aligner = WordAlign(self.lambda_, self.use_hungarian)
        self.extractor = PhraseExtract(self.delta, self.alpha)
        self.phrase_aligner = PhraseAlign(self.prune_k,
                                          self.get_score,
                                          self.epsilon)

    def __call__(self, tokens_src, tokens_trg):
        return self.align(tokens_src, tokens_trg)

    def set_params(self, lambda_=0.6, delta=0.6, alpha=0.01, hungarian=False,
                   prune_k=-1, get_score=False, epsilon=None):
        self.lambda_ = lambda_
        self.delta = delta
        self.alpha = alpha
        self.use_hungarian = hungarian
        self.prune_k = prune_k
        self.get_score = get_score
        self.epsilon = epsilon
        self.word_aligner.set_params(self.lambda_, self.use_hungarian)
        self.extractor.set_params(self.delta, self.alpha)
        self.phrase_aligner.set_params(self.prune_k,
                                       self.get_score,
                                       self.epsilon)

    def align(self, tokens_src: list, tokens_trg: list):
        try:
            len_src = len(tokens_src)
            len_trg = len(tokens_trg)

            vectors_src = self.vectorizer(tokens_src)
            vectors_trg = self.vectorizer(tokens_trg)

            sim_matrix = get_similarity_matrix(vectors_src, vectors_trg)
            word_alignment = self.word_aligner(sim_matrix)

            phrase_pairs = self.extractor(
                word_alignment, vectors_src, vectors_trg)
            phrase_alignment = self.phrase_aligner(
                phrase_pairs, len_src, len_trg)

        except ValueError:
            word_alignment = []
            if self.get_score:
                phrase_alignment = ([], 0)
            else:
                phrase_alignment = []

        return word_alignment, phrase_alignment
