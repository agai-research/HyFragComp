"""Semantic similarity pieces of the case-based baseline: AHP weights, TF-IDF, SimHash and the text tokenizer."""
import hashlib
import math
import re
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text, min_len=2, stop=True):
    """Lower-cased word tokens without stop words and very short fragments."""
    out = []
    for w in _WORD.findall(text.lower()):
        if len(w) < min_len or (stop and w in ENGLISH_STOP_WORDS):
            continue
        out.append(w)
    return out


# ---------------------------------------------------------------------- AHP
def ahp(matrix, ri, cr_max):
    """Weights from a pairwise comparison matrix: column-normalise, sum rows, normalise; then the consistency test."""
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]
    norm = A / A.sum(axis=0, keepdims=True)
    w = norm.sum(axis=1)
    w = w / w.sum()
    lam = float(np.mean((A @ w) / w))
    ci = (lam - n) / (n - 1)
    cr = ci / ri
    assert cr < cr_max, "AHP judgement matrix fails the consistency test (CR=%.4f)" % cr
    return {"w": w, "lam": lam, "ci": ci, "cr": cr}


def fusion_weights(cfg_cbr):
    """AHP order is (DeBERTa, SimHash, TF-IDF, MCS); alpha is the semantic share and beta renormalises it."""
    r = ahp(cfg_cbr["ahp_matrix"], cfg_cbr["ahp_ri"], cfg_cbr["ahp_cr_max"])
    w = r["w"]
    alpha = float(w[0] + w[1] + w[2])
    beta = {"deberta": float(w[0] / alpha), "simhash": float(w[1] / alpha), "tfidf": float(w[2] / alpha)}
    return {"alpha": alpha, "beta": beta, "w": [float(x) for x in w], "lam": r["lam"], "ci": r["ci"], "cr": r["cr"]}


# ---------------------------------------------------------------------- TF-IDF
def tf(doc):
    """TF_ij = n_ij / sum_k n_kj for one tokenised document."""
    c = Counter(doc)
    tot = sum(c.values())
    return {t: n / tot for t, n in c.items()} if tot else {}


def idf(n_docs, df):
    """IDF_i = log(|D| / |{j : t_i in d_j}|)."""
    return math.log(n_docs / df)


def tfidf_vectors(docs):
    """Plain reference weights for a small corpus (list of token lists): one dict per document."""
    n = len(docs)
    df = Counter(t for d in docs for t in set(d))
    return [{t: v * idf(n, df[t]) for t, v in tf(d).items()} for d in docs]


class TfIdf:
    """TF-IDF cosine of a query against stored documents; the corpus D is the stored documents plus the query."""

    def __init__(self, docs=()):
        self.vocab, self.df = {}, []
        rows, cols, vals = [], [], []
        self.n = 0
        for i, d in enumerate(docs):
            for t, v in tf(d).items():
                j = self.vocab.setdefault(t, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(v)
            self.n += 1
        V = len(self.vocab)
        self.T = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n, V))
        self.T2 = self.T.multiply(self.T).tocsr()
        self.df = np.asarray((self.T > 0).sum(axis=0)).ravel().astype(float)
        N = self.n + 1
        self.idf0 = np.log(N / np.maximum(self.df, 1.0))
        self.base2 = self.T2 @ (self.idf0 ** 2) if V else np.zeros(self.n)

    def sims(self, query):
        """Cosine between the query and every stored document."""
        q = tf(query)
        if self.n == 0 or not q:
            return np.zeros(self.n)
        N = self.n + 1
        known = [(self.vocab[t], v) for t, v in q.items() if t in self.vocab]
        unseen = [v for t, v in q.items() if t not in self.vocab]
        qnorm2 = sum((v * idf(N, 1)) ** 2 for v in unseen)
        if known:
            ix = np.array([i for i, _ in known])
            qv = np.array([v for _, v in known])
            idfq = np.log(N / (self.df[ix] + 1.0))
            qnorm2 += float(np.sum((qv * idfq) ** 2))
            dot = self.T[:, ix] @ (qv * idfq * idfq)
            corr = self.T2[:, ix] @ (idfq ** 2 - self.idf0[ix] ** 2)
        else:
            dot = np.zeros(self.n)
            corr = np.zeros(self.n)
        norm2 = np.maximum(self.base2 + corr, 0.0)
        den = np.sqrt(norm2) * math.sqrt(qnorm2)
        out = np.zeros(self.n)
        ok = den > 1e-12
        out[ok] = np.asarray(dot).ravel()[ok] / den[ok]
        return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------- SimHash
class SimHash:
    """Fingerprint of `bits` bits from weighted token shingles; similarity is one minus the normalised Hamming distance."""

    def __init__(self, bits=64, shingle=2):
        assert bits % 8 == 0 and bits > 0
        self.bits, self.shingle = bits, shingle
        self._h = {}

    def _feat_bits(self, f):
        v = self._h.get(f)
        if v is None:
            raw = hashlib.blake2b(f.encode(), digest_size=self.bits // 8).digest()
            v = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype(np.int32) * 2 - 1
            self._h[f] = v
        return v

    def features(self, doc):
        c = Counter(doc)
        for n in range(2, self.shingle + 1):
            for i in range(len(doc) - n + 1):
                c[" ".join(doc[i:i + n])] += 1
        return c

    def fingerprint(self, doc):
        acc = np.zeros(self.bits, dtype=np.int64)
        for f, w in self.features(doc).items():
            acc += w * self._feat_bits(f)
        return (acc > 0).astype(np.uint8)

    def sim(self, a, b):
        return 1.0 - float(np.count_nonzero(a != b)) / self.bits

    def sims(self, M, q):
        """Similarity of one fingerprint q to the rows of M."""
        if len(M) == 0:
            return np.zeros(0)
        return 1.0 - np.count_nonzero(M != q[None, :], axis=1) / self.bits
