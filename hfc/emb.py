"""Sentence encoder: TF-IDF -> LSA -> two-layer projection trained with a contrastive (InfoNCE) objective."""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from . import nn


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


class SentEnc:
    def __init__(self, k_lsa=256, hid=192, dim=96, seed=0):
        self.k_lsa, self.hid, self.dim, self.seed = k_lsa, hid, dim, seed
        self.vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[A-Za-z0-9<>_'-]+")
        self.svd = TruncatedSVD(k_lsa, random_state=seed)
        rng = np.random.default_rng(seed)
        self.w1 = nn.param((k_lsa, hid), None, rng)
        self.b1 = nn.param((hid,), zeros=True)
        self.w2 = nn.param((hid, dim), None, rng)
        self.cache = {}

    def fit_text(self, corpus):
        x = self.vw.fit_transform(corpus)
        self.svd.fit(x)
        self.cache = {}
        return self

    def lsa(self, texts):
        return _unit(self.svd.transform(self.vw.transform(texts))).astype(np.float32)

    def params(self):
        return [self.w1, self.b1, self.w2]

    def _fwd(self, x):
        h = nn.relu(nn.add(nn.matmul(nn.T(x), self.w1), self.b1))
        return nn.l2norm(nn.matmul(h, self.w2))

    def fit_contrastive(self, groups, steps=1500, batch=96, lr=3e-3, temp=0.07, seed=0, log=print, hard=None, drop=0.35):
        """groups: list of lists of texts; texts in one group are paraphrases (positives)."""
        rng = np.random.default_rng(seed)
        allt = [t for g in groups for t in g]
        base = dict(zip(allt, self.lsa(allt)))
        opt = nn.Adam(self.params(), lr=lr)
        ng = len(groups)
        for s in range(steps):
            ids = hard(rng, batch) if hard else rng.choice(ng, batch, replace=False)
            a, b = [], []
            for i in ids:
                g = groups[i]
                j, k = rng.choice(len(g), 2, replace=False)
                a.append(base[g[j]])
                b.append(base[g[k]])
            xa, xb = np.stack(a), np.stack(b)
            xa = xa * (rng.random(xa.shape) > drop) / (1 - drop)
            xb = xb * (rng.random(xb.shape) > drop) / (1 - drop)
            za, zb = self._fwd(xa), self._fwd(xb)
            lg = nn.scale(nn.matmul_t(za, zb), 1.0 / temp)
            lg2 = nn.scale(nn.matmul_t(zb, za), 1.0 / temp)
            loss = nn.add(nn.ce_logits(lg, np.arange(len(ids))), nn.ce_logits(lg2, np.arange(len(ids))))
            opt.zero()
            loss.backward()
            opt.step()
            if (s + 1) % 250 == 0:
                log("  sent-enc step %d loss %.3f" % (s + 1, float(loss.d) / 2))
        self.cache = {}
        return self

    def _raw(self, texts):
        with nn.no_grad():
            return self._fwd(self.lsa(texts)).d

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for s in range(0, len(miss), 4096):
            part = miss[s:s + 4096]
            for t, v in zip(part, self._raw(part)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])
