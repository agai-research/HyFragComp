"""HNSW index over unit vectors (inner product) with metadata-filtered search."""
import heapq
import math
import numpy as np


class HNSW:
    def __init__(self, dim, M=12, ef_construction=64, seed=0, cap=1024):
        self.dim, self.M, self.efc = dim, M, ef_construction
        self.mL = 1.0 / math.log(max(M, 2))
        self.rng = np.random.default_rng(seed)
        self.v = np.zeros((cap, dim), dtype=np.float32)
        self.keys = []
        self.nb = [[]]  # nb[level][id] -> list of neighbour ids
        self.lvl = []
        self.entry = -1
        self.top = -1

    def __len__(self):
        return len(self.keys)

    def _search_layer(self, q, eps, ef, lvl):
        visited = set(eps)
        s0 = {e: float(self.v[e] @ q) for e in eps}
        cand = [(-s, e) for e, s in s0.items()]
        heapq.heapify(cand)
        res = [(s, e) for e, s in s0.items()]
        heapq.heapify(res)
        nbl = self.nb[lvl]
        while cand:
            ns, c = heapq.heappop(cand)
            if len(res) >= ef and -ns < res[0][0]:
                break
            nb = [x for x in nbl[c] if x not in visited]
            if not nb:
                continue
            visited.update(nb)
            ss = self.v[nb] @ q
            for x, sx in zip(nb, ss.tolist()):
                if len(res) < ef or sx > res[0][0]:
                    heapq.heappush(cand, (-sx, x))
                    heapq.heappush(res, (sx, x))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted(res, reverse=True)

    def _prune(self, i, lvl):
        mmax = self.M * 2 if lvl == 0 else self.M
        lst = self.nb[lvl][i]
        if len(lst) > mmax:
            sims = self.v[lst] @ self.v[i]
            keep = np.argsort(-sims)[:mmax]
            self.nb[lvl][i] = [lst[j] for j in keep]

    def add(self, vec, key=None):
        i = len(self.keys)
        if i >= len(self.v):
            self.v = np.vstack([self.v, np.zeros_like(self.v)])
        self.v[i] = vec
        self.keys.append(key if key is not None else i)
        lv = int(-math.log(max(self.rng.random(), 1e-12)) * self.mL)
        self.lvl.append(lv)
        while len(self.nb) <= lv:
            self.nb.append([[] for _ in range(i)])
        for l in range(len(self.nb)):
            while len(self.nb[l]) <= i:
                self.nb[l].append([])
        if self.entry < 0:
            self.entry, self.top = i, lv
            return i
        ep = [self.entry]
        for l in range(self.top, lv, -1):
            ep = [self._search_layer(vec, ep, 1, l)[0][1]]
        for l in range(min(self.top, lv), -1, -1):
            found = self._search_layer(vec, ep, self.efc, l)
            chosen = [e for _, e in found[:self.M]]
            self.nb[l][i] = list(chosen)
            for e in chosen:
                self.nb[l][e].append(i)
                self._prune(e, l)
            ep = [e for _, e in found]
        if lv > self.top:
            self.entry, self.top = i, lv
        return i

    def add_batch(self, vecs, keys):
        for v, k in zip(vecs, keys):
            self.add(v, k)

    def knn(self, q, k, ef=None, pred=None):
        """Filtered search; widens ef until k hits pass `pred` (keys), then falls back to a scan."""
        n = len(self.keys)
        if n == 0:
            return []
        ef = max(ef or 32, 2 * k)
        while True:
            ep = [self.entry]
            for l in range(self.top, 0, -1):
                ep = [self._search_layer(q, ep, 1, l)[0][1]]
            found = self._search_layer(q, ep, min(ef, n), 0)
            hits = [(s, self.keys[i]) for s, i in found if pred is None or pred(self.keys[i])]
            if len(hits) >= k or ef >= n:
                break
            ef *= 2
        if len(hits) < k and pred is not None:
            hits = self.exact(q, k, pred)
        return hits[:k]

    def exact(self, q, k, pred=None):
        n = len(self.keys)
        sims = self.v[:n] @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            if pred is None or pred(self.keys[i]):
                out.append((float(sims[i]), self.keys[i]))
                if len(out) >= k:
                    break
        return out


class Extra:
    """Small exact side index for vectors added after the shared HNSW was built."""

    def __init__(self, dim):
        self.v = np.zeros((0, dim), dtype=np.float32)
        self.keys = []

    def add(self, vec, key):
        self.v = np.vstack([self.v, vec[None].astype(np.float32)])
        self.keys.append(key)

    def drop(self, key):
        if key in self.keys:
            i = self.keys.index(key)
            self.v = np.delete(self.v, i, axis=0)
            self.keys.pop(i)

    def knn(self, q, k, pred=None):
        if not self.keys:
            return []
        sims = self.v @ q
        out = [(float(s), key) for s, key in zip(sims, self.keys) if pred is None or pred(key)]
        out.sort(reverse=True)
        return out[:k]


class Index:
    """Shared base HNSW plus a per-world exact overlay."""

    def __init__(self, base, dim):
        self.base, self.extra = base, Extra(dim)

    def knn(self, q, k, ef=None, pred=None):
        a = self.base.knn(q, k, ef, pred) if self.base is not None else []
        b = self.extra.knn(q, k, pred)
        return sorted(a + b, reverse=True)[:k]
