"""AllSet-style structural encoder for directed, pattern-typed hyperedges."""
import numpy as np
from . import nn

PATTERNS = {"seq": 0, "and": 1, "xor": 2, "loop": 3, "der": 4}


class Packed:
    """Padded incidence arrays of a hypergraph (nodes 0..n-1, edges 0..E-1)."""

    def __init__(self, n, edges, cap_set=6, cap_inc=8):
        E = len(edges)
        self.n, self.E = n, E
        self.tail = np.full((max(E, 1), cap_set), n, dtype=np.int64)
        self.head = np.full((max(E, 1), cap_set), n, dtype=np.int64)
        self.pat = np.zeros(max(E, 1), dtype=np.int64)
        inc_h = [[] for _ in range(n)]
        inc_t = [[] for _ in range(n)]
        for i, (tl, hd, p) in enumerate(edges):
            tl, hd = list(tl)[:cap_set], list(hd)[:cap_set]
            self.tail[i, :len(tl)] = tl
            self.head[i, :len(hd)] = hd
            self.pat[i] = p
            for v in hd:
                inc_h[v].append(i)
            for v in tl:
                inc_t[v].append(i)
        self.inc_head = np.full((n, cap_inc), E, dtype=np.int64)
        self.inc_tail = np.full((n, cap_inc), E, dtype=np.int64)
        for v in range(n):
            a, b = inc_h[v][:cap_inc], inc_t[v][:cap_inc]
            self.inc_head[v, :len(a)] = a
            self.inc_tail[v, :len(b)] = b


class PMA:
    """Attention pooling with one learned seed and a learned null element."""

    def __init__(self, d, rng):
        self.seed = nn.param((d,), 0.1, rng)
        self.null = nn.param((d,), 0.1, rng)
        self.wq = nn.param((d, d), None, rng)
        self.wk = nn.param((d, d), None, rng)
        self.wv = nn.param((d, d), None, rng)
        self.wo = nn.param((d, d), None, rng)
        self.w1 = nn.param((d, d), None, rng)
        self.w2 = nn.param((d, d), 0.05, rng)
        self.d = d

    def params(self):
        return [self.seed, self.null, self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]

    def __call__(self, feats, mask):
        d = self.d
        shp = feats.d.shape[:-2]
        nul = nn.add(nn.T(np.zeros(shp + (1, d), dtype=nn.DTYPE)), self.null)
        z = nn.concat([feats, nul], axis=-2)
        m = np.concatenate([mask, np.ones(shp + (1,), dtype=bool)], axis=-1)
        q = nn.matmul(nn.reshape(self.seed, (1, d)), self.wq)
        k = nn.matmul(z, self.wk)
        s = nn.scale(nn.reshape(nn.matmul(k, nn.reshape(nn.reshape(q, (d,)), (d, 1))), shp + (z.d.shape[-2],)), 1.0 / np.sqrt(d))
        att = nn.softmax_masked(s, m)
        out = nn.wsum(att, nn.matmul(z, self.wv))
        out = nn.add(nn.matmul(out, self.wo), self.seed)
        return nn.add(out, nn.matmul(nn.relu(nn.matmul(out, self.w1)), self.w2))


class Layer:
    def __init__(self, d, dp, rng):
        self.pma_t, self.pma_h, self.pma_o = PMA(d, rng), PMA(d, rng), PMA(d, rng)
        self.eps = nn.param((len(PATTERNS) + 1, dp), 0.3, rng)
        self.e1 = nn.param((d + dp, d), None, rng)
        self.e2 = nn.param((d, d), 0.05, rng)
        self.v1 = nn.param((2 * d, d), None, rng)
        self.v2 = nn.param((d, d), 0.05, rng)
        self.g = nn.param((d,), ones=True)
        self.b = nn.param((d,), zeros=True)

    def params(self):
        return (self.pma_t.params() + self.pma_h.params() + self.pma_o.params()
                + [self.eps, self.e1, self.e2, self.v1, self.v2, self.g, self.b])


class AllSet:
    def __init__(self, d=96, dp=16, layers=2, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.dp, self.L = d, dp, layers
        self.layers = [Layer(d, dp, rng) for _ in range(layers)]
        self.wz = nn.param((d, d), None, rng)
        self.mask_vec = nn.param((d,), 0.1, rng)

    def params(self):
        out = [self.wz, self.mask_vec]
        for l in self.layers:
            out += l.params()
        return out

    def state(self):
        return [p.d.copy() for p in self.params()]

    def load(self, st):
        for p, v in zip(self.params(), st):
            p.d[...] = v

    def save(self, path):
        np.savez(path, *self.state(), d=self.d, dp=self.dp, L=self.L)

    @staticmethod
    def from_file(path):
        z = np.load(path)
        m = AllSet(int(z["d"]), int(z["dp"]), int(z["L"]))
        m.load([z["arr_%d" % i] for i in range(len(m.params()))])
        return m

    def _layer(self, lay, h, P, chunk):
        n, E, d = P.n, P.E, self.d
        hp = nn.concat([h, nn.T(np.zeros((1, d), dtype=nn.DTYPE))], axis=0)
        if E == 0:
            he_all = nn.T(np.zeros((1, d), dtype=nn.DTYPE))
        else:
            parts = []
            for s in range(0, E, chunk):
                tl = P.tail[s:s + chunk]
                mt = nn.gather(hp, tl)
                me = lay.pma_t(mt, tl < n)
                ep = nn.gather(lay.eps, P.pat[s:s + chunk])
                he = nn.matmul(nn.relu(nn.matmul(nn.concat([me, ep], axis=-1), lay.e1)), lay.e2)
                parts.append(he)
            he_all = parts[0] if len(parts) == 1 else nn.concat(parts, axis=0)
        hep = nn.concat([he_all, nn.T(np.zeros((1, d), dtype=nn.DTYPE))], axis=0)
        outs = []
        for s in range(0, n, chunk):
            ih, it = P.inc_head[s:s + chunk], P.inc_tail[s:s + chunk]
            a = lay.pma_h(nn.gather(hep, ih), ih < E)
            b = lay.pma_o(nn.gather(hep, it), it < E)
            upd = nn.matmul(nn.relu(nn.matmul(nn.concat([a, b], axis=-1), lay.v1)), lay.v2)
            hs = nn.gather(h, np.arange(s, min(s + chunk, n))) if n > chunk else h
            outs.append(nn.layernorm(nn.add(hs, upd), lay.g, lay.b))
        return outs[0] if len(outs) == 1 else nn.concat(outs, axis=0)

    def forward(self, h0, P, masked=None, chunk=4096):
        """h0: [n, d] array; masked: bool array of nodes whose input is replaced by the mask token"""
        x = nn.T(h0)
        if masked is not None and masked.any():
            keep = (~masked).astype(np.float32)[:, None]
            ind = masked.astype(np.float32)[:, None]
            x = nn.add(nn.mul(x, nn.T(keep)), nn.mul(nn.T(ind), self.mask_vec))
        h = x
        for lay in self.layers:
            h = self._layer(lay, h, P, chunk)
        return h

    def zstr(self, h):
        return nn.matmul(h, self.wz)
