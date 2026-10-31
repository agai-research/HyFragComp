"""Minimal reverse-mode autograd on NumPy arrays, plus Adam."""
import numpy as np

_state = {"grad": True}
DTYPE = np.float32


class no_grad:
    def __enter__(self):
        self.prev = _state["grad"]
        _state["grad"] = False

    def __exit__(self, *a):
        _state["grad"] = self.prev


class T:
    """Tensor node holding data, gradient and a backward closure."""
    __slots__ = ("d", "g", "_b", "_p", "rg")

    def __init__(self, d, rg=False, p=(), b=None):
        self.d = np.asarray(d, dtype=DTYPE)
        self.g = None
        self.rg = rg
        self._p = p
        self._b = b

    def backward(self):
        order, seen = [], set()

        def visit(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for q in n._p:
                visit(q)
            order.append(n)

        visit(self)
        self.g = np.ones_like(self.d)
        for n in reversed(order):
            if n._b is not None and n.g is not None:
                n._b(n.g)


def _mk(data, parents, back):
    rg = _state["grad"] and any(p.rg for p in parents)
    if not rg:
        return T(data)
    return T(data, True, tuple(p for p in parents if p.rg), back)


def _acc(t, g):
    if not t.rg:
        return
    t.g = g if t.g is None else t.g + g


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def param(shape, scale=None, rng=None, zeros=False, ones=False):
    if zeros:
        return T(np.zeros(shape), True)
    if ones:
        return T(np.ones(shape), True)
    rng = rng or np.random.default_rng(0)
    scale = scale if scale is not None else np.sqrt(2.0 / shape[0])
    return T(rng.normal(0, scale, shape), True)


def add(a, b):
    out = a.d + b.d

    def back(g):
        _acc(a, _unbroadcast(g, a.d.shape))
        _acc(b, _unbroadcast(g, b.d.shape))
    return _mk(out, (a, b), back)


def mul(a, b):
    out = a.d * b.d

    def back(g):
        _acc(a, _unbroadcast(g * b.d, a.d.shape))
        _acc(b, _unbroadcast(g * a.d, b.d.shape))
    return _mk(out, (a, b), back)


def scale(a, c):
    def back(g):
        _acc(a, g * c)
    return _mk(a.d * c, (a,), back)


def matmul(a, w):
    """a [..., k] @ w [k, m]"""
    out = a.d @ w.d

    def back(g):
        k, m = w.d.shape
        _acc(a, g @ w.d.T)
        _acc(w, a.d.reshape(-1, k).T @ g.reshape(-1, m))
    return _mk(out, (a, w), back)


def relu(a):
    m = a.d > 0

    def back(g):
        _acc(a, g * m)
    return _mk(a.d * m, (a,), back)


def concat(ts, axis=-1):
    out = np.concatenate([t.d for t in ts], axis=axis)
    sizes = [t.d.shape[axis] for t in ts]

    def back(g):
        parts = np.split(g, np.cumsum(sizes)[:-1], axis=axis)
        for t, pg in zip(ts, parts):
            _acc(t, pg)
    return _mk(out, tuple(ts), back)


def gather(a, idx):
    """a [N, d], idx integer array of any shape -> [*idx.shape, d]"""
    out = a.d[idx]

    def back(g):
        ga = np.zeros_like(a.d)
        np.add.at(ga, idx, g)
        _acc(a, ga)
    return _mk(out, (a,), back)


def reshape(a, shape):
    def back(g):
        _acc(a, g.reshape(a.d.shape))
    return _mk(a.d.reshape(shape), (a,), back)


def softmax_masked(s, mask):
    """softmax over the last axis; mask True = keep"""
    z = np.where(mask, s.d, -1e9)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z) * mask
    p = e / (e.sum(axis=-1, keepdims=True) + 1e-12)

    def back(g):
        gs = p * (g - (g * p).sum(axis=-1, keepdims=True))
        _acc(s, gs)
    return _mk(p, (s,), back)


def wsum(att, v):
    """att [..., K], v [..., K, d] -> [..., d]"""
    out = np.einsum("...k,...kd->...d", att.d, v.d)

    def back(g):
        _acc(att, np.einsum("...d,...kd->...k", g, v.d))
        _acc(v, np.einsum("...k,...d->...kd", att.d, g))
    return _mk(out, (att, v), back)


def layernorm(x, gamma, beta, eps=1e-5):
    mu = x.d.mean(axis=-1, keepdims=True)
    var = x.d.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x.d - mu) * inv
    out = xh * gamma.d + beta.d

    def back(g):
        n = x.d.shape[-1]
        _acc(gamma, _unbroadcast(g * xh, gamma.d.shape))
        _acc(beta, _unbroadcast(g, beta.d.shape))
        gx = g * gamma.d
        gi = inv * (gx - gx.mean(axis=-1, keepdims=True) - xh * (gx * xh).mean(axis=-1, keepdims=True))
        _acc(x, gi)
    return _mk(out, (x, gamma, beta), back)


def l2norm(x, eps=1e-8):
    n = np.sqrt((x.d ** 2).sum(axis=-1, keepdims=True)) + eps
    out = x.d / n

    def back(g):
        _acc(x, (g - out * (g * out).sum(axis=-1, keepdims=True)) / n)
    return _mk(out, (x,), back)


def ce_logits(logits, labels):
    """mean cross-entropy over rows"""
    z = logits.d - logits.d.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    n = len(labels)
    loss = -np.log(p[np.arange(n), labels] + 1e-12).mean()

    def back(g):
        gl = p.copy()
        gl[np.arange(n), labels] -= 1
        _acc(logits, gl / n * g)
    return _mk(np.float32(loss), (logits,), back)


def matmul_t(a, b):
    """a [n, d] @ b[m, d].T -> [n, m] (both tensors)"""
    out = a.d @ b.d.T

    def back(g):
        _acc(a, g @ b.d)
        _acc(b, g.T @ a.d)
    return _mk(out, (a, b), back)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = [np.zeros_like(q.d) for q in params]
        self.v = [np.zeros_like(q.d) for q in params]
        self.t = 0

    def zero(self):
        for q in self.p:
            q.g = None

    def step(self):
        self.t += 1
        for i, q in enumerate(self.p):
            if q.g is None:
                continue
            g = q.g + self.wd * q.d
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            q.d -= self.lr * mh / (np.sqrt(vh) + self.eps)
