"""Shared immutable repository (base hypergraph, vectors, HNSW) and the per-method mutable world."""
import heapq
import pickle
from collections import defaultdict
from dataclasses import replace

import numpy as np

from . import data, enc
from .allset import AllSet, Packed, PATTERNS
from .index import HNSW, Index
from .model import HGView, Onto, subst_ok
from .util import path
from .data import frag_from_dict, edge_from_dict


class Models:
    """Encoders shared by all methods of one process: sentence encoder, AllSet, precomputed fragment vectors."""

    def __init__(self, cfg, seed=0):
        from . import train
        self.cfg = cfg
        self.alpha = cfg["encoder"]["alpha"]
        train.run(cfg, seed)
        self.senc = enc.sentence_encoder(cfg, seed)
        self.allset = AllSet.from_file(path("results", "ckpt", "allset_%d.npz" % seed))
        z = np.load(path("results", "cache", "emb_%d.npz" % seed))
        self.fids = [str(x) for x in z["fids"]]
        self.row = {f: i for i, f in enumerate(self.fids)}
        self.h0, self.zstr = z["h0"], z["zstr"]
        self.dim = self.h0.shape[1] * 2
        self._vec = None

    def vec(self, fid):
        i = self.row[fid]
        return enc.fuse(self.h0[i], self.zstr[i], self.alpha)

    def all_vec(self):
        if self._vec is None:
            self._vec = enc.fuse(self.h0, self.zstr, self.alpha)
        return self._vec

    def sem(self, texts):
        return self.senc.encode(texts)

    def goal_vectors(self, goals, edges):
        """e_g, z^str(g) for each sub-goal, through the same AllSet encoder (goal hypergraph A+(P))."""
        e = self.senc.encode([g.text for g in goals]).astype(np.float32)
        ix = {g.gid: i for i, g in enumerate(goals)}
        el = [([ix[a] for a in tl], [ix[b] for b in hd], PATTERNS[p]) for tl, hd, p in edges]
        P = Packed(len(goals), el, cap_set=6, cap_inc=10)
        z = enc.encode_graph(self.allset, e, P)
        return e, z

    def local_embed(self, text, nbr_texts, edges_loc):
        """Inductive embedding of one new fragment from its neighbourhood (one forward pass)."""
        txt = [text] + nbr_texts
        h0 = self.senc.encode(txt).astype(np.float32)
        P = Packed(len(txt), edges_loc, cap_set=6, cap_inc=10)
        z = enc.encode_graph(self.allset, h0, P)
        return h0[0], z[0]


class Repo:
    """First `size` fragments of the main dataset (hot procedures first) with a cached HNSW over Z."""

    def __init__(self, cfg, models, size=None):
        self.cfg, self.models = cfg, models
        size = size or cfg["repo"]["default_size"]
        m = data.main_data()
        pids, n, ents = [], 0, set()
        for p in m["procedures"]:
            add = 9 + (0 if p["iso3"] in ents else 1)
            if n + add > size and pids:
                break
            pids.append(p["pid"])
            ents.add(p["iso3"])
            n += add
        self.pids = pids
        self.size = n
        self.base = data.load_repo(pids)
        self.fids = list(self.base.frags)
        self.jur = {f: f.split(".")[0] for f in self.fids}
        self.index = self._index()

    def _index(self):
        ic = self.cfg["index"]
        key = "hnsw_%d_%d_%.3f_%d_%d.pkl" % (self.size, self.models.h0.shape[1], self.models.alpha, ic["M"], ic["ef_construction"])
        f = path("results", "cache", key)
        if f.exists():
            return pickle.loads(f.read_bytes())
        V = self.models.all_vec()
        h = HNSW(V.shape[1], ic["M"], ic["ef_construction"], seed=0, cap=len(self.fids) + 16)
        for fid in self.fids:
            h.add(V[self.models.row[fid]], fid)
        f.write_bytes(pickle.dumps(h))
        return h


class World:
    """Mutable state of one method: fragment overlay, overlay index, record store, clock and event queue."""

    def __init__(self, repo, cfg, models):
        self.repo, self.cfg, self.models = repo, cfg, models
        self.hg = HGView(repo.base)
        self.onto = Onto()
        self.fidx = Index(repo.index, models.dim)
        self.t = 0
        self.vecs = {}              # vectors of fragments added after the base was built
        self.records = {}
        self.rev_log = defaultdict(list)
        self.q_nu = []              # (nu_hi, fid) min-heap of window ends
        self.lam = defaultdict(float)   # EWMA update rate per fragment
        self.updates = 0
        self.last_upd = {}
        self.busy = set()
        self.suspect = set()        # fragment ids whose dependants are suspect
        for f in repo.base.frags.values():
            if f.nu[1] < 10 ** 6:
                heapq.heappush(self.q_nu, (f.nu[1], f.fid))

    def vec(self, fid):
        v = self.vecs.get(fid)
        return v if v is not None else self.models.vec(fid)

    def now(self):
        return self.t


    # ------------------------------------------------------------------ change events
    def add_with_edges(self, f, edges):
        """Insert a new fragment with its hyperedges and embed it inductively from its neighbourhood (one forward pass)."""
        self.hg.add_fragment(f)
        for e in edges:
            self.hg.add_edge(e)
        nbr = []
        for e in edges:
            for x in list(e.tail) + list(e.head):
                if x != f.fid and x not in nbr and self.hg.frag(x) is not None:
                    nbr.append(x)
        nbr = nbr[:10]
        ix = {f.fid: 0}
        for k, x in enumerate(nbr):
            ix[x] = k + 1
        el = []
        for e in edges:
            tl = [ix[v] for v in e.tail if v in ix]
            hd = [ix[v] for v in e.head if v in ix]
            if tl and hd:
                el.append((tl, hd, PATTERNS[e.pat]))
        h0, z = self.models.local_embed(f.text, [self.hg.frag(x).text for x in nbr], el)
        v = enc.fuse(h0, z, self.models.alpha)
        self.vecs[f.fid] = v
        self.fidx.extra.drop(f.fid)
        self.fidx.extra.add(v, f.fid)
        if f.nu[1] < 10 ** 6:
            heapq.heappush(self.q_nu, (f.nu[1], f.fid))

    def _update(self, new, out, cnt):
        old = self.hg.frag(new.fid)
        sub = old is not None and subst_ok(new, old, self.onto)
        cnt["subst"] += 10
        self.hg.revise(new)
        r = self.rate(new.fid)
        self.lam[new.fid] = (1 - self.cfg["memory"]["lambda_ewma"]) * self.lam[new.fid] + self.cfg["memory"]["lambda_ewma"] * r
        self.updates += 1
        out["changed"].append(new.fid)
        out["subst"][new.fid] = bool(sub)
        if new.nu[1] < 10 ** 6:
            heapq.heappush(self.q_nu, (new.nu[1], new.fid))
        if new.fid in self.vecs or new.fid in self.models.row:
            pass

    def rate(self, fid):
        """updates per day observed for the fragment (inverse of the gap to its previous update)"""
        last = self.last_upd.get(fid)
        self.last_upd[fid] = self.t
        return 0.0 if last is None else 1.0 / max(self.t - last, 1.0)

    def advance(self, t):
        """Pop every validity window that ended at or before t."""
        out = []
        while self.q_nu and self.q_nu[0][0] <= t:
            hi, fid = heapq.heappop(self.q_nu)
            f = self.hg.frag(fid)
            if f is not None and f.nu[1] == hi:
                out.append(fid)
        self.t = max(self.t, t)
        return out

    def apply(self, ev, cnt):
        """Apply one change event to the fragment overlay; returns the set of changed, expired and added fragments."""
        ty = ev["type"]
        self.t = max(self.t, ev["t"])
        out = {"type": ty, "changed": [], "subst": {}, "expired": [], "added": [], "gone": []}
        if ty == "update":
            self._update(frag_from_dict(ev["new"]), out, cnt)
        elif ty == "withdraw":
            self.hg.withdraw(ev["fid"])
            out["changed"].append(ev["fid"])
            out["gone"].append(ev["fid"])
            self.updates += 1
        elif ty == "window":
            f = self.hg.frag(ev["fid"])
            if f is not None:
                self._update(replace(f, nu=(f.nu[0], int(ev["nu_hi"])), rev=f.rev + 1), out, cnt)
                out["subst"][ev["fid"]] = False
        elif ty == "regulation":
            for ch in ev["changes"]:
                sub = self.apply(dict(ch, t=ev["t"]), cnt)
                for k in ("changed", "added", "gone", "expired"):
                    out[k] += sub[k]
                out["subst"].update(sub["subst"])
        elif ty == "tick":
            out["expired"] = self.advance(ev["t"])
        for a in ev.get("adds", []):
            f = frag_from_dict(a["frag"])
            self.add_with_edges(f, [edge_from_dict(x) for x in a["edges"]])
            out["added"].append(f.fid)
        return out
