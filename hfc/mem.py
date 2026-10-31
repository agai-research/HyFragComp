"""Composition records, ephemeral tier, goal-pattern index Z_G, count-min sketch and frequency-gated promotion."""
import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import enc
from .index import HNSW
from .model import Fragment, Edge, Goal, GoalGraph, INF_DAY, Ctx
from . import build


@dataclass
class Rec:
    """Composition record c = (T, W, phi, stamps, nu, n, U, ttl, scope) kept as a materialised view."""
    cid: str
    kind: str                     # rec | cmp
    goals: list
    edges: list
    e: np.ndarray
    z: np.ndarray
    phi: dict
    stamps: dict
    esig: dict
    conns: list
    nu: tuple
    owner: str
    users: set
    scope: str
    n: int = 1
    t0: float = 0.0
    t_last: float = 0.0
    ctx_atoms: frozenset = frozenset()
    suspect: bool = False
    evicted: bool = False
    cmp_fid: str = ""
    iso: str = ""
    slot_types: list = field(default_factory=list)

    def slot_type(self, i):
        return self.slot_types[i]


def slot_sig(g):
    return tuple(sorted(g.slots))


class Memory:
    """Ephemeral tier C_e with TTL, hit counters, requester sets and the index Z_G."""

    def __init__(self, world, cfg):
        self.w, self.cfg = world, cfg
        self.mc = cfg["memory"]
        self.recs = {}
        self.zg = HNSW(world.models.dim, cfg["index"]["M"], cfg["index"]["ef_construction"], seed=1)
        self.by_frag = {}
        self.k_u = self.mc["k_u"]
        self.ttl = self.mc["tau_e"]
        self.ttl_shared = self.mc["shared_ttl"]

    # ------------------------------------------------------------------ insert
    def add(self, r):
        self.recs[r.cid] = r
        a = self.w.models.alpha
        for i, g in enumerate(r.goals):
            self.zg.add(enc.fuse(r.e[i], r.z[i], a), (r.cid, i))
        for fs in r.phi.values():
            for f in fs:
                self.by_frag.setdefault(f, set()).add(r.cid)
        if r.cmp_fid:
            self.by_frag.setdefault(r.cmp_fid, set()).add(r.cid)

    def alive(self, r, t):
        if r.evicted:
            return False
        if r.kind == "rec" and t - r.t_last > (self.ttl_shared if r.scope == "shared" else self.ttl):
            return False
        return r.nu[0] <= t < r.nu[1]

    def recall(self, qvec, k, t, pred_extra=None):
        def pred(key):
            r = self.recs.get(key[0])
            return r is not None and self.alive(r, t) and (pred_extra is None or pred_extra(r))
        return self.zg.knn(qvec, k, ef=max(32, 2 * k), pred=pred)

    def touch(self, r, user, t):
        r.n += 1
        r.t_last = t
        r.users.add(user)
        if r.scope == "u" and len(r.users) >= self.k_u:
            r.scope = "shared"

    def note(self, r, user, t):
        """A requester matched a private record: count it towards the sharing threshold without exposing the record."""
        r.users.add(user)
        r.t_last = t
        if len(r.users) >= self.k_u:
            r.scope = "shared"

    def evict(self, cid):
        r = self.recs.get(cid)
        if r is not None:
            r.evicted = True

    def holding(self, fid):
        return [self.recs[c] for c in self.by_frag.get(fid, ()) if c in self.recs and not self.recs[c].evicted]

    def sizes(self):
        live = [r for r in self.recs.values() if not r.evicted]
        return {"records": sum(r.kind == "rec" for r in live), "cmp_records": sum(r.kind == "cmp" for r in live), "goal_vectors": len(self.zg)}


class CountMin:
    """Count-min sketch with periodic halving of every counter."""

    def __init__(self, width, depth, window, seed=0):
        self.w, self.d, self.W = width, depth, window
        self.c = np.zeros((depth, width), dtype=np.int64)
        self.seen = 0
        self.salt = [hashlib.blake2b(bytes([i, seed]), digest_size=8).digest() for i in range(depth)]

    def _h(self, key):
        b = key.encode()
        return [int.from_bytes(hashlib.blake2b(b, key=s, digest_size=8).digest(), "little") % self.w for s in self.salt]

    def add(self, key):
        for r, h in enumerate(self._h(key)):
            self.c[r, h] += 1

    def est(self, key):
        return int(min(self.c[r, h] for r, h in enumerate(self._h(key))))

    def tick(self):
        self.seen += 1
        if self.seen % self.W == 0:
            self.c >>= 1


def region_sig(fids, conns):
    """Canonical hash of sorted fragment ids and the typed hyperedges among them."""
    s = ",".join(sorted(fids)) + "|" + ",".join(sorted("%s>%s" % (u, v) for u, v, _ in conns if u in fids and v in fids))
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def regions(hg, fset, conns, rmax):
    """Admissible regions of one workflow: hyperedge-closed fork sets, each optionally extended along the chain."""
    blocks = {}
    for f in fset:
        fr = hg.frag(f)
        if fr is None or fr.ftype == "ENT" or fr.kind == "cmp":
            continue
        blocks.setdefault(fr.proc, []).append(fr)
    out = []
    pred, succ = {}, {}
    for u, v, _ in conns:
        pred.setdefault(v, set()).add(u)
        succ.setdefault(u, set()).add(v)
    for P, frs in blocks.items():
        ids = {f.fid for f in frs}
        fork = {f.fid for f in frs if f.step in ("elig", "docs", "extra", "fee")}
        if len(fork) < 2:
            continue
        chain = [f for f in frs if f.step == "appt"]
        tail = [f for f in frs if f.step == "sub"]
        if 2 <= len(fork) <= rmax:
            out.append((P, frozenset(fork)))
    return out


def lift(hg, onto, fids, conns, new_id, text, rev=1):
    """Composite fragment f_R with its DER hyperedge and lifted edges (Definition: lifting)."""
    fr = [hg.frag(x) for x in fids]
    inner = [(u, v) for u, v, _ in conns if u in fids and v in fids]
    post = frozenset().union(*[f.post for f in fr])
    out = frozenset().union(*[f.out for f in fr])
    pre = frozenset().union(*[f.pre for f in fr])
    pre = frozenset(a for a in pre if not any(a in onto.close(f.post) for f in fr))
    inp = frozenset(x for f in fr for x in f.inp if x not in out)
    roles = frozenset().union(*[f.roles for f in fr])
    nu = (max(f.nu[0] for f in fr), min(f.nu[1] for f in fr))
    # critical path over the internal connections
    pred = {}
    for u, v in inner:
        pred.setdefault(v, []).append(u)
    memo = {}

    def lp(x, key):
        if (x, key) in memo:
            return memo[(x, key)]
        f = hg.frag(x)
        v = getattr(f, key) + max([lp(p, key) for p in pred.get(x, [])] or [0])
        memo[(x, key)] = v
        return v
    dmin = max(lp(x, "dmin") for x in fids)
    dmax = max(lp(x, "dmax") for x in fids)
    guard = next((f.guard for f in fr if f.guard), "")
    prio = max(f.prio for f in fr)
    lvl = 1 + max(f.level for f in fr)
    proc = fr[0].proc
    cmp = Fragment(fid=new_id, text=text, pre=pre, post=post, inp=inp, out=out, roles=roles, dmin=dmin, dmax=dmax, ftype="INT",
                   nu=nu, rev=rev, kind="cmp", supp=tuple(sorted(fids)), level=lvl, proc=proc, step="cmp", guard=guard, prio=prio)
    fs = set(fids)
    edges = [Edge(new_id + ".d", tuple(sorted(fids)), (new_id,), "der")]
    seen = set()
    for f in fids:
        for e in hg.edges_head(f):
            if e.pat in ("der", "loop") or e.eid in seen or (set(e.tail) & fs) or e.eid.startswith(new_id + "."):
                continue
            seen.add(e.eid)
            head = tuple(sorted((set(e.head) - fs) | {new_id}))
            edges.append(Edge(new_id + ".in." + e.eid, e.tail, head, e.pat, e.guard, e.route))
        for e in hg.edges_tail(f):
            if e.pat in ("der", "loop") or e.eid in seen or (set(e.head) & fs) or e.eid.startswith(new_id + "."):
                continue
            seen.add(e.eid)
            tail = tuple(sorted((set(e.tail) - fs) | {new_id}))
            edges.append(Edge(new_id + ".out." + e.eid, tail, e.head, e.pat, e.guard, e.route))
    return cmp, edges


def expand_atomic(hg, fids):
    """Atomic fragments behind a cover that may contain composites."""
    out = []
    for f in fids:
        fr = hg.frag(f)
        if fr is not None and fr.kind == "cmp":
            out += list(fr.supp)
        else:
            out.append(f)
    return tuple(dict.fromkeys(out))


def window(hg, fids):
    lo, hi = 0, INF_DAY
    for f in fids:
        fr = hg.frag(f)
        if fr is not None:
            lo, hi = max(lo, fr.nu[0]), min(hi, fr.nu[1])
    return (lo, hi)


def make_rec(world, chk, cid, goals, edges, E, Z, phi, ctx, owner, scope, t, kind="rec", users=None, n=1, cmp_fid=""):
    """Build a composition record from verified material: stamps, rule signatures and window come from the current state."""
    hg = world.hg
    frs = set(x for v in phi.values() for x in v)
    fset = frozenset(frs | {ctx.jur + ".ent"})
    conns, _ = chk.derive_conns(fset, ctx)
    stamps = {}
    for f in frs:
        fr = hg.frag(f)
        if fr is not None:
            stamps[f] = fr.rev
    esig = {}
    for _, _, eid in conns:
        e = hg.edge(eid)
        if e is not None:
            esig[eid] = (e.pat, e.guard)
    gcopy = [Goal(g.gid, g.key, g.text, g.omega, {k: "" for k in g.slots}, g.act, g.obj) for g in goals]
    return Rec(cid=cid, kind=kind, goals=gcopy, edges=[(tuple(a), tuple(b), p) for a, b, p in edges], e=np.asarray(E, dtype=np.float32),
               z=np.asarray(Z, dtype=np.float32), phi={int(k): tuple(v) for k, v in phi.items()}, stamps=stamps, esig=esig,
               conns=conns, nu=window(hg, fset), owner=owner, users=set(users or {owner}), scope=scope, n=n, t0=t, t_last=t,
               ctx_atoms=ctx.atoms, cmp_fid=cmp_fid, iso=ctx.jur)


def rec_from_json(world, chk, models, d, t):
    """Stored record from the dataset: goal texts are encoded through the same encoders as requests."""
    goals = []
    for g in d["goals"]:
        P, step, qual = g["key"]
        goals.append(Goal(g["gid"], (P, step, qual), g["text"], build.goal_omega(P, step), dict(g["slots"]), step, P.split(".")[-1]))
    edges = [(tuple(a), tuple(b), p) for a, b, p in d["edges"]]
    E, Z = models.goal_vectors(goals, edges)
    phi = {int(k): tuple(v) for k, v in d["phi"].items()}
    atoms = ["applicant_present", "jur:" + d["iso"], "valid_permit_held" if d["qual"] == "renew" else "no_valid_permit"]
    if d["prio"]:
        atoms.append("priority_status")
    ctx = Ctx(frozenset(atoms), frozenset(build.DATA_ITEMS), frozenset({"agency", "citizen"}), 90, d["iso"])
    r = make_rec(world, chk, d["cid"], goals, edges, E, Z, phi, ctx, d["user"], d["scope"], d["t"], users=d["users"], n=d["hits"])
    r.t_last = max(d["t"], t)
    return r
