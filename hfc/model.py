"""Core data structures: ontology, fragments, hyperedges, hypergraph views, goals, contexts."""
from dataclasses import dataclass, field
from typing import Optional

INF_DAY = 10 ** 7

CONFLICTS = [("no_valid_permit", "valid_permit_held")]


class Onto:
    """Atoms 'a:p' are specialisations of the generic atom 'a'."""

    def __init__(self):
        self.conf = {}
        for a, b in CONFLICTS:
            self.conf[a] = b
            self.conf[b] = a
        self._cache = {}

    def close(self, atoms):
        key = atoms if isinstance(atoms, frozenset) else frozenset(atoms)
        c = self._cache.get(key)
        if c is None:
            out = set(key)
            for a in key:
                if ":" in a:
                    out.add(a.split(":", 1)[0])
            c = frozenset(out)
            if len(self._cache) < 200000:
                self._cache[key] = c
        return c

    def consistent(self, atoms):
        return not any(self.conf.get(a) in atoms for a in atoms if a in self.conf)

    def entails(self, avail, need):
        """need is derivable from avail (closure under generalisation) and no conflicting atom is required"""
        cl = self.close(avail)
        if not need <= cl:
            return False
        return not any(self.conf.get(a) in cl for a in need if a in self.conf)


@dataclass(frozen=True)
class Fragment:
    fid: str
    text: str
    pre: frozenset
    post: frozenset
    inp: frozenset
    out: frozenset
    roles: frozenset
    dmin: int
    dmax: int
    ftype: str
    nu: tuple
    rev: int = 1
    kind: str = "atm"
    supp: tuple = ()
    level: int = 0
    proc: str = ""
    step: str = ""
    variant: str = ""
    guard: str = ""
    prio: int = 0


@dataclass(frozen=True)
class Edge:
    eid: str
    tail: tuple
    head: tuple
    pat: str
    guard: str = ""
    route: str = ""


@dataclass(frozen=True)
class Ctx:
    atoms: frozenset
    data: frozenset
    roles: frozenset
    deadline: int = 60
    jur: str = ""


def subst_ok(new, old, onto):
    """new is a substitutable revision of old (Definition of substitutability)."""
    return (onto.entails(old.pre, new.pre) and onto.entails(new.post, old.post)
            and new.inp <= old.inp and old.out <= new.out and old.roles <= new.roles
            and new.dmin <= old.dmin and new.dmax >= old.dmax)


class Hypergraph:
    """Immutable base repository shared by all worlds."""

    def __init__(self):
        self.frags = {}
        self.edges = {}
        self.by_head = {}
        self.by_tail = {}

    def add_fragment(self, f):
        self.frags[f.fid] = f

    def add_edge(self, e):
        self.edges[e.eid] = e
        for v in e.head:
            self.by_head.setdefault(v, []).append(e.eid)
        for v in e.tail:
            self.by_tail.setdefault(v, []).append(e.eid)


class HGView:
    """Mutable overlay: revisions, withdrawals, composites and extra edges of one method's world."""

    def __init__(self, base):
        self.base = base
        self.cur = {}
        self.hist = {}
        self.gone = set()
        self.edge_add = {}
        self.edge_del = set()
        self.add_head = {}
        self.add_tail = {}
        self.nu_over = {}

    # fragments
    def frag(self, fid):
        if fid in self.gone:
            return None
        f = self.cur.get(fid)
        return f if f is not None else self.base.frags.get(fid)

    def frag_at(self, fid, rev):
        for f in self.hist.get(fid, ()):
            if f.rev == rev:
                return f
        b = self.base.frags.get(fid)
        return b if b is not None and b.rev == rev else None

    def revise(self, new):
        fid = new.fid
        if fid not in self.hist:
            b = self.base.frags.get(fid)
            self.hist[fid] = [b] if b is not None else []
        self.hist[fid].append(new)
        self.cur[fid] = new

    def add_fragment(self, f):
        self.cur[f.fid] = f
        self.hist.setdefault(f.fid, []).append(f)

    def withdraw(self, fid):
        self.gone.add(fid)

    def alive(self, fid, t):
        f = self.frag(fid)
        return f is not None and f.nu[0] <= t < f.nu[1]

    # edges
    def edge(self, eid):
        if eid in self.edge_del:
            return None
        e = self.edge_add.get(eid)
        return e if e is not None else self.base.edges.get(eid)

    def add_edge(self, e):
        self.edge_del.discard(e.eid)
        self.edge_add[e.eid] = e
        for v in e.head:
            lst = self.add_head.setdefault(v, [])
            if e.eid not in lst:
                lst.append(e.eid)
        for v in e.tail:
            lst = self.add_tail.setdefault(v, [])
            if e.eid not in lst:
                lst.append(e.eid)

    def del_edge(self, eid):
        self.edge_del.add(eid)

    def edges_head(self, fid):
        ids = list(dict.fromkeys(list(self.base.by_head.get(fid, ())) + self.add_head.get(fid, [])))
        return [e for e in (self.edge(i) for i in ids) if e is not None and fid in e.head]

    def edges_tail(self, fid):
        ids = list(dict.fromkeys(list(self.base.by_tail.get(fid, ())) + self.add_tail.get(fid, [])))
        return [e for e in (self.edge(i) for i in ids) if e is not None and fid in e.tail]

    def all_fids(self):
        s = set(self.base.frags) | set(self.cur)
        return [f for f in s if f not in self.gone]

    def stamps_ok(self, fid, rev, onto):
        """current revision of fid is substitutable for the stamped one"""
        cur, old = self.frag(fid), self.frag_at(fid, rev)
        if cur is None or old is None:
            return False
        return cur.rev == rev or subst_ok(cur, old, onto)


@dataclass
class Goal:
    gid: int
    key: tuple
    text: str
    omega: frozenset
    slots: dict = field(default_factory=dict)
    act: str = ""
    obj: str = ""


@dataclass
class GoalGraph:
    goals: list
    edges: list  # (tuple of pred gids, (gid,), pat)

    def preds(self, gid):
        out = set()
        for tl, hd, _ in self.edges:
            if gid in hd:
                out.update(tl)
        return out

    def order(self):
        """topological order of gids"""
        seen, out = set(), []
        ids = [g.gid for g in self.goals]
        pend = list(ids)
        while pend:
            prog = False
            for g in list(pend):
                if self.preds(g) <= seen:
                    out.append(g)
                    seen.add(g)
                    pend.remove(g)
                    prog = True
            if not prog:
                out.extend(pend)
                break
        return out
