"""Transition systems of CAFC: fragment LTS, context properties, formulas, grounding and the execution domains."""
from collections import deque, namedtuple

CTRL, UNCTRL = "!", "?"

Act = namedtuple("Act", "name kind frag cost")        # an action with its controllability and the fragment it belongs to
Annot = namedtuple("Annot", "P E G")                  # annotation omega = <precondition, effect events, goal>


class StateCap(Exception):
    """The execution domain grew past the configured state cap."""


class Formula:
    """Disjunction of conjunctive clauses; a clause assigns at most one state to each context property."""

    def __init__(self, clauses):
        self.cl = frozenset(frozenset(c) for c in clauses)

    @staticmethod
    def top():
        return Formula([()])

    @staticmethod
    def bottom():
        return Formula([])

    @staticmethod
    def lit(prop, state):
        return Formula([((prop, state),)])

    @staticmethod
    def conj(assign):
        return Formula([tuple(assign.items())])

    def is_false(self):
        return not self.cl

    def or_(self, other):
        return Formula(self.cl | other.cl)

    def and_(self, other):
        out = []
        for a in self.cl:
            for b in other.cl:
                m = a | b
                if len({p for p, _ in m}) == len(m):
                    out.append(m)
        return Formula(out)

    def props(self):
        return {p for c in self.cl for p, _ in c}

    def holds(self, conf):
        """conf maps a property name to its current state"""
        return any(all(conf.get(p) == s for p, s in c) for c in self.cl)

    def restrict(self, static, cnt=None):
        """Evaluate the literals on static properties: drop violated clauses, strip satisfied literals."""
        out = []
        for c in self.cl:
            keep, ok = [], True
            for p, s in c:
                if p in static:
                    if cnt is not None:
                        cnt["ctx_ev"] += 1
                    if static[p] != s:
                        ok = False
                        break
                else:
                    keep.append((p, s))
            if ok:
                out.append(tuple(keep))
        return Formula(out)

    def conflict(self, static):
        """First static property that falsifies the formula (for the first clause), or None"""
        for c in sorted(self.cl, key=lambda x: sorted(x)):
            for p, s in sorted(c):
                if p in static and static[p] != s:
                    return p
            return None
        return None


class LTS:
    """Deterministic fragment f = <S, s0, I, O, R>; I holds the controllable and O the uncontrollable actions."""

    def __init__(self, name, states, s0, acts, trans):
        self.name, self.S, self.s0 = name, tuple(states), s0
        self.acts = {a.name: a for a in acts}
        self.I = frozenset(a.name for a in acts if a.kind == CTRL)
        self.O = frozenset(a.name for a in acts if a.kind == UNCTRL)
        assert not (self.I & self.O) and s0 in self.S
        self.R = {s: [] for s in self.S}
        for s, a, s2 in trans:
            assert s in self.R and s2 in self.R and a in self.acts
            assert all(a != x for x, _ in self.R[s]), "fragment must be deterministic"
            self.R[s].append((a, s2))


class Prop:
    """Context property p = <L, l0, E, T>: states, initial state, events and event-labelled transitions."""

    def __init__(self, name, L, l0, E=(), T=()):
        self.name, self.L, self.l0, self.E = name, tuple(L), l0, frozenset(E)
        self.T = {}
        for l, e, l2 in T:
            assert l in self.L and l2 in self.L and e in self.E
            assert (l, e) not in self.T
            self.T[(l, e)] = l2
        assert l0 in self.L


class Context:
    """A context C = {p1..pn} whose properties have disjoint event sets."""

    def __init__(self, props):
        self.props = list(props)
        seen = set()
        for p in self.props:
            assert not (seen & p.E), "event sets of context properties must be disjoint"
            seen |= p.E
        self.events = seen

    def init(self):
        return {p.name: p.l0 for p in self.props}


def goal_state(goal, prop):
    """State of `prop` named by the (conjunctive) goal formula, or None"""
    if goal is None:
        return None
    for c in goal.cl:
        for p, s in c:
            if p == prop.name:
                return s
    return None


def ground_prop(prop, annot):
    """Definition 8: action-labelled transitions of one context property for one annotated action."""
    m = {}
    ev = annot.E & prop.E
    if ev:
        for (l, e), l2 in prop.T.items():
            if e in ev:
                m[l] = l2
    lg = goal_state(annot.G, prop)
    if lg is not None:
        for l in prop.L:
            m.setdefault(l, lg)
    if not ev and lg is None:
        m = {l: l for l in prop.L}
    return m


class GroundedContext:
    """Grounded context: synchronous product of the grounded properties, guards read on the whole configuration."""

    def __init__(self, ctx, annots, cnt):
        self.cnt = cnt
        self.static = {p.name: p.l0 for p in ctx.props if not p.T}
        self.dyn = [p for p in ctx.props if p.T]
        self.pos = {p.name: j for j, p in enumerate(self.dyn)}
        self.sidx = [{l: i for i, l in enumerate(p.L)} for p in self.dyn]
        self.l0 = tuple(self.sidx[j][p.l0] for j, p in enumerate(self.dyn))
        self.grounded = {}
        self.guard, self.moves, self.locked = {}, {}, {}
        for a, an in annots.items():
            P = an.P.restrict(self.static, cnt)
            if P.is_false():
                self.locked[a] = an.P.conflict(self.static)
                continue
            self.guard[a] = self.compile(P)
            mv = []
            for j, p in enumerate(self.dyn):
                m = ground_prop(p, an)
                self.grounded[(p.name, a)] = m
                cnt["ground"] += len(m)
                if len(m) == len(p.L) and all(k == v for k, v in m.items()):
                    continue
                mv.append((j, {self.sidx[j][k]: self.sidx[j][v] for k, v in m.items()}))
            self.moves[a] = mv

    def compile(self, formula):
        """Clauses over configuration indexes; static literals are evaluated away."""
        f = formula.restrict(self.static, self.cnt)
        return [[(self.pos[p], self.sidx[self.pos[p]][s]) for p, s in c] for c in f.cl]

    def holds(self, comp, conf):
        self.cnt["ctx_ev"] += 1
        return any(all(conf[j] == s for j, s in c) for c in comp)

    def step(self, conf, a):
        """Configuration reached by action a from conf, or None when a is locked or its guard is false."""
        g = self.guard.get(a)
        if g is None:
            return None
        self.cnt["ctx_ev"] += 1
        if not any(all(conf[j] == s for j, s in c) for c in g):
            return None
        new = list(conf)
        for j, mp in self.moves[a]:
            nx = mp.get(conf[j])
            if nx is None:
                return None
            new[j] = nx
        return tuple(new)

    def names(self, conf):
        return {p.name: p.L[conf[j]] for j, p in enumerate(self.dyn)}

    def reachable(self):
        """States of the grounded context alone, over all of its actions."""
        seen, q = {self.l0}, deque([self.l0])
        while q:
            c = q.popleft()
            for a in self.guard:
                n = self.step(c, a)
                if n is not None and n not in seen:
                    seen.add(n)
                    q.append(n)
        return seen


class Sys:
    """An explicit transition system with controllable/uncontrollable actions, as the planner reads it."""

    def __init__(self, R, kind, init, cost=None, frag=None):
        self.R, self.kind, self.init = R, kind, init
        self.cost = cost or {}
        self.frag = frag or {}
        self._pred = None

    @property
    def pred(self):
        if self._pred is None:
            p = {}
            for s, trs in self.R.items():
                for _, s2 in trs:
                    p.setdefault(s2, set()).add(s)
            self._pred = p
        return self._pred

    def size(self):
        return len(self.R)

    def n_trans(self):
        return sum(len(t) for t in self.R.values())


class Domain(Sys):
    """Context-aware execution domain: asynchronous product of the fragments synchronised with the grounded context."""

    def __init__(self, ltss, annots, ctx, cnt, cap):
        self.ltss = list(ltss)
        self.annots = dict(annots)
        self.gc = GroundedContext(ctx, self.annots, cnt)
        kind, cost, frag = {}, {}, {}
        for l in self.ltss:
            for a in l.acts.values():
                kind[a.name], cost[a.name], frag[a.name] = a.kind, a.cost, a.frag
        sidx = [{s: i for i, s in enumerate(l.S)} for l in self.ltss]
        self.rows = [{sidx[i][s]: [(a, sidx[i][s2]) for a, s2 in l.R[s]] for s in l.S} for i, l in enumerate(self.ltss)]
        init = (tuple(sidx[i][l.s0] for i, l in enumerate(self.ltss)), self.gc.l0)
        super().__init__({}, kind, init, cost, frag)
        self.full = 1
        for l in self.ltss:
            self.full *= len(l.S)
        for p in self.gc.dyn:
            self.full *= len(p.L)
        self.cnt, self.cap = cnt, cap
        self._explore()
        self.pruned = self.full - len(self.R)

    def _explore(self):
        R, gc, cnt = self.R, self.gc, self.cnt
        R[self.init] = []
        q = deque([self.init])
        while q:
            s = q.popleft()
            fs, conf = s
            out = R[s]
            for i, rows in enumerate(self.rows):
                for a, s2 in rows[fs[i]]:
                    c2 = gc.step(conf, a)
                    if c2 is None:
                        continue
                    ns = (fs[:i] + (s2,) + fs[i + 1:], c2)
                    out.append((a, ns))
                    cnt["lts_trans"] += 1
                    if ns not in R:
                        if len(R) >= self.cap:
                            raise StateCap(len(R))
                        R[ns] = []
                        q.append(ns)

    def goal_states(self, formula):
        comp = self.gc.compile(formula)
        return {s for s in self.R if self.gc.holds(comp, s[1])}

    def describe(self, s):
        fs, conf = s
        return ({l.name: l.S[fs[i]] for i, l in enumerate(self.ltss)}, self.gc.names(conf))
