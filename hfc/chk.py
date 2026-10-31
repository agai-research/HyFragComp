"""Compatibility predicates chi_*, connection derivation, feasibility checks C1-C5 and workflow-net soundness."""
from collections import defaultdict


class Checks:
    """All deterministic checks of one world; every evaluation is counted in `cnt`."""

    def __init__(self, world, cnt):
        self.w, self.hg, self.onto, self.cnt = world, world.hg, world.onto, cnt

    # ------------------------------------------------------------------ guards and XOR selection
    def guard_true(self, g, ctx):
        self.cnt["guard"] += 1
        return (not g) or (g in ctx.atoms)

    def selected(self, e, ctx):
        """Head members of a XOR hyperedge that the guard selects on this context."""
        mem = [self.hg.frag(x) for x in e.head]
        mem = [f for f in mem if f is not None]
        guarded = [f for f in mem if f.guard and self.guard_true(f.guard, ctx)]
        if guarded:
            top = max(f.prio for f in guarded)
            return {f.fid for f in guarded if f.prio == top}
        return {f.fid for f in mem if not f.guard}

    # ------------------------------------------------------------------ chi predicates
    def chi_fun(self, f, avail):
        self.cnt["chi_fun"] += 1
        return self.onto.entails(avail, f.pre)

    def chi_dat(self, f, data):
        self.cnt["chi_dat"] += 1
        return f.inp <= data

    def chi_rol(self, f, ctx):
        self.cnt["chi_rol"] += 1
        return f.roles <= ctx.roles

    def chi_tmp(self, f, ctx, t, offset):
        """f can run inside its validity window and the path to it fits the deadline."""
        self.cnt["chi_tmp"] += 1
        start = t + offset
        return f.nu[0] <= start and start + f.dmin <= f.nu[1] and offset + f.dmin <= ctx.deadline

    # ------------------------------------------------------------------ connections
    def derive_conns(self, fset, ctx, only=None):
        """Connections of the workflow: every licensing hyperedge whose whole tail is present, keeping maximal tails per fragment."""
        conns, bad = [], []
        for v in (only if only is not None else fset):
            f = self.hg.frag(v)
            if f is None or f.ftype == "ENT":
                continue
            groups = {}
            for e in self.hg.edges_head(v):
                if e.pat in ("der", "loop") or not set(e.tail) <= fset:
                    continue
                if e.pat == "xor" and e.guard and v not in self.selected(e, ctx):
                    continue
                k = frozenset(e.tail)
                cur = groups.get(k)
                key = (len(set(e.head) & fset), e.eid)
                if cur is None or key > (len(set(cur.head) & fset), cur.eid):
                    groups[k] = e
            keep = [k for k in groups if not any(k < o for o in groups)]
            if not keep:
                bad.append(v)
                continue
            for k in keep:
                e = groups[k]
                conns += [(u, v, e.eid) for u in e.tail]
        return conns, bad

    def guard_ok(self, v, ctx):
        """A variant chosen under a guarded XOR hyperedge must be the one the guard selects on this context."""
        self.cnt["guard"] += 1
        for e in self.hg.edges_head(v):
            if e.pat == "xor" and e.guard and v in e.head and v not in self.selected(e, ctx):
                return False
        return True

    @staticmethod
    def _topo(nodes, pred):
        """Iterative DFS over predecessors: a topological order and the edges that close a cycle (ignored)."""
        order, back, state = [], set(), {}
        for root in nodes:
            if state.get(root) == 2:
                continue
            stack = [(root, iter(pred.get(root, ())))]
            state[root] = 1
            while stack:
                x, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    state[x] = 2
                    order.append(x)
                    stack.pop()
                elif state.get(nxt) == 1:
                    back.add((nxt, x))          # nxt -> x closes a cycle
                elif state.get(nxt) is None:
                    state[nxt] = 1
                    stack.append((nxt, iter(pred.get(nxt, ()))))
        return order, back

    @staticmethod
    def ancestors(conns):
        pred = defaultdict(set)
        for u, v, _ in conns:
            pred[v].add(u)
        nodes = set(pred) | {u for ps in pred.values() for u in ps}
        order, back = Checks._topo(sorted(nodes), pred)
        memo = {}
        for x in order:                          # predecessors first
            s = set()
            for p in pred.get(x, ()):
                if (p, x) in back:
                    continue
                s.add(p)
                s |= memo.get(p, set())
            memo[x] = s

        def anc(x):
            return memo.get(x, set())
        anc.cycles = back
        return anc, pred

    def offsets(self, fset, conns):
        """Earliest start offset (days, minimal durations) of every fragment along the connections.
        Connections that close a cycle are ignored here and reported by check_conns as a defect."""
        anc, pred = self.ancestors(conns)
        nodes = sorted(set(fset) | set(pred) | {u for ps in pred.values() for u in ps})
        order, back = Checks._topo(nodes, pred)
        memo = {}
        for x in order:
            o = 0
            for p in pred.get(x, ()):
                if (p, x) in back:
                    continue
                f = self.hg.frag(p)
                o = max(o, memo.get(p, 0) + (f.dmin if f else 0))
            memo[x] = o
        return {x: memo.get(x, 0) for x in fset}

    def conn_ok(self, v, ctx, t, anc, off, ctxd):
        """The four chi predicates for the connections entering v (counted once per connection)."""
        f = self.hg.frag(v)
        a = anc(v)
        avail = set(ctx.atoms)
        data = set(ctxd)
        for x in a:
            g = self.hg.frag(x)
            if g is not None:
                avail |= g.post
                data |= g.out
        ok = self.chi_fun(f, frozenset(avail)) & self.chi_dat(f, frozenset(data)) & self.chi_rol(f, ctx) & self.chi_tmp(f, ctx, t, off[v])
        return ok

    def check_conns(self, fset, conns, ctx, t, touched=None):
        """C3 on the listed fragments (all when None); returns failing fragment ids with the failing predicate."""
        anc, _ = self.ancestors(conns)
        off = self.offsets(fset, conns)
        fails = [(v, ["cycle"]) for _u, v in sorted(anc.cycles)]
        ent_in = defaultdict(list)
        for u, v, _ in conns:
            ent_in[v].append(u)
        for v in (touched if touched is not None else list(ent_in)):
            if v not in ent_in:
                continue
            f = self.hg.frag(v)
            a = anc(v)
            avail = set(ctx.atoms)
            data = set(ctx.data)
            for x in a:
                g = self.hg.frag(x)
                if g is not None:
                    avail |= g.post
                    data |= g.out
            n = len(ent_in[v])
            self.cnt["chi_fun"] += n - 1
            self.cnt["chi_dat"] += n - 1
            self.cnt["chi_rol"] += n - 1
            self.cnt["chi_tmp"] += n - 1
            r = []
            if not self.chi_fun(f, frozenset(avail)):
                r.append("chi_fun")
            if not self.chi_dat(f, frozenset(data)):
                r.append("chi_dat")
            if not self.chi_rol(f, ctx):
                r.append("chi_rol")
            if not self.chi_tmp(f, ctx, t, off[v]):
                r.append("chi_tmp")
            if not self.guard_ok(v, ctx):
                r.append("guard")
            if r:
                fails.append((v, r))
        return fails

    # ------------------------------------------------------------------ C1-C5
    def c1_cover(self, graph, cover, fset):
        miss = []
        for g in graph.goals:
            fs = cover.get(g.gid, ())
            post = set()
            for x in fs:
                f = self.hg.frag(x)
                if f is not None and x in fset:
                    post |= self.onto.close(f.post)
            if not g.omega <= post:
                miss.append(g.gid)
        return miss

    def c2_connected(self, fset, conns):
        ents = [x for x in fset if self.hg.frag(x) is not None and self.hg.frag(x).ftype == "ENT"]
        if len(ents) != 1:
            return False
        succ = defaultdict(set)
        for u, v, _ in conns:
            succ[u].add(v)
        seen, st = {ents[0]}, [ents[0]]
        while st:
            for y in succ[st.pop()]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        return seen >= set(fset)

    def c4_conform(self, fset, conns, only=None):
        """every connection is licensed by a rule whose whole tail is in the workflow"""
        bad = []
        for u, v, eid in conns:
            if only is not None and v not in only:
                continue
            self.cnt["c4"] += 1
            e = self.hg.edge(eid)
            if e is None or u not in e.tail or v not in e.head or not set(e.tail) <= fset:
                bad.append((u, v))
        return bad

    # ------------------------------------------------------------------ C5 soundness
    def net(self, fset, conns, loops=1):
        """Workflow-net unfolding: fragment = transition, connection = place, AND-join on all incoming places."""
        T, P = {}, set()
        succ, pred = defaultdict(list), defaultdict(list)
        uniq, seen_uv = [], set()
        for u, v, eid in conns:
            if (u, v) not in seen_uv:
                seen_uv.add((u, v))
                uniq.append((u, v, eid))
        conns = uniq
        for u, v, _ in conns:
            succ[u].append(v)
            pred[v].append(u)
        i, o = "i", "o"
        P |= {i, o}
        ents = [x for x in fset if self.hg.frag(x) is not None and self.hg.frag(x).ftype == "ENT"]
        outp = defaultdict(set)
        pl = {}
        for u, v, eid in conns:
            e = self.hg.edge(eid)
            p = "px:%s:%s" % (eid, u) if (e is not None and e.pat == "xor") else "p:%s>%s" % (u, v)
            pl[(u, v)] = p
            P.add(p)
            outp[u].add(p)
        term = [x for x in fset if not succ[x]]
        for x in term:
            p = "p:%s>end" % x
            P.add(p)
            outp[x].add(p)
        for x in fset:
            pre = {pl[(u, x)] for u in pred[x]} if pred[x] else ({i} if x in ents else set())
            T["t:" + x] = (pre, set(outp[x]))
        T["t:end"] = ({"p:%s>end" % x for x in term}, {o})
        # bounded LOOP unfolding: a rejected submission re-enters the loop head once per allowed iteration
        for e in self._loops(fset):
            tail, head = e.tail[0], e.head[0]
            if tail in fset and head in fset and loops > 0:
                chain = self._chain(head, tail, succ)
                q = "p:loop:%s" % tail
                P.add(q)
                T["t:%s:rej" % tail] = (set(outp[tail]), {q})
                prev = q
                for k, x in enumerate(chain):
                    r = "p:loop:%s:%d" % (x, k)
                    P.add(r)
                    T["t:loop:%s" % x] = ({prev}, {r})
                    prev = r
                T["t:loop:back"] = ({prev}, set(outp[tail]))
        return T, P, i, o

    def _loops(self, fset):
        out = []
        for x in fset:
            for e in self.hg.edges_tail(x):
                if e.pat == "loop" and e.guard and e.head and e.head[0] in fset:
                    out.append(e)
        return out

    @staticmethod
    def _chain(head, tail, succ):
        out, cur = [head], head
        while cur != tail and succ[cur]:
            cur = succ[cur][0]
            out.append(cur)
        return out

    def soundness(self, fset, conns, cap=60000):
        """Soundness per independent branch: branches that meet only at the entry fragment are put in parallel by an
        AND-split and an AND-join, and sound workflow nets stay sound under that composition."""
        hg = self.hg
        ents = {x for x in fset if hg.frag(x) is not None and hg.frag(x).ftype == "ENT"}
        parent = {x: x for x in fset if x not in ents}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v, _ in conns:
            if u in parent and v in parent:
                parent[find(u)] = find(v)
        comp = defaultdict(set)
        for x in parent:
            comp[find(x)].add(x)
        if len(comp) <= 1:
            return self._sound_one(fset, conns, cap)
        total = 0
        for members in comp.values():
            sub = frozenset(members | ents)
            sc = [c for c in conns if c[1] in members]
            ok, n = self._sound_one(sub, sc, cap)
            total += n
            if not ok:
                return False, total
        return True, total

    def _sound_one(self, fset, conns, cap=60000):
        """Reduction to a single transition certifies soundness; otherwise bounded reachability decides it."""
        T, P, i, o = self.net(fset, conns)
        steps = 0
        T = {t: (set(a), set(b)) for t, (a, b) in T.items()}
        ch = True
        while ch:
            ch = False
            ins = defaultdict(set)
            outs = defaultdict(set)
            for t, (a, b) in T.items():
                for p in a:
                    outs[p].add(t)
                for p in b:
                    ins[p].add(t)
            # self-loop transitions
            for t, (a, b) in list(T.items()):
                if a == b and a:
                    del T[t]
                    ch = True
                    steps += 1
            if ch:
                continue
            # parallel transitions
            seen = {}
            for t, (a, b) in list(T.items()):
                k = (frozenset(a), frozenset(b))
                if k in seen:
                    del T[t]
                    ch = True
                    steps += 1
                else:
                    seen[k] = t
            if ch:
                continue
            # series transitions
            for p in list(ins):
                if p in (i, o):
                    continue
                if len(ins[p]) == 1 and len(outs[p]) == 1:
                    t1, t2 = next(iter(ins[p])), next(iter(outs[p]))
                    if t1 != t2 and t1 in T and t2 in T and T[t2][0] == {p}:
                        a1, b1 = T[t1]
                        b2 = T[t2][1]
                        T[t1] = (a1, (b1 - {p}) | b2)
                        del T[t2]
                        ch = True
                        steps += 1
                        break
            if ch:
                continue
            # series places
            for t, (a, b) in list(T.items()):
                if len(a) == 1 and len(b) == 1:
                    p, q = next(iter(a)), next(iter(b))
                    if p != q and p not in (i, o) and len(outs[p]) == 1 and len(ins[q]) == 1 and q != o and q != i:
                        for t2, (a2, b2) in list(T.items()):
                            T[t2] = ({p if x == q else x for x in a2}, {p if x == q else x for x in b2})
                        ch = True
                        steps += 1
                        break
            if ch:
                continue
            # parallel places
            keyp = defaultdict(list)
            for p in set(ins) | set(outs):
                if p in (i, o):
                    continue
                keyp[(frozenset(ins[p]), frozenset(outs[p]))].append(p)
            for k, ps in keyp.items():
                if len(ps) > 1:
                    for p in ps[1:]:
                        for t2, (a2, b2) in list(T.items()):
                            T[t2] = (a2 - {p}, b2 - {p})
                    ch = True
                    steps += len(ps) - 1
                    break
        self.cnt["sound"] += steps + len(T)
        if len(T) == 1:
            a, b = next(iter(T.values()))
            if a == {i} and b == {o}:
                return True, steps
        return self._reach(T, i, o, cap)

    def _reach(self, T, i, o, cap):
        """Bounded exploration of the (reduced) net: option to complete, proper completion, no dead transition."""
        if any(not a for a, b in T.values()):
            return False, 0
        names = sorted(T)
        pl = sorted({p for a, b in T.values() for p in a | b})
        ix = {p: k for k, p in enumerate(pl)}
        tr = [(tuple(ix[p] for p in T[t][0]), tuple(ix[p] for p in T[t][1])) for t in names]
        start = tuple(1 if p == i else 0 for p in pl)
        end = tuple(1 if p == o else 0 for p in pl)
        seen = {start: []}
        fired = set()
        stack = [start]
        while stack:
            m = stack.pop()
            for k, (a, b) in enumerate(tr):
                if all(m[p] >= 1 for p in a):
                    n = list(m)
                    for p in a:
                        n[p] -= 1
                    for p in b:
                        n[p] += 1
                    n = tuple(n)
                    fired.add(k)
                    if max(n) > 1 and n != end:
                        pass
                    if n not in seen:
                        if len(seen) >= cap:
                            return False, len(seen)
                        seen[n] = []
                        stack.append(n)
                    seen[m].append(n)
        for m, nx in seen.items():
            if m[ix[o]] >= 1 and m != end:
                return False, len(seen)
        # option to complete: end reachable from every marking
        rev = defaultdict(list)
        for m, nx in seen.items():
            for n in nx:
                rev[n].append(m)
        ok, st = {end}, [end]
        while st:
            for m in rev[st.pop()]:
                if m not in ok:
                    ok.add(m)
                    st.append(m)
        return (len(ok) == len(seen) and len(fired) == len(tr)), len(seen)

    # ------------------------------------------------------------------ whole feasibility
    def feasibility(self, graph, cover, fset, conns, ctx, t, touched=None, incremental=False):
        """C1-C5 verdicts; with `incremental` only the touched fragments are re-checked for C3/C4."""
        d = {}
        d["C1"] = not self.c1_cover(graph, cover, fset)
        d["C2"] = self.c2_connected(fset, conns)
        d["C3"] = not self.check_conns(fset, conns, ctx, t, touched if incremental else None)
        d["C4"] = not self.c4_conform(fset, conns, set(touched) if incremental and touched is not None else None)
        s, _ = self.soundness(fset, conns)
        d["C5"] = s
        return d
