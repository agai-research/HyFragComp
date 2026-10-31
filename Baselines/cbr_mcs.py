"""Exact maximum common connected subgraph over nodes plus edges, by search on the modular product graph."""
from collections import namedtuple

MCS = namedtuple("MCS", "weight nodes edges sim exact expansions")


class LG:
    """Labelled directed graph; node and edge labels must agree for two elements to match."""

    def __init__(self):
        self.lab, self.kind, self.e, self.nbr = {}, {}, {}, {}

    def add_node(self, nid, label, kind=""):
        if nid not in self.lab:
            self.lab[nid] = label
            self.kind[nid] = kind
            self.nbr[nid] = set()

    def add_edge(self, u, v, label):
        if u == v or u not in self.lab or v not in self.lab:
            return
        self.e[(u, v)] = label
        self.nbr[u].add(v)
        self.nbr[v].add(u)

    def size(self):
        """|G| = nodes + edges."""
        return len(self.lab) + len(self.e)

    def edge_count(self, nodes):
        s = set(nodes)
        return sum(1 for (u, v) in self.e if u in s and v in s)

    def drop(self, nid):
        for v in list(self.nbr[nid]):
            self.e.pop((nid, v), None)
            self.e.pop((v, nid), None)
            self.nbr[v].discard(nid)
        del self.lab[nid], self.kind[nid], self.nbr[nid]

    def shrink(self, cap, order):
        """Keep at most `cap` nodes: remove low-degree nodes of the least informative kinds first."""
        rank = {k: i for i, k in enumerate(order)}
        while len(self.lab) > cap:
            cand = [n for n in self.lab if self.kind[n] in rank]
            if not cand:
                break
            cand.sort(key=lambda n: (rank[self.kind[n]], len(self.nbr[n]), str(n)))
            self.drop(cand[0])


def mcs(g1, g2, budget=10 ** 6):
    """Maximum common connected induced subgraph weighted by nodes + common labelled edges.

    Vertices of the modular product are label-equal node pairs; two vertices are compatible when the pair
    maps distinct nodes and the (possibly absent) labelled edges between them agree in both graphs.
    Connected compatible sets are enumerated once each and pruned with a reachability-aware bound."""
    V = [(a, b) for a in g1.lab for b in g2.lab if g1.lab[a] == g2.lab[b]]
    n1, n2 = g1.size(), g2.size()
    if not V:
        return MCS(0, [], 0, 0.0, True, 0)
    e1, e2 = g1.e, g2.e
    rows = {}

    def row(i):
        r = rows.get(i)
        if r is None:
            a, b = V[i]
            r = set()
            for j, (c, d) in enumerate(V):
                if c != a and d != b and e1.get((a, c)) == e2.get((b, d)) and e1.get((c, a)) == e2.get((d, b)):
                    r.add(j)
            rows[i] = r
        return r

    def gain(j, C):
        c, _ = V[j]
        return sum((1 if (V[i][0], c) in e1 else 0) + (1 if (c, V[i][0]) in e1 else 0) for i in C)

    state = {"best": 0, "set": [], "exp": 0, "exact": True}

    def bound(C, comp, w_nodes):
        have1 = {V[i][0] for i in C}
        have2 = {V[i][1] for i in C}
        by1 = {}
        for j in comp:
            by1.setdefault(V[j][0], []).append(j)
        reach, queue = set(), list(have1)
        while queue:
            x = queue.pop()
            for y in g1.nbr[x]:
                if y in by1 and y not in have1 and y not in reach:
                    reach.add(y)
                    queue.append(y)
        s2 = {V[j][1] for y in reach for j in by1[y]}
        pot = min(len(reach), len(s2))
        ed = min(g1.edge_count(have1 | reach), g2.edge_count(have2 | s2))
        return w_nodes + pot + ed

    def rec(C, comp, w, front):
        if w > state["best"]:
            state["best"], state["set"] = w, list(C)
        state["exp"] += 1
        if state["exp"] > budget:
            state["exact"] = False
            return
        comp = set(comp)
        if bound(C, comp, len(C)) <= state["best"]:
            return
        ext = [j for j in comp if V[j][0] in front]
        ext.sort(key=lambda j: (-gain(j, C), j))
        for v in ext:
            comp.discard(v)
            if state["exp"] > budget:
                state["exact"] = False
                return
            if bound(C, comp | {v}, len(C)) <= state["best"]:
                return
            nxt = comp & row(v)
            rec(C + [v], nxt, w + 1 + gain(v, C), front | g1.nbr[V[v][0]])

    starts = sorted(range(len(V)), key=lambda i: (-len(g1.nbr[V[i][0]]), i))
    done = set()
    for s in starts:
        if state["exp"] > budget:
            state["exact"] = False
            break
        comp = {j for j in row(s) if j not in done}
        rec([s], comp, 1, set(g1.nbr[V[s][0]]))
        done.add(s)
    pairs = [V[i] for i in state["set"]]
    w = state["best"]
    sim = 2.0 * w / (n1 + n2) if (n1 + n2) else 0.0
    return MCS(w, pairs, w - len(pairs), sim, state["exact"], state["exp"])
