"""Case knowledge graph: case nodes, attribute entities, inter-case triples and the four-layer ontology."""
from collections import defaultdict
from dataclasses import dataclass, field

from hfc import lex

from .cbr_mcs import LG

try:
    import networkx as nx
except ImportError:
    nx = None


class MiniGraph:
    """Small directed multigraph used when networkx is missing."""

    def __init__(self):
        self.attr, self.out, self.inn = {}, defaultdict(dict), defaultdict(dict)

    def add_node(self, n, **a):
        self.attr.setdefault(n, {}).update(a)

    def has_node(self, n):
        return n in self.attr

    def node(self, n):
        return self.attr[n]

    def add_edge(self, u, v, rel, owner=None, **a):
        d = dict(a, rel=rel, owner=owner)
        self.out[u][(v, rel, owner)] = d
        self.inn[v][(u, rel, owner)] = d

    def out_edges(self, n):
        return [(k[0], d) for k, d in self.out.get(n, {}).items()]

    def in_edges(self, n):
        return [(k[0], d) for k, d in self.inn.get(n, {}).items()]

    def remove_node(self, n):
        for (v, rel, ow) in list(self.out.get(n, {})):
            self.inn[v].pop((n, rel, ow), None)
        for (u, rel, ow) in list(self.inn.get(n, {})):
            self.out[u].pop((n, rel, ow), None)
        self.out.pop(n, None)
        self.inn.pop(n, None)
        self.attr.pop(n, None)

    def nodes(self):
        return list(self.attr)

    def n_edges(self):
        return sum(len(v) for v in self.out.values())


class NxGraph:
    """The same interface over networkx.MultiDiGraph."""

    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_node(self, n, **a):
        self.g.add_node(n, **a)

    def has_node(self, n):
        return self.g.has_node(n)

    def node(self, n):
        return self.g.nodes[n]

    def add_edge(self, u, v, rel, owner=None, **a):
        self.g.add_edge(u, v, key=(rel, owner), rel=rel, owner=owner, **a)

    def out_edges(self, n):
        return [(v, d) for _, v, d in self.g.out_edges(n, data=True)] if self.g.has_node(n) else []

    def in_edges(self, n):
        return [(u, d) for u, _, d in self.g.in_edges(n, data=True)] if self.g.has_node(n) else []

    def remove_node(self, n):
        if self.g.has_node(n):
            self.g.remove_node(n)

    def nodes(self):
        return list(self.g.nodes)

    def n_edges(self):
        return self.g.number_of_edges()


def new_graph():
    return NxGraph() if nx is not None else MiniGraph()


@dataclass
class Desc:
    """What the knowledge graph needs to know about a case or a query: goals, ordering, context and (for cases) fragments."""
    cid: str
    iso: str
    eco: str
    goals: list                       # (key, text, slot names)
    edges: list                       # (tail positions, head positions, pattern)
    atoms: frozenset
    frags: list = field(default_factory=list)   # per goal: list of Fragment
    weight: int = 1
    owner: str = ""


def concept_of(pid):
    slug = pid.split(".")[-1]
    dom = lex.DOMAINS.get(slug)
    return slug, (dom[0] if dom else slug), (dom[1] if dom else "service")


def gen_atom(a):
    return a.split(":", 1)[0]


class CaseKG:
    """Case nodes, attribute entities, inter-case relations and a four-layer ontology on one directed multigraph."""

    def __init__(self, ccfg):
        self.c = ccfg
        self.layers = list(ccfg["ontology_layers"])
        self.g = new_graph()
        self.keys = defaultdict(set)      # goal key -> case ids
        self.cases = {}                   # cid -> Desc
        self.backend = "networkx" if nx is not None else "mini"

    # ------------------------------------------------------------------ nodes
    def _ent(self, nid, kind, layer=None, **a):
        if not self.g.has_node(nid):
            self.g.add_node(nid, kind=kind, layer=layer, **a)
        return nid

    def _layer(self, name):
        return self.layers.index(name)

    # ------------------------------------------------------------------ insert
    def add(self, d, link=True):
        """Insert a case (or the query graph) with its goal layer, attributes, ontology links and inter-case triples."""
        cn = "case|" + d.cid
        steps, slots, kinds, texts, ins, outs, code = [], [], [], [], set(), set(), []
        self.g.add_node(cn, kind="case", layer=None, label="case")
        gn = []
        for j, (key, text, snames) in enumerate(d.goals):
            P, step, qual = key
            slug, dom, kind = concept_of(P)
            n = "goal|%s|%d" % (d.cid, j)
            gn.append(n)
            self.g.add_node(n, kind="goal", layer=None, label="goal|%s|%s|%s" % key, text=text)
            self.g.add_edge(cn, n, "has_goal", d.cid)
            proc = self._ent("proc|" + P, "proc", label="proc|" + P)
            conc = self._ent("concept|" + slug, "concept", self._layer("concept"), label="concept|" + slug, text=dom)
            feat = self._ent("feat|" + step, "feature", self._layer("feature"), label="feat|" + step)
            jur = self._ent("jur|" + d.iso, "jur", label="jur|" + d.iso, text=d.eco)
            self.g.add_edge(n, proc, "of_proc")
            self.g.add_edge(proc, jur, "in_jur")
            self.g.add_edge(proc, conc, "instance_of")
            self.g.add_edge(n, feat, "type")
            self.g.add_edge(conc, feat, "has_feature")
            for s in snames:
                self.g.add_edge(n, self._ent("slot|" + s, "slot", label="slot|" + s), "has_slot")
            if qual:
                self.g.add_edge(n, self._ent("qual|" + qual, "qual", label="qual|" + qual), "qualified_by")
            steps.append(step)
            kinds.append(kind)
            texts.append(text)
            slots += list(snames)
        for tl, hd, pat in d.edges:
            for a in tl:
                for b in hd:
                    self.g.add_edge(gn[a], gn[b], "pre", None, pat=pat)
        for a in sorted(d.atoms):
            if a.startswith("jur:") or a == "applicant_present":
                continue
            jn = "jur|" + d.iso
            self.g.add_edge(jn, self._ent("ctx|" + a, "ctx", label="ctx|" + a), "cond", d.cid)
        # fragment layer: goal -> fragment -> atomic mechanism -> property
        for j, frs in enumerate(d.frags):
            for f in frs:
                if f is None:
                    continue
                code.append(f.fid)
                fn = self._ent("frag|" + f.fid, "fragment", label="frag", text=f.text)
                self.g.add_edge(gn[j], fn, "realized_by", d.cid)
                mech = self._ent("mech|" + f.step, "mechanism", self._layer("mechanism"), label="mech|" + f.step)
                self.g.add_edge(fn, mech, "implements")
                self.g.add_edge(self._ent("feat|" + d.goals[j][0][1], "feature"), mech, "realized_by")
                props = {gen_atom(a) for a in f.post} | {"out:" + x for x in f.out} | {"role:" + x for x in f.roles}
                ins |= set(f.inp)
                outs |= set(f.out)
                for p in props:
                    self.g.add_edge(mech, self._ent("prop|" + p, "property", self._layer("property"), label="prop|" + p), "has_property")
        # attribute entities of the case
        name = "%s: %s" % (d.eco, ", ".join(dict.fromkeys(concept_of(k[0])[1] for k, _, _ in d.goals)))
        self.g.node(cn).update({"id": d.cid, "economy": d.eco, "iso": d.iso, "name": name, "description": " ".join(texts),
                                "examples": list(texts), "code": sorted(set(code)), "weight": d.weight, "owner": d.owner})
        for rel, vals in (("economy", [d.iso]), ("service_type", kinds), ("parameter", slots), ("capability", steps),
                          ("input", sorted(ins)), ("output", sorted(outs)),
                          ("tag", [k[0] for k, _, _ in d.goals])):
            for v in dict.fromkeys(vals):
                self.g.add_edge(cn, self._ent("attr|%s|%s" % (rel, v), "attr", label="attr|%s" % rel, text=str(v)), "has_" + rel, d.cid)
        self.cases[d.cid] = d
        keys = {k for k, _, _ in d.goals}
        if link:
            self._link(d.cid, keys)
            for k in keys:
                self.keys[k].add(d.cid)

    def _link(self, cid, keys):
        """Inter-case triples to the most overlapping cases (equivalent / extends / refines / overlaps)."""
        cnt = defaultdict(int)
        for k in keys:
            for o in self.keys.get(k, ()):
                if o != cid:
                    cnt[o] += 1
        top = sorted(cnt, key=lambda o: (-cnt[o], o))[: self.c["kg_links"]]
        for o in top:
            ok = {k for k, _, _ in self.cases[o].goals}
            if ok == keys:
                rel = "equivalent"
            elif ok < keys:
                rel = "extends"
            elif keys < ok:
                rel = "refines"
            else:
                rel = "overlaps"
            self.g.add_edge("case|" + cid, "case|" + o, rel, cid, shared=cnt[o])

    def drop(self, cid):
        d = self.cases.pop(cid, None)
        if d is None:
            return
        for k, _, _ in d.goals:
            self.keys[k].discard(cid)
        for j in range(len(d.goals)):
            self.g.remove_node("goal|%s|%d" % (cid, j))
        self.g.remove_node("case|" + cid)

    # ------------------------------------------------------------------ views
    def triples(self, cid):
        """The case in triple form (head, relation, tail), with goal instances written by their labels."""
        cn = "case|" + cid
        out = []
        lab = lambda n: self.g.node(n).get("label", n)
        for v, d in self.g.out_edges(cn):
            if d["rel"] == "has_goal":
                out.append((cn, "has_goal", lab(v)))
                for w, e in self.g.out_edges(v):
                    out.append((lab(v), e["rel"], lab(w)))
            else:
                out.append((cn, d["rel"], lab(v)))
        return out

    def structure(self, cid):
        """Structural normalisation: goal-layer view with merged shared entities and collapsed ordering patterns."""
        cn = "case|" + cid
        lg = LG()
        gids = [v for v, d in self.g.out_edges(cn) if d["rel"] == "has_goal"]
        lab = lambda n: self.g.node(n)["label"]
        kind = lambda n: self.g.node(n)["kind"]
        for n in gids:
            lg.add_node(lab(n), lab(n), "goal")
        shared = set()
        for n in gids:
            for v, d in self.g.out_edges(n):
                if d["rel"] in ("of_proc", "type", "has_slot", "qualified_by"):
                    lg.add_node(lab(v), lab(v), kind(v))
                    lg.add_edge(lab(n), lab(v), d["rel"])
                    if d["rel"] == "of_proc":
                        shared.add(v)
        jurs = set()
        for p in shared:
            for v, d in self.g.out_edges(p):
                if d["rel"] in ("in_jur", "instance_of"):
                    lg.add_node(lab(v), lab(v), kind(v))
                    lg.add_edge(lab(p), lab(v), d["rel"])
                    if d["rel"] == "in_jur":
                        jurs.add(v)
        for j in jurs:
            for v, d in self.g.out_edges(j):
                if d["rel"] == "cond" and d["owner"] == cid:
                    lg.add_node(lab(v), lab(v), kind(v))
                    lg.add_edge(lab(j), lab(v), "cond")
        pre = set()
        for n in gids:
            for v, d in self.g.out_edges(n):
                if d["rel"] == "pre":
                    pre.add((lab(n), lab(v)))
        for u, v in self._reduce(pre):
            lg.add_edge(u, v, "pre")
        lg.shrink(self.c["mcs_max_nodes"], self.c["mcs_drop_order"])
        return lg

    @staticmethod
    def _reduce(pairs):
        """Transitive reduction of the precedence relation."""
        succ = defaultdict(set)
        for u, v in pairs:
            succ[u].add(v)

        def reach(a, skip):
            seen, st = set(), [x for x in succ[a] if x != skip]
            while st:
                x = st.pop()
                if x == skip or x in seen:
                    continue
                seen.add(x)
                st += list(succ[x])
            return seen
        return {(u, v) for u, v in pairs if v not in reach(u, v)}

    def ontology(self):
        """Nodes of each ontology layer, from semantic concept down to property."""
        out = {l: [] for l in self.layers}
        for n in self.g.nodes():
            ly = self.g.node(n).get("layer")
            if ly is not None:
                out[self.layers[ly]].append(n)
        return out

    def related(self, cid):
        return [(self.g.node(o)["id"], d["rel"], d.get("shared", 0)) for o, d in self.g.out_edges("case|" + cid)
                if d["rel"] in ("equivalent", "extends", "refines", "overlaps")]

    def attrs(self, cid):
        return dict(self.g.node("case|" + cid))

    def stats(self):
        return {"backend": self.backend, "cases": len(self.cases), "nodes": len(self.g.nodes()), "triples": self.g.n_edges()}
