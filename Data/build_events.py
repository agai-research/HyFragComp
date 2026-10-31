"""Construct change events: expiry, withdrawal, non-substitutable and substitutable revisions with their successors and bridges."""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfc.data import edge_to_dict, frag_to_dict, frag_from_dict as _from_dict, edge_from_dict as _edge_from
from hfc.model import Edge, HGView, INF_DAY

NEW_ITEM = "residence_proof"      # data item a revised decree makes mandatory (one item per revised fragment)


def item_of(f):
    return "%s:%s" % (NEW_ITEM, f.fid)


def _copy_edges(hg, f, g):
    """Edges of f copied for its successor g."""
    out = []
    seen = set()
    for e in list(hg.edges_head(f.fid)) + list(hg.edges_tail(f.fid)):
        if e.eid in seen or e.pat in ("der", "loop"):
            continue
        seen.add(e.eid)
        out.append(Edge(e.eid + "~" + g.fid.rsplit(".", 1)[-1], tuple(g.fid if x == f.fid else x for x in e.tail),
                        tuple(g.fid if x == f.fid else x for x in e.head), e.pat, e.guard, e.route))
    return out


def successor(hg, f, t_ev, tag="r"):
    g = replace(f, fid=f.fid + "_" + tag, text=f.text + " Updated under the latest regulation.", nu=(int(t_ev), INF_DAY), rev=1)
    return g, _copy_edges(hg, f, g)


def bridge_for(hg, f, iso):
    """Fragment that obtains the newly mandatory item, and the hyperedge that lets the revised fragment take it as input."""
    tag = f.fid.rsplit(".", 1)[-1]
    b = replace(f, fid="%s.bridge_%s" % (f.proc, tag), text="Obtain the residence proof now required before this step of the procedure.",
                pre=frozenset({"dossier_open:" + iso, "jur:" + iso}), post=frozenset({"proof_ok:" + f.proc}), inp=frozenset({"id_scan"}),
                out=frozenset({item_of(f)}), ftype="INT", step="bridge", variant="", guard="", prio=0, nu=(0, INF_DAY), rev=1,
                dmin=1, dmax=3, roles=frozenset({"citizen", "agency"}))
    e_in = Edge(b.fid + ".in", (iso + ".ent",), (b.fid,), "and")
    e_use = Edge(f.fid + ".use.%s" % tag, (iso + ".ent", b.fid), (f.fid,), "and")
    return b, [e_in, e_use]


def make_events(rng, hg, hot, n, t0, t1, mix, p_bridge=1.0, n_subst=0, tag="e"):
    """One event per affected fragment drawn without replacement from `hot`; `mix` = (expire, withdraw, nonsub) weights."""
    hg = HGView(hg.base)   # successors copy the edges as they stand after the earlier events, so combined revisions stay licensed
    pool = [x for x in hot if hg.frag(x) is not None and hg.frag(x).step not in ("ent",)]
    n = min(n, len(pool))
    pick = [pool[i] for i in rng.permutation(len(pool))[:n]]
    w = [float(x) for x in mix]
    w = [x / sum(w) for x in w]
    evs = []
    times = sorted(float(x) for x in rng.uniform(t0, t1, size=n + n_subst))
    for i, fid in enumerate(pick):
        f = hg.frag(fid)
        t = times[i]
        kind = ("expire", "withdraw", "nonsub")[int(rng.choice(3, p=w))]
        iso = f.proc.split(".")[0]
        if kind in ("expire", "withdraw"):
            g, es = successor(hg, f, t, "r%d" % i)
            adds = [{"frag": frag_to_dict(g), "edges": [edge_to_dict(e) for e in es]}]
            if kind == "expire":
                ev = {"type": "window", "fid": fid, "nu_hi": int(t), "t": t, "adds": adds}
            else:
                ev = {"type": "withdraw", "fid": fid, "t": t, "adds": adds}
        else:
            new = replace(f, inp=f.inp | {item_of(f)}, rev=f.rev + 1)
            adds = []
            if rng.random() < p_bridge:
                b, es = bridge_for(hg, f, iso)
                adds = [{"frag": frag_to_dict(b), "edges": [edge_to_dict(e) for e in es]}]
            ev = {"type": "update", "new": frag_to_dict(new), "t": t, "adds": adds}
        for a in adds:
            hg.add_fragment(_from_dict(a["frag"]))
            for ed in a["edges"]:
                hg.add_edge(_edge_from(ed))
        ev["kind"] = kind
        ev["id"] = "%s%d" % (tag, i)
        evs.append(ev)
    rest = [x for x in pool if x not in set(pick)]
    for j in range(min(n_subst, len(rest))):
        f = hg.frag(rest[int(rng.integers(len(rest)))])
        t = times[n + j]
        new = replace(f, dmax=f.dmax + 2, rev=f.rev + 1)
        evs.append({"type": "update", "new": frag_to_dict(new), "t": t, "adds": [], "kind": "subst", "id": "%ss%d" % (tag, j)})
    evs.sort(key=lambda e: e["t"])
    return evs
