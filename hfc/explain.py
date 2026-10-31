"""Witnesses, symbolic explanation and the explicit JSON form of a final composition."""
from collections import defaultdict


def _topo(fset, conns):
    pred = defaultdict(set)
    for u, v, _ in conns:
        pred[v].add(u)
    seen, order = set(), []
    pend = sorted(fset)
    while pend:
        prog = False
        for x in list(pend):
            if pred[x] <= seen:
                order.append(x)
                seen.add(x)
                pend.remove(x)
                prog = True
        if not prog:
            order += pend
            break
    return order


def witnesses(graph, plan, out, hg, t, rejections=(), hide_scope=False):
    """One witness per sub-goal: record, match parts, status, validity evidence, stamps, junctions, failed check."""
    wit = {}
    rej = defaultdict(list)
    for fid, why in rejections:
        rej[why].append(fid)
    base = dict(plan.witnesses) if plan is not None else {}
    for g in graph.goals:
        w = dict(base.get(g.gid, {}))
        st = plan.status.get(g.gid) if plan is not None else None
        w["status"] = st or "new"
        fr = out["cover"].get(g.gid, ())
        w["fragments"] = list(fr)
        w["omega"] = sorted(g.omega)
        if st:
            parts = w.get("m", [0, 0, 0])
            line = "%s: %s from record %s (tier %s), match %.2f = %.2f + %.2f + %.2f, valid at t=%.1f within window %s" % (
                g.text[:60], "switched variant of" if st == "switch" else "reused", w.get("record", "?"), w.get("tier", "?"),
                sum(parts), parts[0], parts[1], parts[2], t, list(w.get("nu", ())))
            if st == "switch":
                line += ", %s replaced by %s because the guard no longer holds" % (w.get("switched_from"), list(fr))
            if g.slots:
                line += ", slots " + ", ".join("%s := %s" % kv for kv in g.slots.items())
        else:
            why = w.get("failed", "no-match")
            if hide_scope and why == "V5":
                # a scope rejection would reveal that another requester's private record matched
                why = "no-match"
                w.pop("failed", None)
                for k in ("record", "m", "nu", "tier", "stamps"):
                    w.pop(k, None)
            line = "%s: composed anew with %s (previous reuse refused: %s)" % (g.text[:60], list(fr), why)
        w["line"] = line
        wit[g.gid] = w
    extra = []
    for why, ids in rej.items():
        extra.append("rejected %s by %s" % (sorted(set(ids))[:4], why))
    for k, w in wit.items():
        pass
    return wit, extra


def composition_json(graph, plan, out, hg, ctx, res_status, junctions, feas):
    """Ordered workflow elements with patterns, guards, I/O, provenance and verification status."""
    fset, conns = out["fset"], out["conns"]
    order = _topo(fset, conns)
    owner = {}
    for gid, fs in out["cover"].items():
        for f in fs:
            owner.setdefault(f, gid)
    els = []
    for x in order:
        f = hg.frag(x)
        if f is None:
            continue
        gid = owner.get(x)
        st = res_status.get(gid, "new") if gid is not None else "entry"
        els.append({"id": x, "type": f.ftype, "kind": f.kind, "text": f.text, "pre": sorted(f.pre), "post": sorted(f.post),
                    "inputs": sorted(f.inp), "outputs": sorted(f.out), "roles": sorted(f.roles), "guard": f.guard,
                    "window": list(f.nu), "rev": f.rev, "goal": gid, "origin": st if st in ("reuse", "switch", "new") else "entry"})
    rel = {}
    for u, v, eid in conns:
        rel.setdefault(eid, {"tail": set(), "head": set()})
        rel[eid]["tail"].add(u)
        rel[eid]["head"].add(v)
    rels = []
    for eid, d in sorted(rel.items()):
        e = hg.edge(eid)
        rels.append({"edge": eid, "pattern": e.pat if e else "", "guard": e.guard if e else "", "tail": sorted(d["tail"]), "head": sorted(d["head"])})
    return {"elements": els, "relations": rels, "junctions": [list(j) for j in junctions], "verification": dict(feas),
            "repair_rounds": out.get("repairs", 0)}
