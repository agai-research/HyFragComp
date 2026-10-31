"""Protocol of each experiment series. Every function runs one method with one seed and returns a JSON-able dict.

e1  impact of re-validation under staleness (Fig. 4)       xs: staleness rates (%)
e5  repository growth over a request stream (Fig. 5)       xs: requests served (checkpoints)
e7  scalability with the repository size (Fig. 6)          xs: |F|
e6  latency by pipeline stage (Fig. 7)                     xs: number of sub-goals |G|, three overlap modes
x1  matching quality of stored compositions (RQ2)          by request category and xs: |F|
x2  quality of the evolving memory (RQ3)                    xs: requests served
x3  privacy and explainability of reuse (RQ5)              literals, membership inference, witness faithfulness
"""
import json
import re

import numpy as np

from Data.build_requests import Gen, steps_of
from Data.inst import build_stream
from hfc.req import request_from

from .harness import MEMORY, atoms_of, base_id, build, env, reference, replay, serve, valid

ROLE_STAGE = {"cur": "abstraction", "int": "abstraction", "pln": "planning", "ret": "delta recomposition",
              "sel": "delta recomposition", "com": "delta recomposition", "ver": "delta recomposition", "exp": "explanation"}
TIMER_STAGE = {"abstraction": "abstraction", "decompose": "planning", "encoding": "structural encoding", "recall": "recall + alignment",
               "revalidation": "re-validation", "memory": "re-validation", "recompose": "delta recomposition",
               "verify": "verification", "explain": "explanation", "promotion": "promotion", "padding": "re-validation"}


def summary(rows):
    def m(k):
        return float(np.mean([r[k] for r in rows])) if rows else None
    reused = sum(r["reused"] for r in rows)
    goals = sum(r["n_goals"] for r in rows)
    return {"n": len(rows), "correct": m("correct"), "bad_reuse": m("bad_reuse"), "over": m("over"), "calls": m("calls"),
            "lat_ms": m("lat_ms"), "compute_ms": m("compute_ms"), "rho": m("rho"), "crash": sum(r["crash"] for r in rows),
            "coverage": reused / max(1, goals), "reuse_prec": sum(r["reused_ok"] for r in rows) / max(1, reused)}


def slim(rows):
    keys = ("i", "tag", "correct", "bad_reuse", "over", "calls", "lat_ms", "compute_ms", "rho", "reused", "reused_ok", "n_goals", "crash")
    return [{k: r[k] for k in keys} for r in rows]


# ---------------------------------------------------------------- e1
def e1(method, seed, p, tuned=True):
    out = {}
    for x in p["xs"]:
        cfg, models, repo = env(method, p.get("size", 1000), tuned=tuned)
        inst = build_stream(cfg, repo, seed, n_req=p["n_req"], stale=x / 100.0)
        rows = replay(build(method, repo, cfg, models, seed), inst)
        out[str(x)] = {"summary": summary(rows), "rows": slim(rows)}
    return out


# ---------------------------------------------------------------- e5 / x2
def stream_run(method, seed, p, tuned=True, memory_stats=False):
    cfg, models, repo = env(method, p.get("size", 1000), tuned=tuned)
    inst = build_stream(cfg, repo, seed, n_req=p["n_req"], n_tpl=p["n_tpl"], warm=False, stale=cfg["streams"]["stale_default"])
    m = build(method, repo, cfg, models, seed)
    evs = []
    trace = []
    xs = set(p["xs"])

    def on_event(o, ms, err):
        evs.append((o.get("evals", 0), o.get("llm_calls", 0), ms, err))

    def on_row(m, req, res, row):
        n = row["i"] + 1
        if n not in xs:
            return
        rec = {"n": n, "events": len(evs), "event_errors": int(sum(e[3] for e in evs))}
        if hasattr(m, "promo"):
            rec["promoted"] = m.promo.promoted
            rec["composites_live"] = sum(1 for c in m.promo.members if m.w.hg.frag(c) is not None)
        if hasattr(m, "mem"):
            rec["records"] = sum(1 for r in m.mem.recs.values() if r.kind == "rec" and not getattr(r, "evicted", False))
        elif hasattr(m, "cases"):
            rec["records"] = len(m.cases)
        if memory_stats:
            tot, st = memory_entries(m, req.t)
            e = np.array([x[:3] for x in evs]) if evs else np.zeros((1, 3))
            rec.update({"mem": tot, "mem_stale": st / max(1, tot), "ev_evals": float(e[:, 0].mean()),
                        "ev_llm": float(e[:, 1].mean()), "ev_ms": float(e[:, 2].mean())})
        trace.append(rec)
    rows = replay(m, inst, on_row, on_event)
    w = p.get("window", 100)
    return {"windows": {str(x): {"summary": summary(rows[max(0, x - w):x]), "rows": slim(rows[max(0, x - w):x])} for x in p["xs"]},
            "trace": trace}


def memory_entries(m, t):
    hg = m.w.hg
    recs = [r for r in m.mem.recs.values() if not getattr(r, "evicted", False)] if hasattr(m, "mem") else \
        [c.rec for c in getattr(m, "cases", {}).values()]
    tot = st = 0
    for r in recs:
        fids = set()
        for v in getattr(r, "phi", {}).values():
            fids |= set(v)
        tot += 1
        st += int(any(not valid(hg, f, t) or getattr(r, "stamps", {}).get(f) not in (None, hg.frag(f).rev) for f in fids))
    return tot, st


def e5(method, seed, p, tuned=True):
    return stream_run(method, seed, p, tuned)


def x2(method, seed, p, tuned=True):
    return stream_run(method, seed, p, tuned, memory_stats=True)


# ---------------------------------------------------------------- e7
def e7(method, seed, p, tuned=True):
    out = {}
    for size in p["xs"]:
        cfg, models, repo = env(method, size, tuned=tuned)
        inst = build_stream(cfg, repo, seed, n_req=p["n_req"], stale=cfg["streams"]["stale_default"])
        m = build(method, repo, cfg, models, seed)
        rows = replay(m, inst)
        s = summary(rows)
        if hasattr(m, "mem"):
            s["records"] = sum(1 for r in m.mem.recs.values() if r.kind == "rec")
        out[str(size)] = {"summary": s, "rows": slim(rows)}
    return out


# ---------------------------------------------------------------- e6
SIZES = {3: ["light"], 5: ["ext"], 8: ["std", "std"], 12: ["std", "std", "std"], 18: ["ext", "ext", "std", "std"]}


def _stages(res):
    st = res.stats
    out = {}
    for k, v in st.get("t", {}).items():
        out[TIMER_STAGE.get(k, k)] = out.get(TIMER_STAGE.get(k, k), 0.0) + 1000.0 * v
    for r, v in st.get("llm_ms", {}).items():
        out[ROLE_STAGE[r]] = out.get(ROLE_STAGE[r], 0.0) + v
    return out


def e6(method, seed, p, tuned=True):
    cfg, models, repo = env(method, p.get("size", 5000), tuned=tuned)
    gen = Gen(cfg, repo.pids, 500 + seed)
    rng = np.random.default_rng(seed)
    by_eco = {}
    for pr in gen.procs:
        by_eco.setdefault(pr["iso3"], []).append(pr)
    ecos = sorted(e for e, v in by_eco.items() if len(v) >= 8)
    out = {}
    for g in p["xs"]:
        kinds = SIZES[g]
        for mode in ("full", "partial", "none"):
            m = build(method, repo, cfg, models, seed)
            recs = []
            for k in range(p["n"]):
                eco = ecos[int(rng.integers(len(ecos)))]
                ps = [by_eco[eco][i] for i in rng.choice(len(by_eco[eco]), len(kinds), replace=False)]
                blocks = list(zip(ps, kinds))
                t = 1.0 + k
                user = "u%d_%d" % (seed, k)
                if mode == "full":
                    warm = blocks
                elif mode == "partial" and len(blocks) > 1:
                    warm = blocks[: len(blocks) // 2]
                elif mode == "partial":
                    steps = steps_of(kinds[0])
                    warm = [(ps[0], steps[: (len(steps) + 1) // 2])]
                else:
                    warm = []
                if warm:
                    serve(m, request_from(gen.request("w%d" % k, t, user, warm, "first", False)))
                req = request_from(gen.request("q%d" % k, t + 0.5, user, blocks, "first", False))
                res, crash = serve(m, req)
                if crash:
                    continue
                recs.append({"total_ms": res.stats.get("total_ms", 0.0), "calls": res.stats.get("calls_total", 0),
                             "stages": _stages(res), "mode": res.mode})
            out["%d_%s" % (g, mode)] = recs
    return out


# ---------------------------------------------------------------- x1
def x1(method, seed, p, tuned=True):
    out = {"by_tag": {}, "size": {}}
    if method not in MEMORY:
        return out
    cfg, models, repo = env(method, 1000, tuned=tuned)
    inst = build_stream(cfg, repo, seed, n_req=p["n_req"], stale=0.0)
    rows = replay(build(method, repo, cfg, models, seed), inst)
    for t in sorted({r["tag"] for r in rows}):
        rr = [r for r in rows if r["tag"] == t]
        out["by_tag"][t] = {"reused": sum(r["reused"] for r in rr), "reused_ok": sum(r["reused_ok"] for r in rr),
                            "goals": sum(r["n_goals"] for r in rr)}
    for size in p["xs"]:
        cfg, models, repo = env(method, size, tuned=tuned)
        inst = build_stream(cfg, repo, seed, n_req=p["n_req_size"], stale=0.0)
        rr = replay(build(method, repo, cfg, models, seed), inst)
        n = sum(r["reused"] for r in rr)
        out["size"][str(size)] = {"prec": sum(r["reused_ok"] for r in rr) / max(1, n),
                                  "correct_share": sum(r["reused_ok"] for r in rr) / max(1, sum(r["n_goals"] for r in rr))}
    return out


# ---------------------------------------------------------------- x3
LTYPES = ["name", "email", "phone", "id", "date", "amount"]


def _literal(rng, kind, lex):
    fn = lex.FIRST[int(rng.integers(len(lex.FIRST)))]
    ln = lex.LAST[int(rng.integers(len(lex.LAST)))]
    if kind == "name":
        lit = "%s %s" % (fn, ln)
        return "on behalf of " + lit, lit
    if kind == "email":
        lit = "%s.%s%d@mail.com" % (fn.lower(), ln.lower()[:3], int(rng.integers(10, 99)))
        return "with a copy to " + lit, lit
    if kind == "phone":
        lit = "+216 %d" % int(rng.integers(20000000, 99999999))
        return "and call me on " + lit, lit
    if kind == "id":
        lit = "%s%d" % ("".join(chr(65 + int(x)) for x in rng.integers(0, 26, 2)), int(rng.integers(1000000, 9999999)))
        return "for passport " + lit, lit
    if kind == "date":
        lit = "%d/%d/2026" % (int(rng.integers(1, 28)), int(rng.integers(1, 12)))
        return "before " + lit, lit
    lit = "%d TND" % int(rng.integers(50, 900))
    return "and pay " + lit, lit


def _inject(inst, seed):
    from hfc import lex
    rng = np.random.default_rng(seed + 77)
    meta = {}
    for i, d in enumerate(inst["requests"]):
        kind = LTYPES[i % len(LTYPES)]
        sents = re.split(r"(?<=\.) ", d["text"])
        idx = [k for k, s in enumerate(sents) if re.match(r"(First|Next|Also|Then|Finally|Please),? ", s)]
        if not idx:
            continue
        k = idx[int(rng.integers(len(idx)))]
        phrase, lit = _literal(rng, kind, lex)
        sents[k] = sents[k][:-1] + " " + phrase + "."
        d["text"] = " ".join(sents)
        meta[d["rid"]] = (kind, lit, d["user"])
    return meta


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    s = sum(np.sum(x < neg) + 0.5 * np.sum(x == neg) for x in pos) / (len(pos) * len(neg))
    return float(max(s, 1 - s))


def x3(method, seed, p, tuned=True):
    out = {}
    # (a) literals
    if method in MEMORY:
        cfg, models, repo = env(method, 1000, tuned=tuned)
        inst = build_stream(cfg, repo, seed, n_req=p["n_leak"], stale=0.10)
        meta = _inject(inst, seed)
        expl = {k: 0 for k in LTYPES}
        recent = []

        def on_row(m, req, res, row):
            for kind, lit, u in recent[-300:]:
                if u != req.user and lit in res.explanation:
                    expl[kind] += 1
            if req.rid in meta:
                recent.append(meta[req.rid])
        m = build(method, repo, cfg, models, seed)
        replay(m, inst, on_row)
        shared = []
        if hasattr(m, "mem"):
            for r in m.mem.recs.values():
                if r.scope == "shared":
                    shared.append(" ".join(g.text + " " + json.dumps(g.slots) for g in r.goals))
            for fid in m.promo.members:
                f = m.w.hg.frag(fid)
                if f is not None:
                    shared.append(f.text)
        else:
            for c in m.cases.values():
                shared.append(" ".join(g.text + " " + json.dumps(g.slots) for g in c.rec.goals))
        S = "\n".join(shared)
        out["leak"] = {}
        for kind in LTYPES:
            lits = [lit for (k, lit, u) in meta.values() if k == kind]
            out["leak"][kind] = {"shared": sum(1 for l in lits if l in S) / max(1, len(lits)), "explained": expl[kind] / max(1, len(lits))}
    # (b) membership inference
    cfg, models, repo = env(method, 5000, tuned=tuned)
    m = build(method, repo, cfg, models, seed)
    gen = Gen(cfg, repo.pids, 1000 + seed)
    rng = np.random.default_rng(seed)
    procs = list(gen.procs)
    rng.shuffle(procs)
    order = [nv for nv in p["nv"] for _ in range(p["trials"])]
    rng.shuffle(order)
    lat = {nv: [] for nv in p["nv"]}
    t = 1.0
    for k, nv in enumerate(order):
        pr = procs[k % len(procs)]
        for v in range(nv):
            serve(m, request_from(gen.request("v%d_%d" % (k, v), t, "victim%d_%d" % (seed, v), [(pr, "std")], "first", False)))
            t += 0.01
        res, crash = serve(m, request_from(gen.request("a%d" % k, t, "attacker%d" % seed, [(pr, "std")], "first", False)))
        t += 0.01
        if not crash:
            lat[nv].append(res.stats.get("total_ms", 0.0))
    out["membership"] = {str(nv): _auc(lat[nv], lat[0]) for nv in p["nv"] if nv > 0}
    out["membership_latency"] = {str(nv): float(np.mean(lat[nv])) if lat[nv] else None for nv in p["nv"]}
    # (c) witness faithfulness
    if method in MEMORY:
        out["faithful"] = {}
        for x in p["xs"]:
            cfg, models, repo = env(method, 1000, tuned=tuned)
            inst = build_stream(cfg, repo, seed, n_req=p["n_fid"], stale=x / 100.0)
            c = {"n": 0, "complete": 0, "true": 0}

            def on_row(m, req, res, row):
                hg = m.w.hg
                ref = reference(req.truth)
                for g, s in res.status.items():
                    if s not in ("reuse", "switch"):
                        continue
                    w = res.witnesses.get(g, {})
                    c["n"] += 1
                    c["complete"] += int(all(k in w for k in ("record", "m", "nu")) and bool(w.get("stamps")))
                    nu = w.get("nu")
                    fs = atoms_of(hg, res.cover.get(g, ()))
                    truth = all(valid(hg, a, req.t) for a in fs) and {base_id(a) for a in fs} <= ref
                    c["true"] += int(nu is not None and nu[0] <= req.t < nu[1] and truth)
            replay(build(method, repo, cfg, models, seed), inst, on_row)
            out["faithful"][str(x)] = {"complete": c["complete"] / max(1, c["n"]), "faithful": c["true"] / max(1, c["n"]), "n": c["n"]}
    return out


SERIES = {"e1": e1, "e5": e5, "e6": e6, "e7": e7, "x1": x1, "x2": x2, "x3": x3}
