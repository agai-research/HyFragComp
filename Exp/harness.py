"""Shared harness of the experiments: method construction, event replay and scoring against reference workflows.

Correctness (Eq. 18 of the paper): the atomic fragment set of the returned workflow equals the reference
fragment set of the request (a revision f' of a reference fragment counts as that fragment) and every used
fragment is valid at the request time according to the staleness oracle (the current repository state).
"""
import re
import time
from pathlib import Path

import numpy as np

from Test.common import make_env
from hfc import config
from hfc.build import reference_fragments
from hfc.mem import expand_atomic
from hfc.methods import make
from hfc.repo import Models, Repo
from hfc.req import Result, request_from

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"HyFragComp": "ours", "HyFragComp-WR": "whole", "HyFragComp-NR": "nore", "Scratch": "scratch",
           "CBR-KG-Comp": "cbr", "CD-CompFrag": "cafc"}
MEMORY = ["HyFragComp", "HyFragComp-WR", "HyFragComp-NR", "CBR-KG-Comp"]


def load_cfg(method=None, tuned=True, overrides=None):
    files = []
    if tuned and method is not None:
        for f in ("tuned_shared.yaml", "tuned_%s.yaml" % METHODS[method]):
            if (ROOT / "configs" / f).exists():
                files.append(f)
    cfg = config.load(*files)
    for k, v in (overrides or {}).items():
        cfg.set(k, v)
    return cfg


def env(method, size, seed_models=0, tuned=True, overrides=None):
    cfg = load_cfg(method, tuned, overrides)
    models = Models(cfg, seed_models)
    repo = Repo(cfg, models, size)
    return cfg, models, repo


def build(method, repo, cfg, models, seed):
    return make(METHODS[method], repo, cfg, models, seed)


def base_id(fid):
    """Strip the suffix of a revision created by a change event (X.docs_r3 -> X.docs)."""
    return re.sub(r"_r\d+$", "", fid)


def reference(truth):
    out = set()
    for b in truth["blocks"]:
        steps = b["steps"]
        kind = "ext" if "extra" in steps else ("std" if "elig" in steps else "light")
        out |= set(reference_fragments(b["pid"], b["pid"].split(".")[0], kind, b["qual"], b["prio"]))
    return out


def atoms_of(hg, fs):
    a = set()
    for f in fs:
        a |= set(expand_atomic(hg, (f,)))
    return {x for x in a if not x.endswith(".ent") and ".bridge_" not in x}


def valid(hg, f, t):
    fr = hg.frag(f)
    return fr is not None and fr.nu[0] <= t < fr.nu[1]


def score(m, req, res):
    """(correct, incorrect reuse, stale fragment used, over-coverage, reused sub-goals, correctly reused sub-goals)."""
    if not res.ok or not res.frags:
        return 0, 0, 0, 0, 0, 0
    hg = m.w.hg
    ref = reference(req.truth)
    atoms = atoms_of(hg, res.frags)
    used = {base_id(a) for a in atoms}
    stale = int(any(not valid(hg, a, req.t) for a in atoms))
    correct = int(used == ref and not stale)
    over = int(bool(used - ref))
    bad = n = ok = 0
    for g, s in res.status.items():
        if s not in ("reuse", "switch"):
            continue
        n += 1
        fs = atoms_of(hg, res.cover.get(g, ()))
        good = bool(fs) and {base_id(x) for x in fs} <= ref and all(valid(hg, x, req.t) for x in fs)
        ok += int(good)
        bad = bad or int(not good)
    return correct, int(bad), stale, over, n, ok


def serve(m, req):
    """Serve one request; a crash is recorded as a failed request instead of stopping the series."""
    try:
        return m.serve(req), 0
    except Exception as e:  # noqa: BLE001 - every failure must be counted, not hidden
        res = Result(rid=req.rid, mode="crash", ok=False, stats={"calls_total": 0, "total_ms": 0.0, "compute_ms": 0.0})
        res.info["failure"] = repr(e)[:200]
        return res, 1


def row_of(m, req, res, crash, i):
    c, bad, st, over, n, ok = score(m, req, res)
    s = res.stats
    return {"i": i, "tag": req.truth.get("tag", "") if req.truth else "", "ok": int(res.ok), "mode": res.mode, "crash": crash,
            "correct": c, "bad_reuse": bad, "stale": st, "over": over, "reused": n, "reused_ok": ok,
            "n_goals": res.n_goals, "rho": res.rho, "calls": s.get("calls_total", 0), "lat_ms": s.get("total_ms", 0.0),
            "compute_ms": s.get("compute_ms", 0.0), "llm_ms": s.get("llm_total_ms", 0.0)}


def replay(m, inst, on_row=None, on_event=None):
    """Apply the change events in time order before each request; return one row per request."""
    if hasattr(m, "preload"):
        m.preload(inst["records"], 0.0)
    evs = sorted(inst["events"], key=lambda e: e["t"])
    j = 0
    rows = []
    for i, d in enumerate(inst["requests"]):
        while j < len(evs) and evs[j]["t"] <= d["t"]:
            t0 = time.perf_counter()
            try:
                o = m.handle(evs[j]) or {}
                err = 0
            except Exception:  # noqa: BLE001
                o, err = {}, 1
            if on_event:
                on_event(o, 1000.0 * (time.perf_counter() - t0), err)
            j += 1
        req = request_from(d)
        res, crash = serve(m, req)
        row = row_of(m, req, res, crash, i)
        if on_row:
            on_row(m, req, res, row)
        rows.append(row)
    return rows


def ci95(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "std": sd, "ci95": 1.96 * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0, "n": int(len(v))}
