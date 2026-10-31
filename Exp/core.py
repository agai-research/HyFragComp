"""Run an experiment series, aggregate it, and write tables and figures.

results/raw/<exp>/<profile>/<method>_s<seed>.json   one file per method and seed (resumable)
results/agg/<exp>_<profile>.json                     mean, std, 95% CI and n over runs; paired comparisons
results/tables/<exp>_<profile>.csv                   the aggregated values in tabular form
results/figs/<exp>_<profile>.pdf|png                 the figure of the series
"""
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from hfc import config

from . import plots
from .harness import METHODS, ci95
from .protocols import SERIES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DEFAULT_METHODS = list(METHODS)


def params(exp, profile):
    cfg = config.load()
    p = dict(cfg["experiments"][profile][exp])
    p.setdefault("runs", cfg["experiments"][profile]["runs"])
    return p


def _raw(exp, profile):
    d = RES / "raw" / exp / profile
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job(a):
    exp, profile, method, seed, tuned = a
    f = _raw(exp, profile) / ("%s_s%d.json" % (method, seed))
    if f.exists():
        return str(f), 0.0
    t0 = time.time()
    out = SERIES[exp](method, seed, params(exp, profile), tuned=tuned)
    f.write_text(json.dumps({"exp": exp, "profile": profile, "method": method, "seed": seed, "tuned": tuned, "data": out}))
    return str(f), time.time() - t0


def run_series(exp, runs=None, profile="full", methods=None, jobs=2, tuned=True):
    if exp not in SERIES:
        raise SystemExit("unknown series %s; available: %s" % (exp, ", ".join(SERIES)))
    p = params(exp, profile)
    runs = runs or p["runs"]
    todo = [(exp, profile, m, s, tuned) for s in range(runs) for m in (methods or DEFAULT_METHODS)]
    if jobs <= 1:
        it = map(_job, todo)
    else:
        pool = mp.get_context("fork").Pool(jobs)
        it = pool.imap_unordered(_job, todo)
    for f, sec in it:
        print("  %s (%.0fs)" % (Path(f).name, sec), flush=True)
    if jobs > 1:
        pool.close()
        pool.join()


# ---------------------------------------------------------------- aggregation
def _load(exp, profile, methods, runs):
    out = {}
    for f in sorted(_raw(exp, profile).glob("*.json")):
        d = json.loads(f.read_text())
        if methods and d["method"] not in methods:
            continue
        if runs and d["seed"] >= runs:
            continue
        out.setdefault(d["method"], {})[d["seed"]] = d["data"]
    return out


def paired(rows_a, rows_b, key, n_boot=2000, seed=0):
    """Paired difference a - b over the requests of the same run and setting, with bootstrap 95% CI and Cohen's d."""
    a = np.asarray([r[key] for r in rows_a], float)
    b = np.asarray([r[key] for r in rows_b], float)
    n = min(len(a), len(b))
    if n == 0:
        return None
    d = a[:n] - b[:n]
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    sd = d.std(ddof=1) if n > 1 else 0.0
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "cohen_d": float(d.mean() / sd) if sd > 0 else None, "n": int(n)}


def _settings(exp, data):
    """(setting key, per-seed dict of summary, per-seed rows) for the row-based series."""
    if exp in ("e1", "e7"):
        return {x: ({s: v[x]["summary"] for s, v in data.items()}, {s: v[x]["rows"] for s, v in data.items()}) for x in next(iter(data.values()))}
    if exp in ("e5", "x2"):
        k0 = next(iter(data.values()))["windows"]
        return {x: ({s: v["windows"][x]["summary"] for s, v in data.items()}, {s: v["windows"][x]["rows"] for s, v in data.items()}) for x in k0}
    return {}


def aggregate(exp, profile="full", methods=None, runs=None):
    raw = _load(exp, profile, methods, runs)
    agg = {"exp": exp, "profile": profile, "methods": {}, "paired": {}}
    if exp in ("e1", "e7", "e5", "x2"):
        for m, data in raw.items():
            agg["methods"][m] = {}
            for x, (summ, _rows) in _settings(exp, data).items():
                keys = next(iter(summ.values())).keys()
                agg["methods"][m][x] = {k: ci95([summ[s][k] for s in summ]) for k in keys}
            if exp in ("e5", "x2"):
                agg["methods"][m]["trace"] = {}
                for t in next(iter(data.values()))["trace"]:
                    n = str(t["n"])
                    vals = [next((u for u in data[s]["trace"] if u["n"] == t["n"]), None) for s in data]
                    agg["methods"][m]["trace"][n] = {k: ci95([v[k] for v in vals if v and k in v]) for k in t if k != "n"}
        if "HyFragComp" in raw:
            base = _settings(exp, raw["HyFragComp"])
            for m, data in raw.items():
                if m == "HyFragComp":
                    continue
                other = _settings(exp, data)
                for key in ("correct", "calls", "lat_ms", "rho", "bad_reuse"):
                    ra, rb = [], []
                    for x in base:
                        for s in base[x][1]:
                            if x in other and s in other[x][1]:
                                ra += base[x][1][s]
                                rb += other[x][1][s]
                    agg["paired"].setdefault(m, {})[key] = paired(ra, rb, key)
    else:
        agg["methods"] = {m: {str(s): v for s, v in data.items()} for m, data in raw.items()}
    f = RES / "agg"
    f.mkdir(parents=True, exist_ok=True)
    (f / ("%s_%s.json" % (exp, profile))).write_text(json.dumps(agg, indent=1))
    return agg


def write_tables(exp, agg, profile="full"):
    f = RES / "tables"
    f.mkdir(parents=True, exist_ok=True)
    out = f / ("%s_%s.csv" % (exp, profile))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        if exp in ("e1", "e7", "e5", "x2"):
            w.writerow(["method", "x", "metric", "mean", "std", "ci95", "n"])
            for m, xs in agg["methods"].items():
                for x, mets in xs.items():
                    if x == "trace":
                        for n, tm in mets.items():
                            for k, v in tm.items():
                                w.writerow([m, n, k, v["mean"], v["std"], v["ci95"], v["n"]])
                        continue
                    for k, v in mets.items():
                        w.writerow([m, x, k, v["mean"], v["std"], v["ci95"], v["n"]])
            w.writerow([])
            w.writerow(["paired vs HyFragComp", "metric", "diff (HyFragComp - method)", "ci95 low", "ci95 high", "cohen_d", "n"])
            for m, d in agg["paired"].items():
                for k, v in d.items():
                    if v:
                        w.writerow([m, k, v["diff"], v["ci95"][0], v["ci95"][1], v["cohen_d"], v["n"]])
        else:
            w.writerow(["see results/agg/%s_%s.json" % (exp, profile)])


def render_figures(exp, agg, profile="full"):
    f = RES / "figs"
    f.mkdir(parents=True, exist_ok=True)
    plots.render(exp, agg, f / ("%s_%s" % (exp, profile)))
