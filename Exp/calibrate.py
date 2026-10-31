"""Calibration of the deterministic backend against the hosted model (GPT-5.6 Luna by default).

  python Exp/calibrate.py run      replays a subset of the workloads with llm.backend=shadow (the rule engine
                                   decides, every role call is also sent to the hosted model) and with
                                   llm.backend=openai (the hosted model decides); needs OPENAI_API_KEY
  python Exp/calibrate.py analyze  reads the logs and writes results/calibration/report.{json,md} and
                                   configs/latency_fitted.yaml (latency model fitted on the measured calls)

Workload (configurable): the protocol of Fig. 4 at staleness 0% and 75% (--n requests each) and the first
--growth requests of the repository-growth stream, all with seed 0.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hfc.util import ensure_hash_seed  # noqa: E402

ensure_hash_seed()
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from Data.inst import build_stream  # noqa: E402
from Exp.harness import build, env, replay  # noqa: E402
from hfc.llm import ROLES, Roles  # noqa: E402

OUT = ROOT / "results" / "calibration"
LOGS = ROOT / "results" / "logs"


def workloads(cfg, repo, a):
    for x in a.stale:
        yield "e1_stale%d" % x, build_stream(cfg, repo, 0, n_req=a.n, stale=x / 100.0)
    if a.growth:
        yield "e5_first%d" % a.growth, build_stream(cfg, repo, 0, n_req=a.growth, n_tpl=400, warm=False,
                                                    stale=cfg["streams"]["stale_default"])


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("shadow.jsonl", "llm_calls.jsonl"):
        (LOGS / f).unlink(missing_ok=True)
    plans = {}
    for backend in ("rule", "shadow", "openai"):
        cfg, models, repo = env("HyFragComp", 1000, overrides={"llm.backend": backend})
        for name, inst in workloads(cfg, repo, a):
            m = build("HyFragComp", repo, cfg, models, 0)
            if backend != "rule":
                m.roles.api.context = {"workload": name, "backend": backend}
            rows = []

            def on_row(m, req, res, row):
                kept = sorted(str(g) for g, s in res.status.items() if s in ("reuse", "switch"))
                rows.append({"rid": req.rid, "mode": res.mode, "kept": kept, "goals": [list(g.key) for g in res.graph.goals] if res.graph else [],
                             "correct": row["correct"], "calls": row["calls"], "lat_ms": row["lat_ms"]})
            replay(m, inst, on_row)
            plans["%s|%s" % (backend, name)] = rows
            if backend == "openai":
                plans["fallbacks|%s" % name] = dict(m.roles.fallbacks)
            print(backend, name, len(rows), flush=True)
    (OUT / "plans.json").write_text(json.dumps(plans))


def fit_latency(recs):
    """Least squares of T = T0 + n_in / r_in + n_out / r_out on the measured, non-cached calls."""
    X = np.array([[1.0, r["tok_in"], r["tok_out"]] for r in recs], float)
    y = np.array([r["ms"] for r in recs], float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    t0, a, b = coef
    pred = X @ coef
    mape = float(np.mean(np.abs(pred - y) / np.maximum(y, 1e-9)))
    return {"fixed_ms": float(max(t0, 0.0)), "prefill_tok_s": float(1000.0 / a) if a > 0 else None,
            "decode_tok_s": float(1000.0 / b) if b > 0 else None, "mape_per_call": mape, "n": int(len(y))}


def agree(role, rule, model):
    if role == "pln":
        f = lambda o: sorted((g["procedure"], g["step"], g.get("qualifier") or "") for g in o.get("goals", []))  # noqa: E731
        return f(rule) == f(model)
    if role == "sel":
        return rule.get("choice") == model.get("choice")
    if role == "ret":
        return len(rule.get("queries", [])) == len(model.get("queries", []))
    if role == "cur":
        return not Roles.has_pii(str(model.get("template", "")))
    return None


def analyze(a):
    OUT.mkdir(parents=True, exist_ok=True)
    sh = [json.loads(l) for l in (LOGS / "shadow.jsonl").read_text().splitlines()]
    rep = {"roles": {}, "requests": {}}
    for role in ROLES:
        rs = [r for r in sh if r["role"] == role]
        if not rs:
            continue
        ag = [agree(role, r["rule"], r["model"]) for r in rs if isinstance(r["model"], dict)]
        ag = [x for x in ag if x is not None]
        rep["roles"][role] = {"calls": len(rs), "ms_mean": float(np.mean([r["ms"] for r in rs])),
                              "tok_in_mean": float(np.mean([r["tok_in"] for r in rs])), "tok_out_mean": float(np.mean([r["tok_out"] for r in rs])),
                              "tok_in_est_mean": float(np.mean([r["tok_in_est"] for r in rs])), "tok_out_est_mean": float(np.mean([r["tok_out_est"] for r in rs])),
                              "agreement": float(np.mean(ag)) if ag else None, "invalid_json": sum(1 for r in rs if not isinstance(r["model"], dict))}
    fresh = [r for r in sh if not r.get("cached")] or sh
    rep["latency_model"] = fit_latency(fresh)
    lm = rep["latency_model"]
    if lm["prefill_tok_s"] and lm["decode_tok_s"]:
        cfg = {"llm": {"latency": {"fixed_ms": round(lm["fixed_ms"], 1), "prefill_tok_s": round(lm["prefill_tok_s"], 1),
                                   "decode_tok_s": round(lm["decode_tok_s"], 1)}}}
        (ROOT / "configs" / "latency_fitted.yaml").write_text("# fitted by Exp/calibrate.py on %d measured calls\n" % lm["n"] + yaml.safe_dump(cfg))
    pf = OUT / "plans.json"
    if pf.exists():
        plans = json.loads(pf.read_text())
        for key in plans:
            if not key.startswith("rule|"):
                continue
            name = key.split("|", 1)[1]
            r, o = plans[key], plans.get("openai|" + name, [])
            n = min(len(r), len(o))
            if not n:
                continue
            rep["requests"][name] = {
                "n": n,
                "same_decomposition": float(np.mean([r[i]["goals"] == o[i]["goals"] for i in range(n)])),
                "same_reuse_plan": float(np.mean([(r[i]["mode"], r[i]["kept"]) == (o[i]["mode"], o[i]["kept"]) for i in range(n)])),
                "same_calls": float(np.mean([r[i]["calls"] == o[i]["calls"] for i in range(n)])),
                "correct_rule": float(np.mean([x["correct"] for x in r[:n]])), "correct_openai": float(np.mean([x["correct"] for x in o[:n]])),
                "latency_rule_ms": float(np.mean([x["lat_ms"] for x in r[:n]])), "latency_openai_ms": float(np.mean([x["lat_ms"] for x in o[:n]])),
                "fallbacks": plans.get("fallbacks|" + name, {})}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    lines = ["# Backend calibration", "", "| role | calls | mean latency (ms) | tokens in | tokens out | agreement |", "|---|---|---|---|---|---|"]
    for role, v in rep["roles"].items():
        lines.append("| %s | %d | %.0f | %.0f | %.0f | %s |" % (role, v["calls"], v["ms_mean"], v["tok_in_mean"], v["tok_out_mean"],
                                                          "-" if v["agreement"] is None else "%.1f%%" % (100 * v["agreement"])))
    lines += ["", "Latency model: T = %.0f ms + n_in/%.0f + n_out/%.0f tokens/s (MAPE per call %.1f%%, %d calls)" % (
        lm["fixed_ms"], lm["prefill_tok_s"], lm["decode_tok_s"], 100 * lm["mape_per_call"], lm["n"]), ""]
    for name, v in rep["requests"].items():
        lines.append("- %s: same decomposition %.1f%%, same reuse plan %.1f%%, same call count %.1f%%, correctness rule %.3f / model %.3f"
                     % (name, 100 * v["same_decomposition"], 100 * v["same_reuse_plan"], 100 * v["same_calls"], v["correct_rule"], v["correct_openai"]))
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "analyze"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stale", type=int, nargs="*", default=[0, 75])
    ap.add_argument("--growth", type=int, default=1000)
    a = ap.parse_args()
    run(a) if a.action == "run" else analyze(a)
