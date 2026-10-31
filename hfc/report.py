"""Terminal summary, JSON, Markdown report and figures of one served request."""
import json
from pathlib import Path

from . import viz


def summary(res, name=""):
    st = res.stats
    lines = ["%s mode=%s ok=%s rho=%.2f goals=%d delta=%d repairs=%d" % (name, res.mode, res.ok, res.rho, res.n_goals, res.delta, res.repairs)]
    for g in res.graph.goals if res.graph else []:
        w = res.witnesses.get(g.gid, {})
        lines.append("  g%d %-28s %-7s -> %s" % (g.gid, "/".join(str(k) for k in g.key[1:] if k) or g.key[1], res.status.get(g.gid, "new"), ", ".join(x.split(".", 1)[-1] for x in res.cover.get(g.gid, ()))))
    if res.info.get("rejections"):
        lines.append("  rejected: " + "; ".join("%s (%s)" % (a.split(".", 1)[-1], b) for a, b in res.info["rejections"][:6]))
    if res.info.get("failed_checks"):
        lines.append("  failed re-validation: " + ", ".join("g%s:%s" % kv for kv in res.info["failed_checks"].items()))
    lines.append("  LLM calls %s, latency %.0f ms (analytic LLM %.0f ms + measured compute %.1f ms)" % (st.get("calls_total"), st.get("total_ms", 0), st.get("llm_total_ms", 0), st.get("compute_ms", 0)))
    lines.append("  verification: " + ", ".join("%s=%s" % kv for kv in sorted(res.info.get("feas", {}).items())))
    for ln in res.explanation.splitlines()[:8]:
        lines.append("  | " + ln)
    return "\n".join(lines)


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_plain(x) for x in (sorted(o) if isinstance(o, (set, frozenset)) else o)]
    if hasattr(o, "item"):
        return o.item()
    return o


def to_json(res):
    return _plain({"rid": res.rid, "mode": res.mode, "ok": res.ok, "rho": res.rho, "delta": res.delta, "n_goals": res.n_goals,
                   "status": res.status, "cover": res.cover, "fragments": res.frags, "junctions": res.junctions,
                   "witnesses": res.witnesses, "explanation": res.explanation, "repairs": res.repairs,
                   "stats": res.stats, "composition": res.info.get("composition"), "rejections": res.info.get("rejections"),
                   "failed_checks": res.info.get("failed_checks"), "verification": res.info.get("feas"),
                   "goals": [{"gid": g.gid, "key": list(g.key), "text": g.text, "slots": g.slots} for g in (res.graph.goals if res.graph else [])]})


def write(res, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = to_json(res)
    (outdir / (name + ".json")).write_text(json.dumps(d, indent=1))
    figs = []
    if res.info.get("composition") and res.info["composition"]["elements"]:
        figs += viz.draw_workflow(res.info["composition"], outdir / (name + "_workflow"), "%s (%s, rho=%.2f)" % (name, res.mode, res.rho))
        if res.graph is not None:
            figs += viz.draw_goals(res.graph, res.status, res.align, outdir / (name + "_goals"), "goal hypergraph and alignment")
    md = ["# %s" % name, "", "- mode: **%s**, rho = %.2f, sub-goals %d, to recompose %d, repair rounds %d" % (res.mode, res.rho, res.n_goals, res.delta, res.repairs),
          "- LLM calls: %s; latency %.0f ms (LLM part analytic)" % (res.stats.get("calls_total"), res.stats.get("total_ms", 0)),
          "- verification: " + ", ".join("%s=%s" % kv for kv in sorted(res.info.get("feas", {}).items())), "", "## Decisions", ""]
    for g in res.graph.goals if res.graph else []:
        md.append("- g%d `%s`: **%s** -> %s" % (g.gid, "/".join(str(k) for k in g.key), res.status.get(g.gid, "new"), ", ".join(res.cover.get(g.gid, ()))))
    md += ["", "## Explanation", "", res.explanation, "", "## Figures", ""] + ["- " + Path(f).name for f in figs]
    (outdir / (name + ".md")).write_text("\n".join(md))
    return figs
