"""Figures of the experiment series (legends inside the axes, one colour and marker per method)."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 6.8, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight"})
ORDER = ["HyFragComp", "HyFragComp-WR", "HyFragComp-NR", "Scratch", "CBR-KG-Comp", "CD-CompFrag"]
ST = {"HyFragComp": ("#e6550d", "o", "-"), "HyFragComp-WR": ("#31a354", "^", "--"), "HyFragComp-NR": ("#dd77bb", "D", ":"),
      "Scratch": ("#3182bd", "s", "--"), "CBR-KG-Comp": ("#6baed6", "v", "-."), "CD-CompFrag": ("#333333", "P", "--")}
STAGES = ["abstraction", "planning", "structural encoding", "recall + alignment", "re-validation", "delta recomposition",
          "verification", "explanation"]


def _ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return np.nan, 0.0
    return float(v.mean()), (float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _line(ax, xs, ms, es, m):
    c, mk, ls = ST.get(m, ("grey", "o", "-"))
    ax.errorbar(range(len(xs)), ms, yerr=es, label=m, color=c, marker=mk, ls=ls, ms=4, lw=1.3, capsize=2)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])


def _series(ax, agg, metric, scale=1.0, methods=None, src=None):
    meths = [m for m in ORDER if m in agg["methods"] and (methods is None or m in methods)]
    xs = None
    for m in meths:
        d = agg["methods"][m] if src is None else agg["methods"][m].get(src, {})
        xs = sorted((k for k in d if k != "trace"), key=float)
        ms = [d[x][metric]["mean"] * scale if d[x].get(metric, {}).get("mean") is not None else np.nan for x in xs]
        es = [(d[x][metric]["ci95"] or 0) * scale if d[x].get(metric) else 0 for x in xs]
        _line(ax, xs, ms, es, m)
    ax.legend(loc="best")
    return xs


def _save(fig, out):
    fig.tight_layout()
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png", dpi=200)
    plt.close(fig)


def render(exp, agg, out):
    globals()["fig_" + exp](agg, out)


def fig_e1(agg, out):
    fig, axs = plt.subplots(1, 3, figsize=(12, 2.9))
    _series(axs[0], agg, "correct")
    axs[0].set_ylabel("correctness")
    _series(axs[1], agg, "bad_reuse")
    axs[1].set_ylabel("incorrect-reuse rate")
    _series(axs[2], agg, "calls")
    axs[2].set_ylabel(r"LLM calls per request $N_{call}$")
    for a in axs:
        a.set_xlabel(r"staleness rate $\pi_{st}$ (%)")
    _save(fig, out)


def fig_e5(agg, out):
    fig, axs = plt.subplots(1, 4, figsize=(15, 2.9))
    _series(axs[0], agg, "coverage", 100)
    axs[0].set_ylabel("sub-goals covered by reuse (%)")
    _series(axs[1], agg, "calls")
    axs[1].set_ylabel(r"LLM calls per request $N_{call}$")
    _series(axs[2], agg, "promoted", src="trace", methods=["HyFragComp", "HyFragComp-WR", "HyFragComp-NR"])
    axs[2].set_ylabel("composite fragments promoted")
    _series(axs[3], agg, "records", src="trace")
    axs[3].set_ylabel("stored composition records")
    for a in axs:
        a.set_xlabel("requests served $n$")
    _save(fig, out)


def fig_e7(agg, out):
    fig, axs = plt.subplots(1, 4, figsize=(15, 2.9))
    _series(axs[0], agg, "lat_ms", 1 / 1000.0)
    axs[0].set_ylabel("end-to-end latency $T$ (s)")
    _series(axs[1], agg, "rho")
    axs[1].set_ylabel(r"reuse ratio $\rho$")
    _series(axs[2], agg, "compute_ms")
    axs[2].set_ylabel("measured compute per request (ms)")
    _series(axs[3], agg, "calls")
    axs[3].set_ylabel(r"LLM calls per request $N_{call}$")
    for a in axs:
        a.set_xlabel("repository size $|F|$")
    _save(fig, out)


def fig_e6(agg, out):
    meths = [m for m in ORDER if m in agg["methods"]]
    first = next(iter(agg["methods"].values()))
    keys = next(iter(first.values())).keys()
    gs = sorted({int(k.split("_")[0]) for k in keys})
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.0))
    if "HyFragComp" in agg["methods"]:
        runs = agg["methods"]["HyFragComp"].values()
        bottom = np.zeros(len(gs))
        for st in STAGES:
            v = []
            for g in gs:
                recs = [r for d in runs for r in d.get("%d_none" % g, [])]
                v.append(np.mean([r["stages"].get(st, 0.0) for r in recs]) / 1000.0 if recs else 0.0)
            axs[0].bar(range(len(gs)), v, bottom=bottom, label=st, edgecolor="black", lw=0.3)
            bottom += np.array(v)
        axs[0].set_xticks(range(len(gs)))
        axs[0].set_xticklabels([str(g) for g in gs])
        axs[0].set_ylabel("HyFragComp latency per stage (s), no overlap")
        axs[0].legend(loc="upper left", fontsize=6)
    for k, mode in enumerate(("full", "partial", "none")):
        ax = axs[k + 1]
        for m in meths:
            runs = agg["methods"][m].values()
            ms, es = [], []
            for g in gs:
                vals = [np.mean([r["total_ms"] for r in d.get("%d_%s" % (g, mode), [])]) / 1000.0 for d in runs if d.get("%d_%s" % (g, mode))]
                a, b = _ci(vals)
                ms.append(a)
                es.append(b)
            _line(ax, gs, ms, es, m)
        ax.set_ylabel("latency $T$ (s), %s overlap" % {"full": "full", "partial": "partial", "none": "no"}[mode])
        ax.legend(loc="upper left")
    for a in axs:
        a.set_xlabel("number of sub-goals $|G|$")
    _save(fig, out)


def fig_x1(agg, out):
    tags = ["same", "prio_flip", "qual_flip", "extend", "subset", "new"]
    lab = ["identical", "priority-\nflipped", "qualifier-\nflipped", "extended", "reduced", "unrelated"]
    meths = [m for m in ORDER if m in agg["methods"] and any(d.get("by_tag") for d in agg["methods"][m].values())]
    fig, axs = plt.subplots(1, 3, figsize=(14, 2.9), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    w = 0.8 / max(1, len(meths))
    for i, m in enumerate(meths):
        runs = list(agg["methods"][m].values())
        vals = [_ci([100 * d["by_tag"][t]["reused_ok"] / d["by_tag"][t]["reused"] for d in runs
                     if t in d["by_tag"] and d["by_tag"][t]["reused"]]) for t in tags]
        axs[0].bar(np.arange(len(tags)) + (i - (len(meths) - 1) / 2) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
                   color=ST[m][0], label=m, edgecolor="black", lw=0.4, capsize=1.5)
        sizes = sorted(runs[0]["size"], key=float)
        for ax, key in ((axs[1], "prec"), (axs[2], "correct_share")):
            v = [_ci([100 * d["size"][z][key] for d in runs]) for z in sizes]
            _line(ax, sizes, [a for a, _ in v], [b for _, b in v], m)
    axs[0].set_xticks(range(len(tags)))
    axs[0].set_xticklabels(lab, fontsize=6.5)
    axs[0].set_ylim(0, 125)
    axs[0].set_ylabel("correctly reused sub-goals (%)")
    axs[0].legend(loc="upper center", ncol=4, fontsize=6.3)
    axs[1].set_ylabel("reuse precision (%)")
    axs[2].set_ylabel("sub-goals correctly reused (%)")
    for a in axs[1:]:
        a.set_xlabel("repository size $|F|$")
        a.legend(loc="best")
    _save(fig, out)


def fig_x2(agg, out):
    fig, axs = plt.subplots(1, 4, figsize=(15, 2.9))
    _series(axs[0], agg, "mem_stale", 100, src="trace", methods=["HyFragComp", "HyFragComp-WR", "HyFragComp-NR", "CBR-KG-Comp"])
    axs[0].set_ylabel("stale stored entries (%)")
    _series(axs[1], agg, "ev_evals", src="trace")
    axs[1].set_yscale("symlog", linthresh=1)
    axs[1].set_ylabel("checks per change event")
    _series(axs[2], agg, "reuse_prec", 100, methods=["HyFragComp", "HyFragComp-WR", "HyFragComp-NR", "CBR-KG-Comp"])
    axs[2].set_ylabel("reuse precision (%)")
    _series(axs[3], agg, "over", 100)
    axs[3].set_ylabel("over-covering workflows (%)")
    for a in axs:
        a.set_xlabel("requests served $n$")
    _save(fig, out)


def fig_x3(agg, out):
    fig, axs = plt.subplots(1, 3, figsize=(14, 2.9), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    mem = [m for m in ORDER if m in agg["methods"] and any("leak" in d for d in agg["methods"][m].values())]
    cats = [("name", "shared", "person names in\nshared memory"), ("name", "explained", "person names in other\nrequesters' explanations"),
            ("struct", "shared", "structured literals\nin shared memory")]
    w = 0.8 / max(1, len(mem))
    for i, m in enumerate(mem):
        runs = [d for d in agg["methods"][m].values() if "leak" in d]
        vals = []
        for kind, where, _ in cats:
            if kind == "struct":
                vals.append(_ci([100 * np.mean([d["leak"][k][where] for k in ("email", "phone", "id", "date", "amount")]) for d in runs]))
            else:
                vals.append(_ci([100 * d["leak"][kind][where] for d in runs]))
        axs[0].bar(np.arange(3) + (i - (len(mem) - 1) / 2) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], color=ST[m][0],
                   label=m, edgecolor="black", lw=0.4, capsize=1.5)
    axs[0].set_xticks(range(3))
    axs[0].set_xticklabels([c[2] for c in cats], fontsize=7)
    axs[0].set_ylabel("injected literals (%)")
    axs[0].legend(loc="upper right", ncol=2)
    for m in [m for m in ORDER if m in agg["methods"]]:
        runs = list(agg["methods"][m].values())
        nv = sorted(runs[0]["membership"], key=int)
        v = [_ci([d["membership"][x] for d in runs]) for x in nv]
        _line(axs[1], nv, [a for a, _ in v], [b for _, b in v], m)
    axs[1].axhline(0.5, color="grey", ls=":", lw=1)
    axs[1].set_ylim(0.4, 1.05)
    axs[1].set_xlabel("distinct earlier requesters of the procedure")
    axs[1].set_ylabel("attack AUC (latency probe)")
    axs[1].legend(loc="center right", ncol=2)
    for m in mem:
        runs = [d for d in agg["methods"][m].values() if "faithful" in d]
        xs = sorted(runs[0]["faithful"], key=float)
        v = [_ci([100 * d["faithful"][x]["faithful"] for d in runs]) for x in xs]
        _line(axs[2], xs, [a for a, _ in v], [b for _, b in v], m)
    axs[2].set_xlabel(r"staleness rate $\pi_{st}$ (%)")
    axs[2].set_ylabel("witnesses with a true validity claim (%)")
    axs[2].legend(loc="lower left")
    _save(fig, out)
