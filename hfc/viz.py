"""BPMN-style drawings of a composed workflow and of the goal hypergraph with its alignment (Graphviz, matplotlib fallback)."""
import shutil
import subprocess
import textwrap
from pathlib import Path

COL = {"reuse": "#b7e1b0", "switch": "#f9c784", "new": "#a9c9ee", "entry": "#dddddd"}
LEG = [("reuse", "reused"), ("switch", "switched variant"), ("new", "newly composed"), ("entry", "entry")]


def _wrap(s, n=24):
    return "\\n".join(textwrap.wrap(s, n)[:3]) or s


def _label(el):
    short = el["id"].split(".", 1)[-1] if "." in el["id"] else el["id"]
    txt = el["text"]
    return "%s\\n%s" % (short, _wrap(txt, 26))


def workflow_dot(comp, title=""):
    els = {e["id"]: e for e in comp["elements"]}
    lines = ['digraph W {', 'rankdir=LR; fontname="Helvetica"; labelloc=t; nodesep=0.35; ranksep=0.5;',
             'label="%s"; node [fontname="Helvetica", fontsize=10]; edge [fontname="Helvetica", fontsize=9];' % title.replace('"', "'")]
    lines.append('start [shape=circle, label="", width=0.3, style=filled, fillcolor="#ffffff", penwidth=2];')
    lines.append('end [shape=doublecircle, label="", width=0.3, style=filled, fillcolor="#ffffff", penwidth=3];')
    for fid, e in els.items():
        shape = "box" if e["kind"] != "cmp" else "box3d"
        lines.append('"%s" [shape=%s, style="rounded,filled", fillcolor="%s", label="%s"];' % (fid, shape, COL.get(e["origin"], "#dddddd"), _label(e)))
    indeg, outdeg = {f: 0 for f in els}, {f: 0 for f in els}
    k = 0
    for r in comp["relations"]:
        tl = [t for t in r["tail"] if t in els]
        hd = [h for h in r["head"] if h in els]
        if not tl or not hd:
            continue
        pat = r["pattern"]
        g = r.get("guard") or ""
        k += 1
        src = tl
        if len(tl) > 1:
            gw = "j%d" % k
            lines.append('%s [shape=diamond, label="+", width=0.35, height=0.35, fontsize=16, style=filled, fillcolor="white"];' % gw)
            for t in tl:
                lines.append('"%s" -> %s;' % (t, gw))
            src = [gw]
        if len(hd) > 1:
            kind = "×" if pat == "xor" else "+"
            gw = "s%d" % k
            lines.append('%s [shape=diamond, label="%s", width=0.35, height=0.35, fontsize=16, style=filled, fillcolor="white"];' % (gw, kind))
            for s in src:
                lines.append('%s -> %s;' % (s if s.startswith(("j", "s")) else '"%s"' % s, gw))
            for h in hd:
                lines.append('%s -> "%s"%s;' % (gw, h, ' [label="%s"]' % g if g else ""))
        else:
            for s in src:
                sa = s if s.startswith(("j", "s")) and s not in els else '"%s"' % s
                style = []
                if pat == "loop":
                    style.append('style=dashed, constraint=false, label="loop"')
                elif g:
                    style.append('label="%s"' % g)
                lines.append('%s -> "%s" [%s];' % (sa, hd[0], ", ".join(style)))
        for t in tl:
            outdeg[t] += 1
        for h in hd:
            indeg[h] += 1
    for f in els:
        if indeg[f] == 0:
            lines.append('start -> "%s";' % f)
        if outdeg[f] == 0:
            lines.append('"%s" -> end;' % f)
    lines.append('subgraph cluster_leg { label="legend"; fontsize=9; style=dashed;')
    for i, (k2, nm) in enumerate(LEG):
        lines.append('leg%d [shape=box, style="rounded,filled", fillcolor="%s", label="%s", fontsize=9];' % (i, COL[k2], nm))
    lines.append("}")
    lines.append("}")
    return "\n".join(lines)


def goals_dot(graph, status, align, title=""):
    lines = ['digraph G {', 'rankdir=LR; fontname="Helvetica"; labelloc=t;', 'label="%s"; node [fontname="Helvetica", fontsize=10];' % title.replace('"', "'")]
    for g in graph.goals:
        st = status.get(g.gid, "new")
        a = align.get(g.gid)
        sub = "" if not a else "\\nmatched %s:g%s (m=%.2f)" % (a[0], a[1], sum(a[2]) if len(a) > 2 and a[2] is not None else 0.0)
        lines.append('g%d [shape=box, style="rounded,filled", fillcolor="%s", label="g%d: %s%s\\n[%s]"];' % (g.gid, COL.get(st, "#dddddd"), g.gid, _wrap(g.text, 30), sub, st))
    for i, (tl, hd, pat) in enumerate(graph.edges):
        if len(tl) > 1:
            lines.append('e%d [shape=diamond, label="%s", width=0.3, height=0.3];' % (i, "+" if pat == "and" else "×"))
            for t in tl:
                lines.append("g%d -> e%d;" % (t, i))
            for h in hd:
                lines.append("e%d -> g%d;" % (i, h))
        else:
            for t in tl:
                for h in hd:
                    lines.append("g%d -> g%d;" % (t, h))
    lines.append("}")
    return "\n".join(lines)


def render_dot(dot, base):
    """Write base.png and base.svg with Graphviz; returns the list of written files."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.with_suffix(".dot")).write_text(dot)
    out = []
    if shutil.which("dot"):
        for fmt in ("png", "svg"):
            p = base.with_suffix("." + fmt)
            r = subprocess.run(["dot", "-T" + fmt, "-Gdpi=130", str(base.with_suffix(".dot")), "-o", str(p)], capture_output=True)
            if r.returncode == 0:
                out.append(str(p))
    return out


def draw_fallback(comp, base):
    """matplotlib layered drawing used when Graphviz is not installed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    els = {e["id"]: e for e in comp["elements"]}
    pred = {f: set() for f in els}
    for r in comp["relations"]:
        for h in r["head"]:
            for t in r["tail"]:
                if h in pred and t in els:
                    pred[h].add(t)
    lvl = {}

    def depth(x, trail=()):
        if x in lvl:
            return lvl[x]
        d = 0
        for q in pred[x]:
            if q not in trail:
                d = max(d, depth(q, trail + (x,)) + 1)
        lvl[x] = d
        return d
    for f in els:
        depth(f)
    cols = {}
    for f, l in lvl.items():
        cols.setdefault(l, []).append(f)
    pos = {}
    for l, fs in cols.items():
        for i, f in enumerate(sorted(fs)):
            pos[f] = (l * 3.0, -i * 1.4)
    fig, ax = plt.subplots(figsize=(3 + 3 * len(cols), 2 + 1.4 * max(len(v) for v in cols.values())))
    for r in comp["relations"]:
        for t in r["tail"]:
            for h in r["head"]:
                if t in pos and h in pos:
                    ax.annotate("", xy=pos[h], xytext=pos[t], arrowprops=dict(arrowstyle="->"))
    for f, (x, y) in pos.items():
        ax.text(x, y, _label(els[f]).replace("\\n", "\n"), ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="round", fc=COL.get(els[f]["origin"], "#dddddd")))
    ax.axis("off")
    base = Path(base)
    out = []
    for fmt in ("png", "svg"):
        p = base.with_suffix("." + fmt)
        fig.savefig(p, dpi=130, bbox_inches="tight")
        out.append(str(p))
    plt.close(fig)
    return out


def draw_workflow(comp, base, title=""):
    files = render_dot(workflow_dot(comp, title), base)
    return files or draw_fallback(comp, base)


def draw_goals(graph, status, align, base, title=""):
    return render_dot(goals_dot(graph, status, align, title), base)
