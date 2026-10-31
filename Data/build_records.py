"""Construct request templates, stored composition records and perturbed requests for the streams."""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Data.build_requests import Gen, STEPS, kind_split, steps_of, zipf_weights


@dataclass
class Templ:
    """A recurring request pattern: blocks of (procedure, kind), the permit qualifier and the priority flag."""
    tid: int
    blocks: list
    qual: str
    prio: bool

    def sig(self):
        return tuple(sorted(p["pid"] for p, _ in self.blocks))


def rng_for(seed, *tag):
    return np.random.default_rng([int(seed)] + [int(x) for x in tag])


def draw_size(rng, sizes, weights):
    w = np.array(weights, dtype=float)
    return int(sizes[int(rng.choice(len(sizes), p=w / w.sum()))])


def make_templates(gen, n, sizes, weights, zipf_eco, zipf_proc, q_renew, q_prio, exclude=()):
    """Templates with distinct procedure signatures that are not in `exclude`."""
    rng = gen.rng
    out, seen = [], set(exclude)
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        kinds = kind_split(rng, draw_size(rng, sizes, weights))
        eco, bl = gen.pick_blocks(kinds, zipf=zipf_eco)
        qual = "renew" if rng.random() < q_renew else "first"
        t = Templ(len(out), bl, qual, bool(rng.random() < q_prio))
        if t.sig() in seen:
            continue
        seen.add(t.sig())
        out.append(t)
    return out


def history(gen, tpls, frac, users, t0=0.0, hits=3):
    """Stored shared records for the first `frac` share of the templates (the most popular ones)."""
    rng = gen.rng
    n = int(round(frac * len(tpls)))
    recs = []
    for i, t in enumerate(tpls[:n]):
        who = sorted(set(int(x) for x in rng.integers(0, users, size=3)))
        recs.append(gen.record("h%d" % i, t0 - 1.0 - i * 0.01, "u%d" % who[0], t.blocks, t.qual, t.prio, scope="shared",
                               users=["u%d" % x for x in who], hits=hits))
    return recs


def fresh_blocks(gen, exclude_pids, kinds, eco=None):
    for _ in range(40):
        e, bl = gen.pick_blocks(kinds, eco=eco, zipf=gen.cfg["streams"]["zipf_proc"], exclude=exclude_pids)
        if not set(p["pid"] for p, _ in bl) & set(exclude_pids):
            return e, bl
    return e, bl


def perturb(gen, tpl, mix, known):
    """Request blocks, qualifier, priority flag and the perturbation tag drawn from the mix."""
    rng = gen.rng
    keys = list(mix)
    w = np.array([mix[k] for k in keys], dtype=float)
    kind = keys[int(rng.choice(len(keys), p=w / w.sum()))]
    blocks, qual, prio = list(tpl.blocks), tpl.qual, tpl.prio
    iso = blocks[0][0]["iso3"]
    if kind == "prio_flip":
        prio = not prio
    elif kind == "qual_flip":
        qual = "renew" if qual == "first" else "first"
    elif kind == "extend":
        used = {p["pid"] for p, _ in blocks}
        _, nb = fresh_blocks(gen, used, [("light", "std", "ext")[int(rng.integers(3))]], eco=iso)
        blocks = blocks + nb
    elif kind == "subset":
        if len(blocks) > 1:
            blocks = blocks[:-1]
        else:
            kind = "same"
    elif kind == "new":
        sz = int(rng.choice([3, 4, 5, 8]))
        _, blocks = fresh_blocks(gen, known, kind_split(rng, sz))
        qual = "renew" if rng.random() < 0.4 else "first"
        prio = bool(rng.random() < 0.25)
    return blocks, qual, prio, kind
