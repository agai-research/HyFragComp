"""Experiment instances: request streams with records and events, and controlled record/request pairs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Data.build_events import make_events
from Data.build_records import Templ, fresh_blocks, history, make_templates, perturb, rng_for
from Data.build_requests import Gen, kind_split, steps_of
from hfc.data import INST, jload, jsave
from hfc.model import HGView

SPLIT = {"test": 1, "val": 2, "test2": 3}


def seed_of(seed, split):
    return int(seed) + (100 if split == "val" else 200 if split == "test2" else 0)


def test_signatures(cfg, repo, procs, n_tpl, seeds=range(5)):
    """Procedure signatures used by the reported test templates; validation instances stay away from them."""
    out = set()
    for s in seeds:
        g = Gen(cfg, procs, rng_for(s, SPLIT["test"], 7).integers(1 << 30))
        sc = cfg["streams"]
        for t in make_templates(g, n_tpl, sc["sizes"], sc["size_w"], sc["zipf_eco"], sc["zipf_proc"], sc["q_renew"], sc["q_prio"]):
            out.add(t.sig())
    return out


def users_of(rng, n, k, zipf):
    w = 1.0 / np.arange(1, n + 1) ** zipf
    return [int(x) for x in rng.choice(n, size=k, p=w / w.sum())]


def request_text_opts(cfg, rng):
    sc = cfg["streams"]
    return {"unseen_p": sc["unseen_p"], "n_trap": int(rng.random() < sc["trap_p"]), "n_fill": int(rng.integers(0, sc["fill_max"] + 1))}


# ---------------------------------------------------------------------- request streams
def build_stream(cfg, repo, seed, split="test", n_req=None, warm=True, stale=None, ev_span="stream", procs=None, sizes=None,
                 n_tpl=None, mix=None, tag=""):
    """Recurring request patterns with stored records, then requests perturbed from the patterns, plus change events."""
    sc = cfg["streams"]
    n_req = n_req or sc["test"]
    sd = seed_of(seed, split)
    procs = procs or repo.pids
    gen = Gen(cfg, procs, int(rng_for(sd, SPLIT[split], 7).integers(1 << 30)))
    rng = rng_for(sd, SPLIT[split], 11)
    sizes = sizes or sc["sizes"]
    w = sc["size_w"] if len(sizes) == len(sc["size_w"]) else [1] * len(sizes)
    n_tpl = n_tpl or sc["n_tpl"]
    exclude = test_signatures(cfg, repo, procs, n_tpl) if split == "val" else set()
    tpls = make_templates(gen, n_tpl, sizes, w, sc["zipf_eco"], sc["zipf_proc"], sc["q_renew"], sc["q_prio"], exclude)
    recs = history(gen, tpls, sc["hist_frac"], sc["users"], 0.0) if warm else []
    known = {p["pid"] for t in tpls for p, _ in t.blocks}
    tw = 1.0 / np.arange(1, len(tpls) + 1) ** sc["zipf_proc"]
    tw /= tw.sum()
    us = users_of(rng, sc["users"], n_req, sc["zipf_user"])
    mix = mix or sc["mix"]
    reqs = []
    for i in range(n_req):
        tpl = tpls[int(rng.choice(len(tpls), p=tw))]
        blocks, qual, prio, kind = perturb(gen, tpl, mix, known)
        t = sc["t_start"] + i / sc["rate"]
        q = gen.request("%s%s%d" % (tag, "q", i), t, "u%d" % us[i], blocks, qual, prio, **request_text_opts(cfg, rng))
        q["truth"]["tag"] = kind
        q["truth"]["tpl"] = tpl.tid
        reqs.append(q)
    # events
    evs = []
    st = sc["stale_default"] if stale is None else stale
    if st > 0:
        hg = HGView(repo.base)
        src = recs if recs else history(gen, tpls, 1.0, sc["users"], 0.0)
        hot = sorted({f for r in src for f in r["frags"] if not f.endswith(".ent")})
        n_aff = int(round(st * len(hot)))
        t1 = sc["t_start"] - 0.5 if ev_span == "before" else sc["t_start"] + n_req / sc["rate"]
        t0 = 1.0 if ev_span == "before" else sc["t_start"]
        evs = make_events(rng_for(sd, SPLIT[split], 13), hg, hot, n_aff, t0, t1, sc["ev_mix"], sc["p_bridge"],
                          int(round(sc["subst_frac"] * len(hot))), tag="v")
    return {"seed": seed, "split": split, "size": repo.size, "records": recs, "events": evs, "requests": reqs,
            "meta": {"templates": len(tpls), "stale": st, "n_req": n_req, "n_rec": len(recs), "n_events": len(evs),
                     "sigs": [list(t.sig()) for t in tpls[:20]]}}


# ---------------------------------------------------------------------- record/request pairs with controlled overlap
def split_blocks(rng, G, m):
    """Request kinds for G goals and the shared part: whole blocks, then a prefix of one block."""
    kinds = kind_split(rng, G)
    order = list(rng.permutation(len(kinds)))
    left, shared = m, []
    for i in order:
        steps = steps_of(kinds[i])
        if left >= len(steps):
            shared.append((i, len(steps)))
            left -= len(steps)
        elif left >= 2 and len(steps) > left:
            shared.append((i, left))
            left = 0
    return kinds, shared


def pair(gen, cfg, rid, t, user, G, ov, qual="first", prio=False, eco=None, dist=0, unseen=None, n_trap=0, n_fill=0, hold=()):
    """One request of G goals and the stored record that shares about `ov` of its goals (None when nothing is shared)."""
    rng = gen.rng
    kinds, shared = split_blocks(rng, G, int(round(ov * G)))
    eco, bl = gen.pick_blocks(kinds, eco=eco, zipf=cfg["streams"]["zipf_proc"], exclude=hold)
    sb = []
    for i, n in shared:
        st = steps_of(kinds[i])[:n]
        sb.append((bl[i][0], st))
    m = sum(len(s) for _, s in sb)
    rec = None
    if sb or dist:
        extra = []
        if dist:
            used = {p["pid"] for p, _ in bl} | set(hold)
            _, extra = fresh_blocks(gen, used, ["std"] * 0 + ["light"] * dist, eco=eco)
        rec = gen.record("p" + rid, t - 2.0, user, sb + extra if (sb or extra) else sb, qual, prio, scope="shared", users=[user, "u_a", "u_b"], hits=3)
    unseen_p = cfg["streams"]["unseen_p"] if unseen is None else unseen
    q = gen.request(rid, t, user, bl, qual, prio, unseen_p=unseen_p, n_trap=n_trap, n_fill=n_fill)
    q["truth"]["overlap"] = m / G
    q["truth"]["shared"] = m
    return rec, q


def siblings(gen, rec_blocks, qual, prio, base_cid, t, user):
    """Distractor records over the same procedures with the other qualifier or priority flag."""
    out = []
    if not rec_blocks:
        return out
    out.append(gen.record(base_cid + "q", t, user, rec_blocks, "renew" if qual == "first" else "first", prio, scope="shared", users=[user, "u_a", "u_b"], hits=3))
    out.append(gen.record(base_cid + "p", t, user, rec_blocks, qual, not prio, scope="shared", users=[user, "u_a", "u_b"], hits=3))
    return out


def build_pairs(cfg, repo, seed, G, ov, n, split="test", procs=None, noise=0, with_siblings=False, tag="", unseen=None, salt=0):
    """n independent pairs on distinct economies/procedures of the repository; the memory holds all records."""
    sd = seed_of(seed, split)
    procs = procs or repo.pids
    gen = Gen(cfg, procs, int(rng_for(sd, SPLIT[split], 17, G, salt).integers(1 << 30)))
    rng = rng_for(sd, SPLIT[split], 19, G, salt)
    sc = cfg["streams"]
    recs, reqs = [], []
    used, every = set(), set()
    for i in range(n):
        rid = "%s%dg%d_%d" % (tag, int(ov * 100), G, i)
        qual = "renew" if rng.random() < sc["q_renew"] else "first"
        prio = bool(rng.random() < sc["q_prio"])
        t = sc["t_start"] + i / sc["rate"]
        rec, q = pair(gen, cfg, rid, t, "u%d" % int(rng.integers(0, sc["users"])), G, ov, qual, prio, None, 0, unseen,
                      int(rng.random() < sc["trap_p"]), int(rng.integers(0, sc["fill_max"] + 1)), hold=used)
        used |= {b["pid"] for b in q["truth"]["blocks"]}
        every |= {b["pid"] for b in q["truth"]["blocks"]}
        if len(used) > 0.8 * len(procs):
            used = set()
        if rec is not None and rec["blocks"]:
            recs.append(rec)
            if with_siblings:
                bl = [(next(p for p in gen.procs if p["pid"] == b["pid"]), b["steps"]) for b in rec["blocks"]]
                recs += siblings(gen, bl, qual, prio, rec["cid"], t - 2.0, rec["user"])
        reqs.append(q)
    for j in range(noise):
        eco, bl = gen.pick_blocks(kind_split(rng, int(rng.choice([3, 4, 5, 8]))), zipf=sc["zipf_proc"], exclude=every)
        recs.append(gen.record("n%d" % j, 0.0, "u0", bl, "first", False, scope="shared", users=["u0", "u1", "u2"], hits=3))
    return {"seed": seed, "split": split, "size": repo.size, "records": recs, "events": [], "requests": reqs,
            "meta": {"G": G, "ov": ov, "n": n, "n_rec": len(recs)}}


def mixed_overlap(cfg, repo, seed, G_lo, G_hi, n, split="test", procs=None, tag=""):
    """Requests of random size in [G_lo, G_hi] with overlap drawn uniformly from 0..1 (distribution of rho)."""
    sd = seed_of(seed, split)
    procs = procs or repo.pids
    gen = Gen(cfg, procs, int(rng_for(sd, SPLIT[split], 23, G_lo).integers(1 << 30)))
    rng = rng_for(sd, SPLIT[split], 29, G_lo)
    sc = cfg["streams"]
    recs, reqs, used = [], [], set()
    for i in range(n):
        G = int(rng.integers(G_lo, G_hi + 1))
        ov = float(rng.uniform(0.0, 1.0))
        qual = "renew" if rng.random() < sc["q_renew"] else "first"
        rec, q = pair(gen, cfg, "%sb%d_%d" % (tag, G_lo, i), sc["t_start"] + i / sc["rate"], "u%d" % int(rng.integers(0, sc["users"])), G, ov, qual, False,
                      None, 1, None, int(rng.random() < sc["trap_p"]), int(rng.integers(0, sc["fill_max"] + 1)), hold=used)
        used |= {b["pid"] for b in q["truth"]["blocks"]}
        if len(used) > 0.8 * len(procs):
            used = set()
        if rec is not None and rec["blocks"]:
            recs.append(rec)
        reqs.append(q)
    return {"seed": seed, "split": split, "size": repo.size, "records": recs, "events": [], "requests": reqs,
            "meta": {"G_lo": G_lo, "G_hi": G_hi, "n": n}}


def build_length(cfg, repo, seed, words, n, split="test", procs=None):
    """Requests of one to three blocks whose text is padded to about `words` words with filler that adds no goal."""
    sd = seed_of(seed, split)
    procs = procs or repo.pids
    gen = Gen(cfg, procs, int(rng_for(sd, SPLIT[split], 31, words).integers(1 << 30)))
    rng = rng_for(sd, SPLIT[split], 37, words)
    sc = cfg["streams"]
    reqs = []
    for i in range(n):
        G = int(rng.choice([3, 4, 5, 8]))
        kinds = kind_split(rng, G)
        _, bl = gen.pick_blocks(kinds, zipf=sc["zipf_proc"])
        qual = "renew" if rng.random() < sc["q_renew"] else "first"
        prio = bool(rng.random() < sc["q_prio"])
        q = gen.request("w%d_%d" % (words, i), sc["t_start"] + i / sc["rate"], "u%d" % int(rng.integers(0, sc["users"])), bl, qual, prio,
                        words=None if words <= 12 else words, unseen_p=sc["unseen_p"], n_trap=int(rng.random() < sc["trap_p"]) if words > 12 else 0,
                        terse=words <= 12)
        q["truth"]["words"] = q["truth"]["n_words"]
        reqs.append(q)
    return {"seed": seed, "split": split, "size": repo.size, "records": [], "events": [], "requests": reqs, "meta": {"words": words, "n": n}}


# ---------------------------------------------------------------------- storage
def save(exp, name, inst):
    p = INST / exp / (name + ".json")
    jsave(inst, p)
    return p


def load(exp, name):
    return jload(INST / exp / (name + ".json"))
