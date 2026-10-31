"""GraphRAG candidate construction and anchored delta recomposition with incremental verification (Algorithm 2)."""
from dataclasses import dataclass, field

import numpy as np

from .chk import Checks


@dataclass
class Om:
    """A partial composition in the beam."""
    fset: frozenset
    cover: dict
    eta: float = 0.0
    log: list = field(default_factory=list)


class GraphRAG:
    """Vector retrieval over Z, hypergraph expansion, compatibility filtering, eta ranking and context assembly."""

    def __init__(self, world, chk, cfg):
        self.w, self.hg, self.chk, self.cfg = world, world.hg, chk, cfg
        self.dc = cfg["delta"]
        self.ic = cfg["index"]
        self.eta_w = self.dc["eta"]

    def _fresh(self, fid, t):
        """A suspect composite is re-validated (and repaired or evicted) at first access, as in Algorithm 3."""
        if fid not in self.w.suspect:
            return True
        cb = getattr(self.w, "lazy_cb", None)
        if cb is None:
            return True
        if fid in self.w.busy:
            return False          # the composite under repair, or a suspect one met during a repair, is not a candidate
        self.w.busy.add(fid)
        try:
            cb(fid, t)
        finally:
            self.w.busy.discard(fid)
        return fid not in self.w.suspect

    # ------------------------------------------------------------------ steps 1-2
    def seeds(self, qvec, jur, t, K):
        hg = self.hg
        base = self.w.repo

        def pred(fid):
            if not self._fresh(fid, t):
                return False
            f = hg.frag(fid)
            return f is not None and f.nu[0] <= t < f.nu[1] and fid.startswith(jur + ".")
        hits = self.w.fidx.knn(qvec, K, ef=self.ic["ef_search"], pred=pred)
        self.chk.cnt["ann"] += 1
        return [(s, f) for s, f in hits if s >= self.dc["theta_sim"]]

    def expand(self, seeds, h, t):
        """Neighbourhood expansion up to depth h: tails (backward), heads and sibling heads (forward); never from ENT."""
        hg = self.hg
        seen = {f for _, f in seeds}
        front = set(seen)
        for _ in range(h):
            nxt = set()
            for x in front:
                fr = hg.frag(x)
                if fr is None or fr.ftype == "ENT":
                    continue
                for e in hg.edges_head(x):
                    if e.pat == "der":
                        continue
                    for y in list(e.tail) + list(e.head):
                        nxt.add(y)
                for e in hg.edges_tail(x):
                    if e.pat == "der":
                        continue
                    for y in e.head:
                        nxt.add(y)
            nxt -= seen
            seen |= nxt
            front = nxt
        out = []
        for y in seen:
            if not self._fresh(y, t):
                continue
            f = hg.frag(y)
            if f is not None and f.ftype != "ENT" and f.nu[0] <= t < f.nu[1]:
                out.append(y)
        return out

    def pool(self, goal, qvec, jur, t, K):
        sd = self.seeds(qvec, jur, t, K)
        return self.expand(sd, self.dc["h"], t), dict((f, s) for s, f in sd)

    # ------------------------------------------------------------------ steps 3-4
    def avail(self, fset, ctx):
        a = set(ctx.atoms)
        d = set(ctx.data)
        for x in fset:
            f = self.hg.frag(x)
            if f is not None:
                a |= f.post
                d |= f.out
        return self.w.onto.close(frozenset(a)), frozenset(d)

    def anc_avail(self, fid, fset, ctx):
        """Atoms and data that actually reach `fid`: outputs of the ancestors reached through its licensing tails inside fset."""
        hg = self.hg
        seen, stack = set(), [fid]
        while stack:
            v = stack.pop()
            for e in hg.edges_head(v):
                if e.pat in ("der", "loop") or not set(e.tail) <= fset:
                    continue
                if e.pat == "xor" and e.guard and v not in self.chk.selected(e, ctx):
                    continue
                for u in e.tail:
                    if u not in seen:
                        seen.add(u)
                        stack.append(u)
        a, d = set(ctx.atoms), set(ctx.data)
        for x in seen:
            g = hg.frag(x)
            if g is not None:
                a |= g.post
                d |= g.out
        return self.w.onto.close(frozenset(a)), frozenset(d)

    def licensed(self, f, fset, ctx):
        for e in self.hg.edges_head(f.fid):
            if e.pat in ("der", "loop") or not set(e.tail) <= fset:
                continue
            if e.pat == "xor" and e.guard and f.fid not in self.chk.selected(e, ctx):
                continue
            return True
        return False

    def screen(self, fid, fset, ctx, t, av, dt):
        """The four chi predicates plus guard and licensing; returns (ok, reason)."""
        f = self.hg.frag(fid)
        if f is None:
            return False, "gone"
        c = self.chk
        if f.ftype == "VAR" and f.guard is not None:
            for e in self.hg.edges_head(fid):
                if e.pat == "xor" and e.guard and fid not in c.selected(e, ctx):
                    return False, "guard"
        if not self.licensed(f, fset, ctx):
            return False, "c4"
        av, dt = self.anc_avail(fid, fset, ctx)
        if not c.chi_fun(f, av):
            return False, "chi_fun"
        if not c.chi_dat(f, dt):
            return False, "chi_dat"
        if not c.chi_rol(f, ctx):
            return False, "chi_rol"
        if not c.chi_tmp(f, ctx, t, 0):
            return False, "chi_tmp"
        return True, ""

    def eta(self, f, g_need, unc_goals, qvec, ctx, vec):
        w = self.eta_w
        gain = 0.0
        post = self.w.onto.close(f.post)
        for om in unc_goals:
            if om:
                gain += len(om & post) / len(om)
        gain /= max(len(unc_goals), 1)
        sim = float(vec @ qvec) if vec is not None else 0.0
        qos = 1.0 / (1.0 + f.dmin)
        margin = max(0.0, 1.0 - f.dmin / max(ctx.deadline, 1))
        return w["w_gain"] * gain + w["w_sim"] * sim + w["w_qos"] * qos + w["w_margin"] * margin

    # ------------------------------------------------------------------ step 5
    def context_text(self, goal, cands, rejected, fset):
        lines = ["subgoal %s end-state %s" % (goal.text, sorted(goal.omega)), "anchor %s" % sorted(fset)]
        for c in cands:
            f = self.hg.frag(c["id"])
            lines.append("fragment %s pre=%s post=%s roles=%s text=%s" % (f.fid, sorted(f.pre), sorted(f.post), sorted(f.roles), f.text))
        for fid, why in rejected:
            lines.append("rejected %s because %s" % (fid, why))
        return "\n".join(lines)


class Composer:
    """Algorithm 2: beam search over anchored extensions, junction checks and bounded repair."""

    def __init__(self, world, roles, cfg, st, cnt):
        self.w, self.roles, self.cfg, self.st, self.cnt = world, roles, cfg, st, cnt
        self.hg = world.hg
        self.chk = Checks(world, cnt)
        self.rag = GraphRAG(world, self.chk, cfg)
        self.dc = cfg["delta"]

    def run(self, ctx, graph, gv, retained, gdelta, t, incremental=True, use_ret_call=False, max_rounds=None, prev_in=None):
        """Returns a dict with ok, fset, conns, cover, repairs, rejections and junction list."""
        hg, chk, rag, st = self.hg, self.chk, self.rag, self.st
        Bw, K, eps = self.dc["B_w"], self.dc["K"], max_rounds if max_rounds is not None else self.dc["eps_max"]
        iso = ctx.jur
        ent = iso + ".ent"
        cover0 = {g: tuple(f) for g, f in retained.items()}
        fset0 = frozenset({ent} | {x for fs in cover0.values() for x in fs})
        goals = {g.gid: g for g in graph.goals}
        e_vec, z_vec = gv
        from . import enc
        qv = {g.gid: enc.fuse(e_vec[i], z_vec[i], self.w.models.alpha) for i, g in enumerate(graph.goals)}
        order = [g for g in graph.order() if g in gdelta]
        beam = [Om(fset0, dict(cover0))]
        rej_all, repairs, charged, pools = [], 0, set(), {}
        rg = {}   # repair rounds spent per blocked sub-goal; eps_max bounds each of them
        t_rag = st.tic("recompose")
        while True:
            unc = [g for g in order if g not in beam[0].cover or not self._covered(goals[g], beam[0].cover[g], beam[0].fset)]
            if not unc:
                break
            g = unc[0]
            goal = goals[g]
            if g not in pools:
                if use_ret_call and not getattr(self, "_ret", False):
                    self.roles.reformulate(graph, st)
                    self._ret = True
                pools[g] = rag.pool(goal, qv[g], iso, t, K)
            nxt, rejected, cand_log = [], [], []
            for om in beam:
                av, dt = rag.avail(om.fset, ctx)
                need = goal.omega - av
                if not need:
                    nxt.append(om)
                    continue
                unc_om = [goals[x].omega - av for x in order if x not in om.cover]
                found = []
                for fid in pools[g][0]:
                    f = hg.frag(fid)
                    if f is None or fid in om.fset:
                        continue
                    if not (rag.w.onto.close(f.post) & need):
                        continue
                    ok, why = rag.screen(fid, om.fset, ctx, t, av, dt)
                    if not ok:
                        rejected.append((fid, why))
                        continue
                    et = rag.eta(f, need, unc_om, qv[g], ctx, self.w.vec(fid) if True else None)
                    found.append((et, fid))
                found.sort(reverse=True)
                if om is beam[0]:
                    cand_log = [{"id": fid, "eta": et, "text": hg.frag(fid).text} for et, fid in found]
                for et, fid in found[:K]:
                    cov = dict(om.cover)
                    fs2 = om.fset | {fid}
                    nxt.append(Om(fs2, cov, om.eta + et, om.log + [(g, fid)]))
            rej_all += rejected
            if g not in charged:
                charged.add(g)
                ctx_text = rag.context_text(goal, cand_log, rejected, beam[0].fset)
                k = self.roles.select(goal, cand_log, ctx_text, st) if cand_log else None
                if k is not None:
                    for om in nxt:
                        if om.log and om.log[-1][1] == cand_log[k]["id"] and om.log[-1][0] == g:
                            om.eta += self.dc["eta"].get("w_sel", 0.5)
            if not nxt:
                # beam exhausted: bounded repair with a bridging fragment
                if rg.get(g, 0) >= eps:
                    st.toc(t_rag)
                    return {"ok": False, "reason": "exhausted:" + goal.text[:40], "rejections": rej_all, "repairs": repairs}
                repairs += 1
                rg[g] = rg.get(g, 0) + 1
                bridged = self._bridge(beam[0], g, goal, ctx, t, pools[g][0])
                self.roles.repair_note([{"goal": g, "rejected": rej_all[-6:]}], goal.text, st)
                if bridged is None:
                    st.toc(t_rag)
                    return {"ok": False, "reason": "no-bridge:" + goal.text[:40], "rejections": rej_all, "repairs": repairs}
                beam = [bridged]
                continue
            # incidental coverage of other goals and pruning
            for om in nxt:
                self._cover_update(om, goals, order)
            nxt.sort(key=lambda o: -o.eta)
            beam = nxt[:Bw]
        st.toc(t_rag)
        best = max(beam, key=lambda o: o.eta)
        if best.fset != fset0:
            # the model states only the part it adds or rewires; retained fragments are copied by the system
            self.roles.compose("%d new fragments, %d anchored" % (len(best.fset - fset0), len(fset0)), {"fragments": sorted(best.fset - fset0)}, st)
        return self._finish(best, ctx, graph, goals, set(fset0), cover0, t, incremental, rej_all, repairs, gdelta, prev_in)

    # ------------------------------------------------------------------ helpers
    def _covered(self, goal, cover, fset):
        post = set()
        for x in cover:
            f = self.hg.frag(x)
            if f is not None and x in fset:
                post |= self.w.onto.close(f.post)
        return goal.omega <= post

    def _cover_update(self, om, goals, order):
        onto = self.w.onto
        for g in order:
            if g in om.cover and self._covered(goals[g], om.cover[g], om.fset):
                continue
            prov = []
            got = set()
            for x in om.fset:
                f = self.hg.frag(x)
                if f is None or f.ftype == "ENT":
                    continue
                inter = onto.close(f.post) & goals[g].omega
                if inter:
                    prov.append(x)
                    got |= inter
            if goals[g].omega <= got:
                om.cover[g] = tuple(sorted(prov))
            elif g in om.cover:
                del om.cover[g]

    def _bridge(self, om, g, goal, ctx, t, pool):
        """Add the fragments that supply a missing precondition or complete a rule tail of the best blocked candidate."""
        hg, rag = self.hg, self.rag
        av, dt = rag.avail(om.fset, ctx)
        need = goal.omega - av
        best = None
        for fid in pool:
            f = hg.frag(fid)
            if f is None or not (self.w.onto.close(f.post) & need):
                continue
            if f.roles - ctx.roles:
                continue
            miss = f.pre - av
            miss_d = f.inp - dt
            for e in hg.edges_head(fid):
                if e.pat in ("der", "loop"):
                    continue
                tl = [x for x in e.tail if x not in om.fset]
                for x in tl:
                    fx = hg.frag(x)
                    if fx is None:
                        continue
                    gain = len(self.w.onto.close(fx.post) & miss) + len(fx.out & miss_d)
                    if not gain and (miss or miss_d):
                        continue
                    ok, why = rag.screen(x, om.fset, ctx, t, av, dt)
                    if ok and (best is None or gain > best[0]):
                        best = (gain, x)
        if best is None:
            return None
        fs = om.fset | {best[1]}
        nom = Om(fs, dict(om.cover), om.eta, om.log + [("bridge", best[1])])
        return nom

    def _finish(self, om, ctx, graph, goals, retained_f, cover0, t, incremental, rej, repairs, gdelta, prev_in=None):
        chk, hg = self.chk, self.hg
        th = self.st.tic("verify")
        fset = set(om.fset)
        conns, bad = chk.derive_conns(frozenset(fset), ctx)
        cover = dict(cover0)
        cover.update({g: c for g, c in om.cover.items()})
        new = fset - retained_f
        touched = set(new)
        for u, v, _ in conns:
            if u in new:
                touched.add(v)
        if prev_in is not None:
            now = {}
            for u, v, eid in conns:
                now.setdefault(v, set()).add((u, eid))
            for v in retained_f:
                if v in now and now[v] != prev_in.get(v, set()):
                    touched.add(v)
        junc = [(u, v) for u, v, _ in conns if (u in touched) or (v in touched)]
        fe = chk.feasibility(graph, cover, frozenset(fset), conns, ctx, t, touched=touched, incremental=incremental)
        rounds = repairs
        ok = all(fe.values()) and not bad
        self.st.toc(th)
        return {"ok": ok, "fset": frozenset(fset), "conns": conns, "cover": cover, "feas": fe, "repairs": repairs,
                "rejections": rej, "junctions": junc, "unlicensed": bad, "new": new}
