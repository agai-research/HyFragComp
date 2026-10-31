"""Algorithm 1: recall, exact alignment, difference, re-validation (V1-V5), variant switch and reuse mode."""
import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import enc


@dataclass
class Plan:
    mode: str = "scratch"
    anchor: str = ""
    align: dict = field(default_factory=dict)       # gid -> (cid, gid', (m1, m2, m3))
    retained: dict = field(default_factory=dict)    # gid -> tuple of fids (after switches)
    status: dict = field(default_factory=dict)      # gid -> reuse | switch
    gdelta: list = field(default_factory=list)
    failed: dict = field(default_factory=dict)      # gid -> reason
    rho: float = 0.0
    delta: dict = field(default_factory=dict)
    witnesses: dict = field(default_factory=dict)
    recs: set = field(default_factory=set)
    scope_rej: int = 0


def pseudo(cid):
    return "c-" + hashlib.blake2b(cid.encode(), digest_size=4).hexdigest()


class Reuser:
    def __init__(self, world, mem, chk, cfg, cnt, revalidate=True, single_record=False):
        self.w, self.mem, self.chk, self.cfg, self.cnt = world, mem, chk, cfg, cnt
        self.hg, self.onto = world.hg, world.onto
        self.mc = cfg["match"]
        self.reval = revalidate
        self.single = single_record
        self.st = None
        self.on_suspect = None
        self.scope_rej = 0

    def _tic(self, name):
        return self.st.tic(name) if self.st is not None else None

    def _toc(self, h):
        if h is not None:
            self.st.toc(h)

    # ------------------------------------------------------------------ alignment
    def score_matrix(self, E, Z, goals, r):
        l1, l2, l3 = self.mc["lambdas"]
        M = l1 * (E @ r.e.T) + l2 * (Z @ r.z.T)
        sl = np.zeros_like(M)
        for i, g in enumerate(goals):
            for j, g2 in enumerate(r.goals):
                sl[i, j] = 1.0 if tuple(sorted(g.slots)) == tuple(sorted(g2.slots)) else 0.0
        return M + l3 * sl, (l1 * (E @ r.e.T), l2 * (Z @ r.z.T), l3 * sl)

    def align(self, graph, E, Z, user, t):
        goals = graph.goals
        n = len(goals)
        theta = self.mc["theta_m"]
        a = self.w.models.alpha
        Q = np.stack([enc.fuse(E[i], Z[i], a) for i in range(n)])
        votes = {}
        for i in range(n):
            for s, (cid, j) in self.mem.recall(Q[i], self.mc["K_a"], t):
                votes.setdefault(cid, 0.0)
                votes[cid] += s
        self.cnt["ann"] += n
        cands = sorted(votes, key=lambda c: -votes[c])[:8]
        per = {}
        best_pair = {}
        vis = []
        for cid in cands:
            r = self.mem.recs[cid]
            if r.kind == "rec" and r.scope == "u" and r.owner != user:
                # V5 at selection time: a private record of another requester is never an anchor; it only counts towards sharing
                self.cnt["V5"] += 1
                self.mem.note(r, user, t)
                self.scope_rej += 1
                continue
            vis.append(cid)
        cands = vis
        for cid in cands:
            r = self.mem.recs[cid]
            M, parts = self.score_matrix(E, Z, goals, r)
            W = np.where(M >= theta, M, 0.0)
            ri, ci = linear_sum_assignment(-W)
            pairs = {int(i): int(j) for i, j in zip(ri, ci) if W[i, j] > 0}
            per[cid] = (pairs, M, parts, float(sum(W[i, j] for i, j in pairs.items())) / n)
            for i in range(n):
                if W[i].max() > 0:
                    j = int(W[i].argmax())
                    if i not in best_pair or W[i, j] > best_pair[i][0]:
                        best_pair[i] = (float(W[i, j]), cid, j)
        return cands, per, best_pair

    # ------------------------------------------------------------------ main
    def plan(self, ctx, graph, E, Z, user, t):
        goals = graph.goals
        n = len(goals)
        P = Plan()
        h = self._tic("recall")
        cands, per, best_pair = self.align(graph, E, Z, user, t)
        self._toc(h)
        if not cands:
            P.gdelta = [g.gid for g in goals]
            P.delta = {"G=": [], "G+": P.gdelta, "G-": [], "E+": len(graph.edges), "E-": 0}
            return P
        anchor = max(cands, key=lambda c: per[c][3])
        P.anchor = anchor
        pairs, M, parts, s = per[anchor]
        align = {}
        for i, j in pairs.items():
            align[goals[i].gid] = (anchor, j, (float(parts[0][i, j]), float(parts[1][i, j]), float(parts[2][i, j])))
        if not self.single:
            for i in range(n):
                g = goals[i].gid
                if g not in align and i in best_pair:
                    m, cid, j = best_pair[i]
                    _, Mx, px, _ = per[cid]
                    align[g] = (cid, j, (float(px[0][i, j]), float(px[1][i, j]), float(px[2][i, j])))
        P.align = align
        P.scope_rej = self.scope_rej
        self._delta(P, graph, align)
        order = graph.order()
        retained, status, failed, wit = {}, {}, {}, {}
        pending = set(g.gid for g in goals if g.gid not in align)
        pend_omega = {g.gid: g.omega for g in goals}
        gmap = {g.gid: g for g in goals}
        avail_ret = set()
        for gid in order:
            g = gmap[gid]
            if gid not in align:
                continue
            cid, j, m = align[gid]
            r = self.mem.recs[cid]
            if r.kind == "rec" and r.scope == "u" and r.owner != user:
                self.mem.note(r, user, t)
            fids = tuple(r.phi[j])
            w = {"record": pseudo(cid), "m": [round(x, 3) for x in m], "tier": "H" if r.kind == "cmp" else "C_e", "t": t, "nu": r.nu}
            if self.on_suspect is not None and r.kind == "cmp":
                self.on_suspect(r)
            h = self._tic("revalidation")
            if self.reval:
                ok, why, new, sw = self.check(ctx, g, r, fids, user, t, avail_ret, [pend_omega[x] for x in pending if x != gid], w)
                if sw:
                    status[gid] = "switch"
                else:
                    status[gid] = "reuse"
            else:
                ok, why, new = True, "", fids
                status[gid] = "reuse"
            self._toc(h)
            if ok:
                retained[gid] = tuple(new)
                for f in new:
                    fr = self.hg.frag(f)
                    if fr is not None:
                        avail_ret |= set(fr.post)
                P.recs.add(cid)
                w["status"] = status[gid]
                w["stamps"] = {f: r.stamps.get(f) for f in fids}
            else:
                failed[gid] = why
                pending.add(gid)
                w["status"] = "new"
                w["failed"] = why
                status.pop(gid, None)
                if why == "V5":
                    P.scope_rej += 1
            wit[gid] = w
        for gid in pending - set(align):
            wit[gid] = {"status": "new", "failed": "no-match"}
        P.retained, P.status, P.failed, P.witnesses = retained, status, failed, wit
        P.gdelta = [g.gid for g in goals if g.gid not in retained]
        P.rho = len(retained) / n
        full = not P.gdelta and P.delta["E+"] == 0 and P.delta["E-"] == 0
        P.mode = "full" if full else ("partial" if P.rho >= self.mc["rho_min"] else "scratch")
        if P.mode == "scratch":
            P.retained, P.status, P.gdelta, P.rho = {}, {}, [g.gid for g in goals], 0.0
        return P

    def _delta(self, P, graph, align):
        """Difference between the request goal graph and the anchor record (hyperedge mapping)."""
        anchor = P.anchor
        r = self.mem.recs[anchor]
        mp = {g: j for g, (c, j, _) in align.items() if c == anchor}
        inv = {j: g for g, j in mp.items()}
        Gp = [g.gid for g in graph.goals if g.gid not in mp]
        Gm = [j for j in range(len(r.goals)) if j not in inv]
        stored = {(tuple(sorted(a)), tuple(sorted(b)), p) for a, b, p in r.edges}
        e_eq = 0
        for tl, hd, pat in graph.edges:
            if all(x in mp for x in tl) and all(x in mp for x in hd):
                key = (tuple(sorted(mp[x] for x in tl)), tuple(sorted(mp[x] for x in hd)), pat)
                if key in stored:
                    e_eq += 1
        P.delta = {"G=": sorted(mp), "G+": Gp, "G-": Gm, "E+": len(graph.edges) - e_eq, "E-": len(stored) - e_eq}

    # ------------------------------------------------------------------ V1-V5
    def check(self, ctx, goal, r, fids, user, t, avail_ret, pend_omegas, w):
        """Returns (ok, failed check, fragments to use, switched?)."""
        cnt = self.cnt
        hg, onto, chk = self.hg, self.onto, self.chk
        frs = [hg.frag(f) for f in fids]
        # V1 validity
        cnt["V1"] += 1
        if any(f is None or not (f.nu[0] <= t < f.nu[1]) for f in frs):
            return False, "V1", fids, False
        # V2 freshness: substitutable revision and licensing hyperedges persist
        cnt["V2"] += 1
        for f in fids:
            rev = r.stamps.get(f)
            if rev is None or not hg.stamps_ok(f, rev, onto):
                return False, "V2", fids, False
        for u, v, eid in r.conns:
            if v in fids:
                e = hg.edge(eid)
                if e is None or r.esig.get(eid) != (e.pat, e.guard):
                    return False, "V2", fids, False
        res = self._v34(ctx, goal, frs, avail_ret, pend_omegas, user, t, r, w)
        if res[0]:
            return True, "", fids, False
        ok, why, extra = res
        if why == "V4g":
            # variant switch: a sibling of the XOR group whose guard holds replaces the member
            new = []
            for f in frs:
                sw = self._sibling(f, ctx, t)
                new.append(sw if sw is not None else f)
            if any(a.fid != b.fid for a, b in zip(new, frs)):
                res2 = self._v34(ctx, goal, new, avail_ret, pend_omegas, user, t, r, w, switched=True)
                if res2[0]:
                    w["switched_from"] = [f.fid for f in frs]
                    return True, "", tuple(f.fid for f in new), True
                return False, res2[1].replace("V4g", "V4"), fids, False
            return False, "V4", fids, False
        return False, why, fids, False

    def _sibling(self, f, ctx, t):
        for e in self.hg.edges_head(f.fid):
            if e.pat == "xor" and e.guard:
                for s in self.chk.selected(e, ctx):
                    if s != f.fid:
                        g = self.hg.frag(s)
                        if g is not None and g.nu[0] <= t < g.nu[1]:
                            return g
        return None

    def _v34(self, ctx, goal, frs, avail_ret, pend_omegas, user, t, r, w, switched=False):
        cnt, chk, onto = self.cnt, self.chk, self.onto
        # V3 entailment
        cnt["V3"] += 1
        av = set(ctx.atoms) | set(avail_ret)
        for om in pend_omegas:
            av |= set(om)
        ent = self.hg.frag(ctx.jur + ".ent")
        if ent is not None:
            av |= set(ent.post)
        post = set()
        pend = list(frs)
        while pend:
            moved = False
            for f in list(pend):
                if onto.entails(frozenset(av) | frozenset(post), f.pre):
                    post |= set(f.post)
                    pend.remove(f)
                    moved = True
            if not moved:
                return False, "V3", None
        if not goal.omega <= onto.close(frozenset(post)):
            return False, "V3", None
        # V4 compatibility: roles, data, window, guard
        cnt["V4"] += 1
        data = set(ctx.data)
        for f in frs:
            if not chk.chi_rol(f, ctx):
                return False, "V4", None
            if not chk.chi_dat(f, frozenset(data)):
                return False, "V4", None
            data |= set(f.out)
            if not chk.chi_tmp(f, ctx, t, 0):
                return False, "V4", None
            if f.guard:
                if not chk.guard_true(f.guard, ctx):
                    return False, "V4g", None
            else:
                for e in self.hg.edges_head(f.fid):
                    if e.pat == "xor" and e.guard and f.fid not in chk.selected(e, ctx):
                        return False, "V4g", None
        # V5 scope
        cnt["V5"] += 1
        if r.kind == "rec" and r.scope == "u" and r.owner != user:
            return False, "V5", None
        return True, "", None
