"""Frequency-gated promotion of admissible regions into composite fragments."""
from collections import defaultdict

import numpy as np

from .mem import CountMin, regions, region_sig, lift, Rec, expand_atomic
from .model import Goal


class Promoter:
    def __init__(self, sysm):
        self.s = sysm
        mc = sysm.cfg["memory"]
        self.mc = mc
        self.sketch = CountMin(mc["sketch_width"], mc["sketch_depth"], mc["W_s"], seed=sysm.seed)
        self.users = defaultdict(set)
        self.cid_of = {}
        self.sig_of = {}
        self.members = {}
        self.cmp_rec = {}
        self.promoted = 0

    # cost model in call-equivalents: composing R from scratch, reusing the lifted composite once, re-validating after a change
    def c_cmp(self, R):
        return self.mc["c_cmp_unit"] * len(R)

    def c_reu(self, R):
        return self.mc["c_reu_unit"]

    def c_rev(self, R):
        return self.mc["c_rev_unit"] * len(R)

    def benefit(self, R, n):
        lam = sum(self.s.w.lam.get(f, 0.0) for f in R)
        return n * (self.c_cmp(R) - self.c_reu(R)) - lam * self.c_rev(R)

    def utility(self, fid):
        sig = self.sig_of.get(fid)
        if sig is None:
            return 0.0
        R = self.members[fid]
        return self.sketch.est(sig) * (self.c_cmp(R) - self.c_reu(R))

    def observe(self, req, graph, E, Z, fset, conns, cover, st, cnt):
        """Count every admissible region of a verified workflow; promote those that pass the gates."""
        s = self.s
        h = st.tic("promotion")
        self.sketch.tick()
        out = []
        for P, R in regions(s.w.hg, fset, conns, self.mc["R_max"]):
            sig = region_sig(R, conns)
            cnt["promo"] += 1
            self.sketch.add(sig)
            self.users[sig].add(req.user)
            n = self.sketch.est(sig)
            cid = self.cid_of.get(sig)
            if cid is not None and s.w.hg.frag(cid) is not None:
                continue
            if n >= self.mc["k"] and len(self.users[sig]) >= self.mc["k_u"] and self.benefit(R, n) > 0:
                c = self.promote(req, graph, E, Z, R, sig, n, conns, cover, st)
                if c:
                    out.append(c)
        st.toc(h)
        return out

    def promote(self, req, graph, E, Z, R, sig, n, conns, cover, st):
        s, hg = self.s, self.s.w.hg
        ids = sorted(R)
        cid = "C:" + sig
        text = s.roles.verbalize("; ".join(hg.frag(x).text for x in ids), st)
        cmp, edges = lift(hg, s.w.onto, ids, conns, cid, text)
        s.w.add_with_edges(cmp, edges)
        self.cid_of[sig] = cid
        self.sig_of[cid] = sig
        self.members[cid] = tuple(ids)
        keep = [i for i, g in enumerate(graph.goals) if cover.get(g.gid) and set(cover[g.gid]) <= set(ids)]
        if keep:
            ix = {graph.goals[i].gid: k for k, i in enumerate(keep)}
            goals = [Goal(ix[graph.goals[i].gid], graph.goals[i].key, graph.goals[i].text, graph.goals[i].omega, {k: "" for k in graph.goals[i].slots}) for i in keep]
            eds = [(tuple(ix[a] for a in tl), tuple(ix[b] for b in hd), p) for tl, hd, p in graph.edges if all(x in ix for x in tl + hd)]
            r = Rec(cid="R" + cid, kind="cmp", goals=goals, edges=eds, e=E[keep].astype(np.float32), z=Z[keep].astype(np.float32),
                    phi={k: (cid,) for k in range(len(keep))}, stamps={cid: cmp.rev}, esig={}, conns=[], nu=cmp.nu, owner="",
                    users=set(), scope="shared", n=n, t0=req.t, t_last=req.t, cmp_fid=cid, iso=req.ctx.jur)
            s.mem.add(r)
            self.cmp_rec[cid] = r.cid
        self.promoted += 1
        return cid
