"""Algorithm 3: cascading invalidation with eager and lazy re-validation of composites."""
from collections import defaultdict, deque
from dataclasses import replace

import numpy as np

from . import build
from .chk import Checks
from .delta import Composer
from .mem import lift
from .model import Ctx, Goal, GoalGraph
from .util import Stats

EVAL_KEYS = ("chi_fun", "chi_dat", "chi_rol", "chi_tmp", "guard", "c4", "sound", "subst", "mark", "der", "rebind", "V1", "V2", "V3", "V4", "V5")


def evals(cnt):
    return int(sum(cnt.get(k, 0) for k in EVAL_KEYS))


class Invalidator:
    def __init__(self, sysm):
        self.s = sysm
        self.level = defaultdict(int)
        self.stats = {"events": 0, "suspect": 0, "eager": 0, "repaired": 0, "evicted": 0, "rebound": 0}

    # ------------------------------------------------------------------ event entry
    def handle(self, ev):
        s = self.s
        cnt = defaultdict(int)
        st = Stats()
        out = s.w.apply(ev, cnt)
        s.w.t = max(s.w.t, ev["t"])
        changed = list(dict.fromkeys(out["changed"] + out["expired"]))
        info = {"changed": len(changed), "suspect": 0, "repaired": 0, "evicted": 0, "rebound": 0, "eager": 0}
        self.stats["events"] += 1
        if not s.invalidate:
            return self._done(cnt, st, info, out)
        order = sorted(changed)
        hot = []
        for fid in order:
            if out["subst"].get(fid, False) and fid not in out["gone"]:
                self.rebind(fid, cnt, info)
            else:
                hot.append(fid)
        for fid in hot:
            self.cascade(fid, ev["t"], cnt, st, info)
        return self._done(cnt, st, info, out)

    def _done(self, cnt, st, info, out):
        info["evals"] = evals(cnt)
        info["llm_calls"] = st.total_calls()
        info["llm_ms"] = st.total_llm_ms()
        info["cnt"] = dict(cnt)
        info["added"] = len(out["added"])
        return info

    # ------------------------------------------------------------------ substitutable update
    def rebind(self, fid, cnt, info):
        s = self.s
        cur = s.w.hg.frag(fid)
        for r in s.mem.holding(fid):
            if fid in r.stamps:
                r.stamps[fid] = cur.rev
                cnt["rebind"] += 1
                r.nu = (max(r.nu[0], cur.nu[0]), min(r.nu[1], cur.nu[1])) if r.kind == "cmp" else r.nu
        for dep in self.dependents(fid, cnt):
            f = s.w.hg.frag(dep)
            if f is None:
                continue
            mem = [s.w.hg.frag(x) for x in f.supp if s.w.hg.frag(x) is not None]
            lo = max(x.nu[0] for x in mem)
            hi = min(x.nu[1] for x in mem)
            if (lo, hi) != tuple(f.nu):
                s.w.hg.revise(replace(f, nu=(lo, hi), rev=f.rev + 1))
            cnt["rebind"] += 1
            r = s.mem.recs.get(s.promo.cmp_rec.get(dep, ""))
            if r is not None:
                r.stamps[dep] = s.w.hg.frag(dep).rev
                r.nu = (lo, hi)
        info["rebound"] += 1
        self.stats["rebound"] += 1

    # ------------------------------------------------------------------ DER traversal
    def dependents(self, fid, cnt):
        """Composites that depend on fid through derivation hyperedges, transitively (backward BFS)."""
        hg = self.s.w.hg
        seen, q = set(), deque([fid])
        while q:
            x = q.popleft()
            for e in hg.edges_tail(x):
                cnt["der"] += 1
                if e.pat != "der":
                    continue
                for y in e.head:
                    if y not in seen:
                        seen.add(y)
                        q.append(y)
        return seen

    # ------------------------------------------------------------------ non-substitutable change
    def cascade(self, fid, t, cnt, st, info):
        s = self.s
        for r in s.mem.holding(fid):
            if not r.suspect:
                r.suspect = True
                cnt["mark"] += 1
        dep = self.dependents(fid, cnt)
        if not dep:
            return
        ut = {y: s.promo.utility(y) for y in dep}
        for y in dep:
            self.level[y] += 1
            s.w.suspect.add(y)
            info["suspect"] += 1
            self.stats["suspect"] += 1
        thr = self.u_eager()
        for y in sorted(dep, key=lambda z: -ut[z]):
            if y in s.w.suspect and ut[y] >= thr:
                info["eager"] += 1
                self.stats["eager"] += 1
                self.revalidate(y, t, cnt, st, info)

    def u_eager(self):
        """Eager threshold calibrated as a percentile of the utilities of the live composites."""
        s = self.s
        vals = [s.promo.utility(c) for c in s.promo.sig_of if s.w.hg.frag(c) is not None]
        if not vals:
            return 0.0
        return float(np.quantile(vals, s.cfg["memory"]["u_eager_pct"]))

    # ------------------------------------------------------------------ re-validation, repair, eviction
    def audit_ctx(self, cmp):
        iso = cmp.proc.split(".")[0]
        atoms = set(cmp.pre) | {"applicant_present", "jur:" + iso}
        return Ctx(frozenset(atoms), frozenset(build.DATA_ITEMS), cmp.roles | frozenset({"agency", "citizen"}), 10 ** 6, iso)

    def revalidate(self, fid, t, cnt, st=None, info=None):
        s = self.s
        st = st or Stats()
        hg = s.w.hg
        cmp = hg.frag(fid)
        if cmp is None:
            s.w.suspect.discard(fid)
            return False
        chk = Checks(s.w, cnt)
        actx = self.audit_ctx(cmp)
        ent = actx.jur + ".ent"
        mem = list(cmp.supp)
        alive = all(hg.frag(m) is not None and hg.alive(m, t) for m in mem)
        ok = False
        if alive:
            fs = frozenset(mem) | {ent}
            conns, bad = chk.derive_conns(fs, actx)
            fails = chk.check_conns(fs, conns, actx, t)
            snd, _ = chk.soundness(fs, conns)
            ok = (not bad) and (not fails) and snd and not chk.c4_conform(fs, conns)
        if ok:
            self._refresh(cmp, mem, cnt)
            s.w.suspect.discard(fid)
            return True
        new = self.repair(cmp, actx, t, cnt, st)
        if new:
            s.w.suspect.discard(fid)
            if info is not None:
                info["repaired"] += 1
            self.stats["repaired"] += 1
            return True
        self.evict(fid, cnt)
        s.w.suspect.discard(fid)
        if info is not None:
            info["evicted"] += 1
        self.stats["evicted"] += 1
        return False

    def _refresh(self, cmp, mem, cnt):
        """The composite stays valid: bump its revision so records stamped on it rebind."""
        s = self.s
        hg = s.w.hg
        lo = max(hg.frag(x).nu[0] for x in mem)
        hi = min(hg.frag(x).nu[1] for x in mem)
        new = replace(cmp, nu=(lo, hi), rev=cmp.rev + 1)
        hg.revise(new)
        r = s.mem.recs.get(s.promo.cmp_rec.get(cmp.fid, ""))
        if r is not None:
            r.stamps[cmp.fid] = new.rev
            r.nu = (lo, hi)

    def repair(self, cmp, actx, t, cnt, st):
        """Targeted repair by Algorithm 2: recompose the composite's end state from the current repository."""
        s = self.s
        hg = s.w.hg
        om = frozenset(cmp.post)
        goal = Goal(0, (cmp.proc, "cmp", ""), cmp.text, om, {}, "cmp", cmp.proc)
        graph = GoalGraph([goal], [])
        e = s.models.senc.encode(cmp.text).astype(np.float32)
        gv = (e, e)
        comp = Composer(s.w, s.roles, s.cfg, st, cnt)
        out = comp.run(actx, graph, gv, {}, [0], t, incremental=False, use_ret_call=False)
        if not out.get("ok"):
            return None
        mem = sorted(x for x in out["fset"] if hg.frag(x) is not None and hg.frag(x).ftype != "ENT")
        if len(mem) < 2:
            return None
        newc, edges = lift(hg, s.w.onto, mem, out["conns"], cmp.fid, cmp.text, rev=cmp.rev + 1)
        for k in [x for x in list(hg.edge_add) if x.startswith(cmp.fid + ".")]:
            hg.del_edge(k)
        hg.revise(newc)
        for ed in edges:
            hg.add_edge(ed)
        s.promo.members[cmp.fid] = tuple(mem)
        r = s.mem.recs.get(s.promo.cmp_rec.get(cmp.fid, ""))
        if r is not None:
            r.stamps[cmp.fid] = newc.rev
            r.nu = newc.nu
        return newc

    def evict(self, fid, cnt):
        """Remove a composite with its derivation and lifted hyperedges and index entries; dependents cascade."""
        s = self.s
        hg = s.w.hg
        for k in [x for x in list(hg.edge_add) if x.startswith(fid + ".")]:
            hg.del_edge(k)
        hg.withdraw(fid)
        s.w.fidx.extra.drop(fid)
        rid = s.promo.cmp_rec.pop(fid, None)
        if rid is not None:
            s.mem.evict(rid)
        for r in s.mem.holding(fid):
            r.evicted = True
        for dep in self.dependents(fid, cnt):
            if hg.frag(dep) is not None:
                self.evict(dep, cnt)
