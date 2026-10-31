"""Context-aware fragment composition baseline (Bucchiarone et al.): domain builder, strong planner, repair by re-planning."""
from collections import defaultdict
from dataclasses import dataclass

from hfc.chk import Checks
from hfc.system import Core
from hfc.util import Stats

from .cafc_lts import (Act, Annot, CTRL, UNCTRL, Context, Domain, Formula, LTS, Prop, StateCap)

CAFC_KEYS = ("ctx_ev", "lts_trans", "ground", "plan_pre", "plan_fwd")


@dataclass
class Live:
    """A composition produced earlier that the adaptation manager keeps watching."""
    rid: str
    ctx: object
    graph: object
    blocks: dict          # procedure id -> set of fragment ids
    t: float


class Tmp:
    """Minimal request view used when an event forces a fallback composition."""

    def __init__(self, ctx, t, rid):
        self.ctx, self.t, self.rid = ctx, t, rid


def evals_of(cnt):
    return int(sum(cnt.get(k, 0) for k in CAFC_KEYS))


class CAFC(Core):
    """Plan every request from the current context; replan the affected block when an event breaks a precondition."""
    name = "cafc"

    def __init__(self, repo, cfg, models, seed=0):
        super().__init__(repo, cfg, models, seed)
        self.cc = cfg["cafc"]
        self.live = {}
        self.watch = defaultdict(set)
        by = defaultdict(list)
        for f in repo.base.frags.values():
            by[f.proc].append(f.fid)
        self.by_proc = by
        self.stats = {"events": 0, "replanned": 0, "failed": 0, "states": 0, "pruned": 0}

    # ------------------------------------------------------------------ domain builder
    def candidates(self, P, iso):
        ids = set(self.by_proc.get(P, ())) | set(self.by_proc.get(iso, ()))
        for fid, f in self.w.hg.cur.items():
            if f.proc in (P, iso):
                ids.add(fid)
        out = []
        for fid in sorted(ids):
            f = self.w.hg.frag(fid)
            if f is not None and f.kind == "atm":
                out.append(f)
        return out

    @staticmethod
    def tokens(f, ctx):
        """Effects of a fragment as context events: its post atoms and the data items it adds to the context."""
        return set(f.post) | {"data:" + o for o in f.out if o not in ctx.data}

    def relevant(self, frs, goals_om, ctx):
        """Backward relevance: keep the fragments that can produce an atom or data item the goal (transitively) needs."""
        onto = self.w.onto
        need = set(goals_om)
        keep = {}
        have = onto.close(ctx.atoms)
        grew = True
        while grew:
            grew = False
            for f in frs:
                if f.fid in keep:
                    continue
                tk = self.tokens(f, ctx)
                if onto.close(tk) & need:
                    keep[f.fid] = f
                    need |= {a for a in f.pre if a not in have}
                    need |= {"data:" + d for d in f.inp if d not in ctx.data}
                    grew = True
        return list(keep.values())

    def build_domain(self, ctx, goals_om, frs, t, cnt):
        """Context properties, annotations, fragment LTSs and the context-aware execution domain of one block."""
        onto = self.w.onto
        have = onto.close(ctx.atoms)
        produced = sorted({a for f in frs for a in self.tokens(f, ctx)})
        props, annots, ltss = {}, {}, []
        for a in produced:
            props[a] = Prop(a, ("F", "T"), "F", E={"set:" + a}, T=[("F", "set:" + a, "T")])

        def static(name, val, want="T"):
            if name not in props:
                props[name] = Prop(name, ("T", "F"), "T" if val else "F")
            return Formula.lit(name, want)

        def data_formula(d):
            if d in ctx.data:
                return static("data:" + d, True)
            if "data:" + d in props:
                return Formula.lit("data:" + d, "T")
            return static("data:" + d, False)

        def atom_formula(a):
            if a in props:
                return Formula.lit(a, "T")
            if a in have:
                return static(a, True)
            c = onto.conf.get(a)
            if c is not None and c in have:
                return static(a, False)
            prov = [p for p in produced if a in onto.close({p})]
            if prov:
                out = Formula.bottom()
                for p in prov:
                    out = out.or_(Formula.lit(p, "T"))
                return out
            return static(a, False)

        for f in frs:
            P = Formula.top()
            for a in sorted(f.pre):
                P = P.and_(atom_formula(a))
            for d in sorted(f.inp):
                P = P.and_(data_formula(d))
            for r in sorted(f.roles):
                P = P.and_(static("role:" + r, r in ctx.roles))
            if f.guard:
                P = P.and_(static("guard:" + f.guard, f.guard in ctx.atoms))
            else:
                for g in sorted({e.guard for e in self.w.hg.edges_head(f.fid) if e.pat == "xor" and e.guard}):
                    P = P.and_(static("guard:" + g, g in ctx.atoms, "F"))
            P = P.and_(static("valid:" + f.fid, f.nu[0] <= t and t + f.dmin <= f.nu[1]))
            s, r = "start:" + f.fid, "reply:" + f.fid
            ev = frozenset("set:" + a for a in self.tokens(f, ctx))
            annots[s] = Annot(P, frozenset(), None)
            annots[r] = Annot(Formula.top(), ev, None)
            acts = [Act(s, CTRL, f.fid, float(f.dmin)), Act(r, UNCTRL, f.fid, 0.0)]
            ltss.append(LTS(f.fid, ("idle", "run", "done"), "idle", acts, [("idle", s, "run"), ("run", r, "done")]))
        G = Formula.top()
        for a in sorted(goals_om):
            G = G.and_(atom_formula(a))
        ctxm = Context(list(props.values()))
        dom = Domain(ltss, annots, ctxm, cnt, self.cc["max_states"])
        return dom, G

    # ------------------------------------------------------------------ strong planner (backward fixpoint)
    def strong_plan(self, dom, G, cnt):
        cov = dom.goal_states(G)
        goal = set(cov)
        table = {}
        new = set(cov)
        while dom.init not in cov and new:
            cand = set()
            for s in new:
                cand |= dom.pred.get(s, set())
            cand -= cov
            add = {}
            for s in cand:
                cnt["plan_pre"] += 1
                ctrl = [(a, n) for a, n in dom.R[s] if dom.kind[a] == CTRL]
                unc = [(a, n) for a, n in dom.R[s] if dom.kind[a] == UNCTRL]
                if unc and not all(n in cov for _, n in unc):
                    continue
                good = [(a, n) for a, n in ctrl if n in cov]
                if good or unc:
                    add[s] = {"ctrl": good, "wait": [n for _, n in unc]}
            if not add:
                break
            table.update(add)
            cov |= set(add)
            new = set(add)
        if dom.init not in cov:
            return None
        return table, goal

    def policy(self, dom, table, goal):
        """Cheapest solution inside the strong table, then the fragments the forward run starts."""
        memo = {}

        def val(s):
            if s in goal:
                return 0.0, None
            if s in memo:
                return memo[s]
            best = (float("inf"), None)
            ent = table[s]
            for a, n in sorted(ent["ctrl"]):
                v = dom.cost[a] + val(n)[0]
                if v < best[0]:
                    best = (v, ("ctrl", a, n))
            if ent["wait"]:
                v = max(val(n)[0] for n in ent["wait"])
                if v < best[0] and not ent["ctrl"]:
                    best = (v, ("wait", None, ent["wait"][0]))
            memo[s] = best
            return best
        s, started, cnt = dom.init, [], 0
        while s not in goal and cnt < 10000:
            v, step = val(s)
            if step is None:
                return None
            kind, a, n = step
            if kind == "ctrl":
                started.append(dom.frag[a])
            s = n
            cnt += 1
        return started

    # ------------------------------------------------------------------ planning of one block
    def plan_block(self, ctx, P, goals_om, t, cnt):
        iso = ctx.jur
        frs = self.relevant(self.candidates(P, iso), goals_om, ctx)
        if not frs:
            return None, "no-fragments"
        try:
            dom, G = self.build_domain(ctx, goals_om, frs, t, cnt)
        except StateCap:
            return None, "state-cap"
        self.stats["states"] += dom.size()
        self.stats["pruned"] += dom.pruned
        sol = self.strong_plan(dom, G, cnt)
        if sol is None:
            return None, "no-strong-plan"
        started = self.policy(dom, *sol)
        cnt["plan_fwd"] += len(started or ())
        if started is None:
            return None, "no-policy"
        return set(started), "ok"

    def plan_request(self, ctx, graph, t, cnt, only=None):
        """Plan each procedure block of the goal graph independently; the shared entry fragment joins every block."""
        blocks = defaultdict(set)
        for g in graph.goals:
            blocks[g.key[0]] |= set(g.omega)
        res, why = {}, "ok"
        for P, om in blocks.items():
            if only is not None and P not in only:
                continue
            fs, w = self.plan_block(ctx, P, om, t, cnt)
            if fs is None:
                return None, w, P
            res[P] = fs
        return res, "ok", None

    def make_out(self, ctx, graph, plans, t, cnt):
        hg = self.w.hg
        fset = {ctx.jur + ".ent"}
        for fs in plans.values():
            fset |= fs
        onto = self.w.onto
        cover = {}
        for g in graph.goals:
            mine = plans.get(g.key[0], set())
            cov = []
            for a in sorted(g.omega):
                for fid in sorted(mine):
                    f = hg.frag(fid)
                    if f is not None and a in onto.close(f.post):
                        cov.append(fid)
                        break
            cover[g.gid] = tuple(dict.fromkeys(cov))
        conns, bad = Checks(self.w, cnt).derive_conns(frozenset(fset), ctx)
        return {"ok": all(cover.values()), "fset": frozenset(fset), "conns": conns, "cover": cover, "repairs": 0,
                "junctions": [], "feas": {}, "rejections": [], "unlicensed": bad}

    # ------------------------------------------------------------------ request path
    def serve(self, req):
        st = Stats()
        cnt = st.pred
        self.w.t = max(self.w.t, req.t)
        self.served += 1
        fr = self.front(req, st)
        if fr is None:
            return self.fail(req, st, "pii")
        graph, gv, _ = fr
        if not graph.goals:
            return self.fail(req, st, "no-goals", graph)
        h = st.tic("planning")
        plans, why, bad = self.plan_request(req.ctx, graph, req.t, cnt)
        st.toc(h)
        info = {"reason": why, "block": bad}
        if plans is None:
            out = self.scratch(req, graph, gv, st, cnt)
            res = self.assemble(req, st, graph, out, "scratch", None, True, "scratch")
        else:
            h = st.tic("translate")
            out = self.make_out(req.ctx, graph, plans, req.t, cnt)
            st.toc(h)
            if not out["ok"]:
                out = self.scratch(req, graph, gv, st, cnt)
                res = self.assemble(req, st, graph, out, "scratch", None, True, "scratch")
                info["reason"] = "uncovered"
            else:
                res = self.assemble(req, st, graph, out, "scratch", None, False, "scratch")
                self.keep(req, graph, plans)
        res.info["cafc"] = info
        res.stats = self.pack(st)
        return res

    # ------------------------------------------------------------------ adaptation manager
    def keep(self, req, graph, plans):
        old = self.live.pop(req.rid, None)
        if old:
            self._unwatch(old)
        lv = Live(req.rid, req.ctx, graph, plans, req.t)
        self.live[req.rid] = lv
        for P, fs in plans.items():
            for f in fs:
                self.watch[f].add(req.rid)
            self.watch[P].add(req.rid)
        cap = self.cc["active"]
        if len(self.live) > cap:
            for rid in list(self.live)[: len(self.live) - cap]:
                self._unwatch(self.live.pop(rid))

    def _unwatch(self, lv):
        for P, fs in lv.blocks.items():
            for f in list(fs) + [P]:
                self.watch[f].discard(lv.rid)

    def handle(self, ev):
        cnt = defaultdict(int)
        st = Stats()
        out = self.w.apply(ev, cnt)
        self.w.t = max(self.w.t, ev["t"])
        changed = list(dict.fromkeys(out["changed"] + out["expired"] + out["added"]))
        self.stats["events"] += 1
        hit = set()
        for fid in changed:
            hit |= self.watch.get(fid, set())
            f = self.w.hg.frag(fid)
            if f is not None:
                hit |= self.watch.get(f.proc, set())
        fixed = failed = 0
        for rid in sorted(hit):
            lv = self.live.get(rid)
            if lv is None:
                continue
            aff = {P for P, fs in lv.blocks.items() if fs & set(changed) or P in {self.w.hg.frag(x).proc for x in changed if self.w.hg.frag(x) is not None}}
            plans, why, bad = self.plan_request(lv.ctx, lv.graph, ev["t"], cnt, only=aff)
            if plans is None:
                failed += 1
                self.stats["failed"] += 1
                out_s = self.scratch(Tmp(lv.ctx, ev["t"], rid), lv.graph, self.models.goal_vectors(lv.graph.goals, lv.graph.edges), st, cnt)
                self._unwatch(lv)
                self.live.pop(rid, None)
                continue
            self._unwatch(lv)
            lv.blocks.update(plans)
            lv.t = ev["t"]
            for P, fs in lv.blocks.items():
                for f in fs:
                    self.watch[f].add(rid)
                self.watch[P].add(rid)
            fixed += 1
            self.stats["replanned"] += 1
        return {"evals": evals_of(cnt), "llm_calls": st.total_calls(), "llm_ms": st.total_llm_ms(), "changed": len(changed), "suspect": len(hit), "repaired": fixed,
                "evicted": failed, "rebound": 0, "cnt": dict(cnt)}
