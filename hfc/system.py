"""Shared pipeline pieces and the HyFragComp method with its ablations (whole, nore)."""
import json
import time
from collections import defaultdict

import numpy as np

from . import explain
from .chk import Checks
from .data import main_data
from .delta import Composer
from .inval import Invalidator, evals
from .llm import Roles
from .mem import Memory, make_rec, rec_from_json, expand_atomic
from .model import Ctx
from .promo import Promoter
from .repo import World
from .req import Result
from .reuse import Reuser
from .util import Stats


class Core:
    """Request front end, scratch composition, explanation and result assembly used by every method."""
    name = "core"

    def __init__(self, repo, cfg, models, seed=0):
        self.repo, self.cfg, self.models, self.seed = repo, cfg, models, seed
        self.w = World(repo, cfg, models)
        self.roles = Roles(cfg, models.senc)
        self.eco = {e["iso3"]: e["name"] for e in main_data()["economies"]}
        self.served = 0
        self.freeze = False

    # ------------------------------------------------------------------ front end (c_cur, c_int, c_pln, encoding)
    def front(self, req, st):
        ctx = req.ctx
        h = st.tic("abstraction")
        a = self.roles.abstract(req.text, self.eco[ctx.jur], st)
        if self.roles.has_pii(a["template"]):
            st.toc(h)
            return None
        clar = self.roles.interpret(a["template"], ctx, st)
        st.toc(h)
        h = st.tic("decompose")
        graph = self.roles.plan(clar, ctx, self.eco[ctx.jur], st)
        st.toc(h)
        h = st.tic("encoding")
        if graph.goals:
            E, Z = self.models.goal_vectors(graph.goals, graph.edges)
        else:
            E = Z = None
        st.toc(h)
        return graph, (E, Z), a

    def scratch(self, req, graph, gv, st, cnt):
        """The shared scratch composer: reformulation, GraphRAG per sub-goal, selection, composition, verification, repair."""
        comp = Composer(self.w, self.roles, self.cfg, st, cnt)
        return comp.run(req.ctx, graph, gv, {}, [g.gid for g in graph.goals], req.t, incremental=False, use_ret_call=True)

    # ------------------------------------------------------------------ results
    def fail(self, req, st, why, graph=None, mode="fail"):
        res = Result(rid=req.rid, mode=mode, ok=False, graph=graph, n_goals=len(graph.goals) if graph else 0)
        res.info["failure"] = why
        res.stats = self.pack(st)
        return res

    def pack(self, st):
        pv = self.cfg["privacy"]
        if pv.get("pad_latency"):
            # measure (d) of Sec. 5.7: memory lookups take the same time whether they hit a shared or a private record
            mem = sum(st.t.get(k, 0.0) for k in ("recall", "revalidation", "memory"))
            target = float(pv.get("pad_ms", 5.0)) / 1000.0
            if mem < target:
                if pv.get("pad_sleep", False):
                    time.sleep(target - mem)
                st.t["padding"] += target - mem
        d = st.to_dict()
        d["total_ms"] = st.total_ms()
        d["compute_ms"] = 1000.0 * sum(st.t.values())
        d["llm_total_ms"] = st.total_llm_ms()
        d["calls_total"] = st.total_calls()
        d["tok_in_total"] = sum(st.tok_in.values())
        d["tok_out_total"] = sum(st.tok_out.values())
        return d

    def assemble(self, req, st, graph, out, mode, plan=None, fallback=False, plan_mode=None):
        hg = self.w.hg
        n = len(graph.goals)
        res = Result(rid=req.rid, mode=mode, graph=graph, n_goals=n)
        res.ok = bool(out.get("ok"))
        res.info["plan_mode"] = plan_mode or mode
        res.info["fallback"] = fallback
        res.repairs = out.get("repairs", 0)
        if not out.get("fset"):
            res.info["failure"] = out.get("reason", "composition")
            res.info["rejections"] = [(a, b) for a, b in out.get("rejections", [])][:40]
            res.stats = self.pack(st)
            return res
        res.frags = tuple(sorted(out["fset"]))
        res.info["composites"] = {x: sorted(hg.frag(x).supp) for x in res.frags if hg.frag(x) is not None and hg.frag(x).kind == "cmp"}
        res.conns = tuple(out["conns"])
        res.cover = {g: tuple(f) for g, f in out["cover"].items()}
        res.junctions = tuple(out.get("junctions", ()))
        res.status = {}
        for g in graph.goals:
            if plan is not None and g.gid in plan.status:
                res.status[g.gid] = plan.status[g.gid]
            else:
                res.status[g.gid] = "new"
        ret = len(plan.retained) if plan is not None and mode in ("full", "partial") else 0
        res.rho = ret / n if n else 0.0
        res.delta = n - ret
        res.align = dict(plan.align) if plan is not None else {}
        res.info["feas"] = out.get("feas", {})
        res.info["rejections"] = [(a, b) for a, b in out.get("rejections", [])][:40]
        res.info["delta_graph"] = dict(plan.delta) if plan is not None else {}
        res.info["scope_rej"] = plan.scope_rej if plan is not None else 0
        res.info["failed_checks"] = dict(plan.failed) if plan is not None else {}
        h = st.tic("explain")
        wit, extra = explain.witnesses(graph, plan, out, hg, req.t, out.get("rejections", []),
                                       hide_scope=bool(self.cfg["privacy"].get("hide_scope_rejections", False)))
        res.witnesses = wit
        res.explanation = self.roles.explain(wit, st)
        if extra:
            res.explanation += "\n" + "\n".join(extra)
        res.info["composition"] = explain.composition_json(graph, plan, out, hg, req.ctx, res.status, res.junctions, out.get("feas", {}))
        st.toc(h)
        res.stats = self.pack(st)
        return res

    # ------------------------------------------------------------------ interface
    def handle(self, ev):
        cnt = defaultdict(int)
        out = self.w.apply(ev, cnt)
        return {"evals": 0, "llm_calls": 0, "llm_ms": 0.0, "changed": len(out["changed"]), "suspect": 0, "repaired": 0, "evicted": 0}

    def preload(self, recs, t=0.0):
        pass


class HyFragComp(Core):
    """Full HyFragComp; `revalidate`, `delta`, `single` and `invalidate` switch the ablations (whole, nore)."""
    name = "ours"

    def __init__(self, repo, cfg, models, seed=0, revalidate=True, delta=True, single=False, invalidate=True, promote=True, name="ours"):
        super().__init__(repo, cfg, models, seed)
        self.name = name
        self.revalidate, self.delta, self.single, self.invalidate, self.promote_on = revalidate, delta, single, invalidate, promote
        self.mem = Memory(self.w, cfg)
        self.promo = Promoter(self)
        self.inv = Invalidator(self)
        self.nrec = 0
        self.cnt_cur, self.st_cur = defaultdict(int), Stats()
        if invalidate:
            self.w.lazy_cb = self._lazy_frag

    def _lazy_frag(self, fid, t):
        self.inv.revalidate(fid, t, self.cnt_cur, self.st_cur)

    # ------------------------------------------------------------------ warm start
    def preload(self, recs, t=0.0):
        cnt = defaultdict(int)
        chk = Checks(self.w, cnt)
        for d in recs:
            self.mem.add(rec_from_json(self.w, chk, self.models, d, t))

    # ------------------------------------------------------------------ events
    def handle(self, ev):
        return self.inv.handle(ev)

    def lazy(self, r):
        if r.cmp_fid in self.w.suspect:
            self.inv.revalidate(r.cmp_fid, self.w.t, self.cnt_cur, self.st_cur)

    # ------------------------------------------------------------------ request path
    def serve(self, req):
        st = Stats()
        cnt = st.pred
        self.cnt_cur, self.st_cur = cnt, st
        self.w.t = max(self.w.t, req.t)
        self.served += 1
        fr = self.front(req, st)
        if fr is None:
            return self.fail(req, st, "pii")
        graph, gv, _ = fr
        if not graph.goals:
            return self.fail(req, st, "no-goals", graph)
        ctx, t = req.ctx, req.t
        E, Z = gv
        chk = Checks(self.w, cnt)
        plan, mode, plan_mode = None, "scratch", "scratch"
        if len(self.mem.recs):
            reuser = Reuser(self.w, self.mem, chk, self.cfg, cnt, revalidate=self.revalidate, single_record=self.single)
            reuser.st = st
            reuser.on_suspect = self.lazy
            plan = reuser.plan(ctx, graph, E, Z, req.user, t)
            mode = plan_mode = plan.mode
            if mode == "partial" and not self.delta:
                mode = "scratch"
        fallback = False
        if mode in ("full", "partial"):
            comp = Composer(self.w, self.roles, self.cfg, st, cnt)
            prev = defaultdict(set)
            for cid in plan.recs:
                for u, v, eid in self.mem.recs[cid].conns:
                    prev[v].add((u, eid))
            out = comp.run(ctx, graph, gv, plan.retained, plan.gdelta, t, incremental=True, prev_in=prev)
            if not out.get("ok"):
                fallback = True
                out = self.scratch(req, graph, gv, st, cnt)
                mode_final = "scratch"
            else:
                mode_final = mode
        else:
            out = self.scratch(req, graph, gv, st, cnt)
            mode_final = "scratch"
        used = plan if mode_final in ("full", "partial") else None
        res = self.assemble(req, st, graph, out, mode_final, used, fallback, plan_mode)
        if plan is not None:
            res.info["plan_rho"] = plan.rho
        if plan is not None and used is None:
            res.info["blocked_witnesses"] = dict(plan.witnesses)
            res.align = dict(plan.align)
        if res.ok and not self.freeze:
            self.after(req, graph, E, Z, out, used, mode_final, st, cnt)
        res.stats = self.pack(st)
        return res

    # ------------------------------------------------------------------ memory update and promotion
    def after(self, req, graph, E, Z, out, plan, mode, st, cnt):
        hg = self.w.hg
        h = st.tic("memory")
        if plan is not None:
            for cid in plan.recs:
                if cid in self.mem.recs:
                    self.mem.touch(self.mem.recs[cid], req.user, req.t)
        atom = {g: expand_atomic(hg, out["cover"].get(g, ())) for g in out["cover"]}
        fatom = frozenset(set(x for v in atom.values() for x in v) | {req.ctx.jur + ".ent"})
        chk = Checks(self.w, cnt)
        if mode != "full":
            self.nrec += 1
            cid = "u%s.%d" % (req.rid, self.nrec)
            r = make_rec(self.w, chk, cid, graph.goals, graph.edges, E, Z, {g: atom[g] for g in atom}, req.ctx, req.user, "u", req.t)
            self.mem.add(r)
            self.trim(req.t)
        st.toc(h)
        if self.promote_on:
            conns, _ = chk.derive_conns(fatom, req.ctx)
            self.promo.observe(req, graph, E, Z, fatom, conns, atom, st, cnt)

    def trim(self, t):
        cap = self.cfg["memory"]["capacity"]
        live = [r for r in self.mem.recs.values() if r.kind == "rec" and not r.evicted]
        if len(live) > cap:
            live.sort(key=lambda r: r.t_last)
            for r in live[:max(1, len(live) // 10)]:
                self.mem.evict(r.cid)
