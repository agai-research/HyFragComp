"""Case-based composition baseline (Lu et al.): knowledge-graph cases, fused semantic and structural similarity, whole-case reuse."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from hfc.chk import Checks
from hfc.mem import expand_atomic, make_rec, rec_from_json
from hfc.reuse import Plan, pseudo
from hfc.system import Core
from hfc.util import Stats

from .cbr_kg import CaseKG, Desc, concept_of
from .cbr_mcs import mcs
from .cbr_sim import SimHash, TfIdf, fusion_weights, tokens


@dataclass
class Case:
    """A stored composition record with its text features, fingerprint and normalised graph."""
    cid: str
    rec: object
    toks: list
    emb: np.ndarray
    fp: np.ndarray
    lg: object
    keymap: dict          # goal key -> position in rec.goals
    pos_gid: list         # position -> gid of the stored goal
    n: int = 1
    t_last: float = 0.0


class CBR(Core):
    """Retrieve the most similar case by fused similarity and recommend it whole; fall back to scratch below the threshold."""
    name = "cbr"

    def __init__(self, repo, cfg, models, seed=0):
        super().__init__(repo, cfg, models, seed)
        self.cc = cfg["cbr"]
        self.fw = fusion_weights(self.cc)
        self.kg = CaseKG(self.cc)
        self.sh = SimHash(self.cc["simhash_bits"], self.cc["simhash_shingle"])
        self.cases = {}
        self.dirty = True
        self.pk = None
        self.nrec = 0

    # ------------------------------------------------------------------ text features
    def _tok(self, text):
        return tokens(text, self.cc["min_token_len"], self.cc["drop_stop_words"])

    def _doc(self, goals, eco):
        """Text of one case or query: sub-goal sentences, economy name and the concept keywords of the goal keys."""
        toks = self._tok(" ".join(t for _, t, _ in goals)) + self._tok(eco)
        for (P, step, qual), _, _ in goals:
            toks += self._tok(concept_of(P)[1]) + [step] + ([qual] if qual else [])
        return toks

    def _emb(self, goals):
        """Deep-semantic vector: mean-pooled sentence embeddings of the sub-goal texts, unit length."""
        v = self.models.senc.encode([t for _, t, _ in goals]).mean(axis=0)
        return (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)

    # ------------------------------------------------------------------ case construction
    def _desc(self, cid, goals, edges, iso, atoms, phi=None, weight=1, owner=""):
        """Knowledge-graph description; goal gids are mapped to positions."""
        pos = {g.gid: i for i, g in enumerate(goals)}
        gl = [(tuple(g.key), g.text, tuple(sorted(g.slots))) for g in goals]
        ed = [(tuple(pos[a] for a in tl), tuple(pos[b] for b in hd), p) for tl, hd, p in edges]
        frags = [[self.w.hg.frag(f) for f in phi.get(g.gid, ())] for g in goals] if phi is not None else []
        return Desc(cid, iso, self.eco[iso], gl, ed, frozenset(atoms), frags, weight, owner)

    def _add(self, rec, t=0.0):
        d = self._desc(rec.cid, rec.goals, rec.edges, rec.iso, rec.ctx_atoms, rec.phi, rec.n, rec.owner)
        self.kg.add(d)
        toks = self._doc(d.goals, d.eco)
        c = Case(rec.cid, rec, toks, self._emb(d.goals), self.sh.fingerprint(toks), self.kg.structure(rec.cid),
                 {tuple(g.key): i for i, g in enumerate(rec.goals)}, [g.gid for g in rec.goals], rec.n, max(t, rec.t_last))
        self.cases[rec.cid] = c
        self.dirty = True
        return c

    def _evict(self, cid):
        self.cases.pop(cid, None)
        self.kg.drop(cid)
        self.dirty = True

    def _pack(self):
        if not self.dirty and self.pk is not None:
            return self.pk
        ids = list(self.cases)
        cs = [self.cases[i] for i in ids]
        self.pk = {"ids": ids,
                   "emb": np.stack([c.emb for c in cs]) if cs else np.zeros((0, 1), np.float32),
                   "fp": np.stack([c.fp for c in cs]) if cs else np.zeros((0, self.sh.bits), np.uint8),
                   "tf": TfIdf([c.toks for c in cs])}
        self.dirty = False
        return self.pk

    # ------------------------------------------------------------------ interface
    def preload(self, recs, t=0.0):
        chk = Checks(self.w, defaultdict(int))
        for d in recs:
            self._add(rec_from_json(self.w, chk, self.models, d, t), t)

    def handle(self, ev):
        return super().handle(ev)

    # ------------------------------------------------------------------ query processing and retrieval
    def query(self, req, graph):
        """Dual path: a triple-form query graph imported into the case graph, and keywords with text features."""
        d = self._desc("query", graph.goals, graph.edges, req.ctx.jur, req.ctx.atoms)
        self.kg.add(d, link=False)
        out = {"lg": self.kg.structure("query"), "triples": self.kg.triples("query"), "desc": d}
        self.kg.drop("query")
        out["toks"] = self._doc(d.goals, d.eco)
        out["emb"] = self._emb(d.goals)
        out["fp"] = self.sh.fingerprint(out["toks"])
        return out

    def retrieve(self, q, cnt):
        """Fused similarity of the query to the cases; MCS only for the best semantic candidates."""
        pk = self._pack()
        beta, alpha = self.fw["beta"], self.fw["alpha"]
        s_deb = np.clip(pk["emb"] @ q["emb"], 0.0, 1.0)
        s_sh = self.sh.sims(pk["fp"], q["fp"])
        s_tf = pk["tf"].sims(q["toks"])
        attr = beta["deberta"] * s_deb + beta["simhash"] * s_sh + beta["tfidf"] * s_tf
        top = np.argsort(-attr, kind="stable")[: self.cc["top_semantic"]]
        rows = []
        for i in top:
            c = self.cases[pk["ids"][i]]
            m = mcs(q["lg"], c.lg, self.cc["mcs_budget"])
            cnt["mcs_exp"] += m.expansions
            fused = alpha * float(attr[i]) + (1.0 - alpha) * m.sim
            rows.append({"cid": c.cid, "sim": fused, "attr": float(attr[i]), "stru": m.sim, "mcs_exact": m.exact,
                         "deberta": float(s_deb[i]), "simhash": float(s_sh[i]), "tfidf": float(s_tf[i])})
        cnt["cbr_cand"] += len(rows)
        rows.sort(key=lambda r: (-r["sim"], -r["attr"], r["cid"]))
        return rows

    # ------------------------------------------------------------------ whole-case recommendation
    def recommend(self, req, graph, hit):
        """Rebind the slots of the recommended case to the request; returns (out, plan) or None when the case lacks a goal."""
        c = self.cases[hit["cid"]]
        rec = c.rec
        align, cover, gp = {}, {}, {}
        for g in graph.goals:
            j = c.keymap.get(tuple(g.key))
            if j is None:
                return None
            gid = c.pos_gid[j]
            align[g.gid] = (c.cid, j, (hit["attr"], hit["stru"], 0.0))
            cover[g.gid] = tuple(rec.phi.get(gid, ()))
            gp[g.gid] = j
        if any(not v for v in cover.values()):
            return None
        fset = frozenset({f for v in rec.phi.values() for f in v} | {req.ctx.jur + ".ent"})
        out = {"ok": True, "fset": fset, "conns": list(rec.conns), "cover": cover, "repairs": 0, "junctions": [], "feas": {},
               "rejections": []}
        mapped = set(gp.values())
        stored = {(tuple(sorted(a)), tuple(sorted(b)), p) for a, b, p in c.rec.edges}
        pos_of = {gid: i for i, gid in enumerate(c.pos_gid)}
        eq = 0
        for tl, hd, pat in graph.edges:
            key = (tuple(sorted(c.pos_gid[gp[x]] for x in tl)), tuple(sorted(c.pos_gid[gp[x]] for x in hd)), pat)
            eq += key in stored
        plan = Plan(mode="full", anchor=c.cid, align=align, retained=dict(cover), status={g: "reuse" for g in cover}, gdelta=[],
                    rho=1.0, recs={c.cid})
        plan.delta = {"G=": sorted(gp), "G+": [], "G-": sorted(set(range(len(c.pos_gid))) - mapped),
                      "E+": len(graph.edges) - eq, "E-": len(stored) - eq}
        for g in graph.goals:
            plan.witnesses[g.gid] = {"record": pseudo(c.cid), "m": [round(hit["attr"], 3), round(hit["stru"], 3), 0.0], "tier": "cbr",
                                     "t": req.t, "nu": rec.nu, "status": "reuse", "stamps": {}}
        return out, plan

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
        E, Z = gv
        info = {"threshold": self.cc["sim_threshold"], "cases": len(self.cases), "reason": "empty"}
        used, out, plan, pal = False, None, None, {}
        if self.cases:
            h = st.tic("recall")
            q = self.query(req, graph)
            rows = self.retrieve(q, cnt)
            st.toc(h)
            h = st.tic("alignment")
            best = rows[0]
            info.update({"best": best["cid"], "sim": best["sim"], "parts": {k: best[k] for k in ("deberta", "simhash", "tfidf", "stru")},
                         "top": [(r["cid"], round(r["sim"], 4)) for r in rows[:5]], "mcs_exact": best["mcs_exact"],
                         "query_triples": len(q["triples"])})
            top = self.cases[best["cid"]]
            for g in graph.goals:
                j = top.keymap.get(tuple(g.key))
                if j is not None:
                    pal[g.gid] = (top.cid, j, (best["attr"], best["stru"], 0.0))
            if best["sim"] < self.cc["sim_threshold"]:
                info["reason"] = "below_threshold"
            else:
                rc = self.recommend(req, graph, best)
                if rc is None:
                    info["reason"] = "uncovered"
                else:
                    out, plan = rc
                    used = True
                    info["reason"] = "reuse"
            st.toc(h)
        if used:
            res = self.assemble(req, st, graph, out, "full", plan, False, "full")
            if self.cc["rebind_slots"]:
                res.info["rebind"] = {g.gid: dict(g.slots) for g in graph.goals if g.slots}
            self.touch(plan.anchor, req.t)
        else:
            out = self.scratch(req, graph, gv, st, cnt)
            res = self.assemble(req, st, graph, out, "scratch", None, bool(self.cases), "scratch")
            res.align = pal
            if res.ok and not self.freeze:
                self.remember(req, graph, E, Z, out, st, cnt)
        res.info["cbr"] = info
        res.stats = self.pack(st)
        return res

    # ------------------------------------------------------------------ memory update
    def touch(self, cid, t):
        c = self.cases.get(cid)
        if c is not None:
            c.n += 1
            c.t_last = t
            self.kg.g.node("case|" + cid)["weight"] = c.n

    def remember(self, req, graph, E, Z, out, st, cnt):
        """A newly composed solution becomes a case."""
        h = st.tic("memory")
        hg = self.w.hg
        atom = {g: expand_atomic(hg, out["cover"].get(g, ())) for g in out["cover"]}
        self.nrec += 1
        cid = "k%s.%d" % (req.rid, self.nrec)
        rec = make_rec(self.w, Checks(self.w, cnt), cid, graph.goals, graph.edges, E, Z, atom, req.ctx, req.user, "u", req.t)
        self._add(rec, req.t)
        cap = self.cfg["memory"]["capacity"]
        if len(self.cases) > cap:
            old = sorted(self.cases.values(), key=lambda c: c.t_last)
            for c in old[:max(1, len(old) // 10)]:
                self._evict(c.cid)
        st.toc(h)
