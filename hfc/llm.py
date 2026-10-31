"""One underlying language model specialised by role configurations.

Backends (llm.backend in configs/default.yaml):
  rule    deterministic rule-and-embedding engine; latency is computed from token counts with llm.latency
  openai  every role is answered by the hosted model (GPT-5.6 Luna by default); latency and tokens are measured
  shadow  the rule engine decides, and every call is also sent to the hosted model; both answers, the measured
          latency and the tokens are logged to results/logs/shadow.jsonl (calibration, Exp/calibrate.py)
In all backends, the number of calls is the number of role invocations decided by the pipeline.
"""
import json
import re

import numpy as np

from . import build, lex
from .model import Goal, GoalGraph
from .util import ntok

ROLES = ("cur", "int", "pln", "ret", "sel", "com", "ver", "exp")
SYSTEM = {
    "cur": "You are the curator. Mask every personal literal of the request and return an anonymised template.",
    "int": "You are the interpreter. Resolve references and return the clarified prompt with its context atoms.",
    "pln": "You are the planner. Decompose the clarified prompt into typed sub-goals and their ordering.",
    "ret": "You are the retriever. Reformulate each sub-goal into a retrieval query over the fragment repository.",
    "sel": "You are the selector. Choose the fragment that best realises the sub-goal among the compatible candidates.",
    "com": "You are the composer. Assemble the selected fragments through the composition rules of the repository.",
    "ver": "You are the verifier. Explain the defects of the composition and propose a bridging repair.",
    "exp": "You are the explainer. State what was reused, switched or newly composed, and under which validity.",
}
DEC = {r: {"temperature": 0.0} for r in ROLES}
FORMAT = {
    "cur": ' Return JSON {"template": string}, replacing names by <NAME>, e-mail addresses by <EMAIL>, phone numbers by <PHONE>, dates by <DATE>, identity numbers by <ID>, amounts by <AMOUNT> and cities of residence by <CITY>.',
    "int": ' Return JSON {"clarified": string} without greetings or personal remarks.',
    "pln": ' Return JSON {"goals": [{"procedure": slug, "step": one of "elig","docs","extra","appt","sub", "qualifier": "first", "renew" or "", "text": the clause, "channel": string or null}]}, using only the procedure slugs listed in the input.',
    "ret": ' Return JSON {"queries": [string, ...]} with one query per sub-goal, in the same order.',
    "sel": ' Return JSON {"choice": the id of the chosen candidate}.',
    "com": ' Return JSON {"plan": object} describing the assembly.',
    "ver": ' Return JSON {"repair": object} describing the repair.',
    "exp": ' Return JSON {"explanation": string} restating each witness in one line, without adding facts.',
}

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ()-]{7,}\d")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
_IDNO = re.compile(r"\b[A-Z]{1,3}\d{5,9}\b")
_AMOUNT = re.compile(r"\b\d[\d,.]*\s?(TND|EUR|USD|euros|dinars|dollars)\b")
_NAME_INTRO = re.compile(r"\b((?i:my name is|i am|i'm|this is|name:))\s+((?:[A-Z][a-z]+)(?:\s+(?:Ben|Bin|El|Al)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)*)")
_CITY_INTRO = re.compile(r"\b(I live in|I am based in|I currently live in|I am from|resident of)\s+([A-Z][a-z]+)")


class Roles:
    """Role configurations of the single language model; calls are counted and priced analytically."""

    def __init__(self, cfg, senc):
        self.cfg = cfg
        self.senc = senc
        self.lat = cfg["llm"]["latency"]
        self.theta = cfg["planner"]["theta_plan"]
        self.w_lex = cfg["planner"]["w_lex"]
        self.p_sem = cfg["planner"]["sem_only"]
        self.backend = cfg["llm"].get("backend", "rule")
        if self.backend not in ("rule", "openai", "shadow"):
            raise ValueError("llm.backend must be rule, openai or shadow")
        self.api = None
        if self.backend != "rule":
            from .llm_api import ApiClient
            self.api = ApiClient(cfg)
        self.fallbacks = {r: 0 for r in ROLES}
        doms = sorted(((v[0], k) for k, v in lex.DOMAINS.items()), key=lambda x: -len(x[0]))
        self.dom_rx = re.compile("|".join(re.escape(d) for d, _ in doms), re.I)
        self.dom_slug = {d.lower(): k for d, k in doms}
        self.eco_names = {e["name"]: e["iso3"] for e in __import__("hfc.data", fromlist=["x"]).main_data()["economies"]}
        self._ex = {}

    # ------------------------------------------------------------------ accounting
    def charge(self, st, role, ctx_text, out_obj):
        nin = ntok(SYSTEM[role]) + ntok(ctx_text)
        nout = ntok(out_obj if isinstance(out_obj, str) else json.dumps(out_obj, default=list))
        ms = self.lat["fixed_ms"] + 1000.0 * nin / self.lat["prefill_tok_s"] + 1000.0 * nout / self.lat["decode_tok_s"]
        st.llm(role, nin, nout, ms)

    def emit(self, st, role, ctx_text, rule_out, payload=None, parse=None, norm=None, acct=None):
        """Account for one role invocation and return its answer according to the backend."""
        acct = rule_out if acct is None else acct
        if self.backend == "rule":
            self.charge(st, role, ctx_text, acct)
            return rule_out
        user = json.dumps(payload if payload is not None else {"input": ctx_text}, default=list)
        mock = norm(rule_out) if norm else rule_out
        obj, text, nin, nout, ms, cached = self.api.complete(role, SYSTEM[role] + FORMAT[role], user,
                                                             self.cfg["llm"]["decoding"].get("c_" + role, DEC[role]), mock_answer=mock)
        if self.backend == "shadow":
            self.charge(st, role, ctx_text, acct)
            nin_e = ntok(SYSTEM[role]) + ntok(ctx_text)
            nout_e = ntok(acct if isinstance(acct, str) else json.dumps(acct, default=str))
            rec = {"role": role, "rule": mock, "model": obj if obj is not None else text, "ms": ms, "tok_in": nin, "tok_out": nout,
                   "tok_in_est": nin_e, "tok_out_est": nout_e, "cached": cached, **self.api.context}
            with open(self.api.log_file.parent / "shadow.jsonl", "a") as f:
                f.write(json.dumps(rec, default=list) + "\n")
            return rule_out
        st.llm(role, nin, nout, ms)
        val = None
        if obj is not None and parse is not None:
            try:
                val = parse(obj)
            except (KeyError, TypeError, ValueError, IndexError):
                val = None
        elif obj is not None:
            val = obj
        if val is None:
            self.fallbacks[role] += 1
            st.extra["llm_fallbacks"] = st.extra.get("llm_fallbacks", 0) + 1
            return rule_out
        return val

    # ------------------------------------------------------------------ c_cur
    def abstract(self, text, jur_name, st):
        """Mask personal literals; the jurisdiction name and procedure words stay."""
        t = text
        n = 0
        for rx, tag in ((_EMAIL, "<EMAIL>"), (_PHONE, "<PHONE>"), (_AMOUNT, "<AMOUNT>"), (_DATE, "<DATE>"), (_IDNO, "<ID>")):
            t, k = rx.subn(tag, t)
            n += k
        t, k = _NAME_INTRO.subn(lambda m: m.group(1) + " <NAME>", t)
        n += k
        t, k = _CITY_INTRO.subn(lambda m: m.group(1) + " <CITY>", t)
        n += k
        t = re.sub(r"\s+", " ", t).strip()
        out = {"template": t, "masked": n}

        def parse(o):
            tpl = re.sub(r"\s+", " ", str(o["template"])).strip()
            return {"template": tpl, "masked": tpl.count("<")} if tpl else None
        return self.emit(st, "cur", text, out, {"request": text}, parse, norm=lambda o: {"template": o["template"]})

    @staticmethod
    def has_pii(text):
        return any(rx.search(text) for rx in (_EMAIL, _PHONE, _DATE, _IDNO, _AMOUNT)) or bool(_NAME_INTRO.search(text)) or bool(_CITY_INTRO.search(text))

    def verbalize(self, region_text, st):
        out = "Composite: " + region_text
        return self.emit(st, "cur", region_text, out, {"verbalize_region": region_text},
                         lambda o: str(o.get("template") or o.get("text") or "") or None, norm=lambda o: {"template": o})

    # ------------------------------------------------------------------ c_int
    def interpret(self, template, ctx, st):
        clar = re.sub(r"\b(Hello|Hi|Dear (?:team|sir|madam)|Good (?:morning|afternoon))[,.! ]*", "", template).strip()
        clar = re.sub(r"\s+", " ", clar)
        out = {"clarified": clar, "atoms": sorted(ctx.atoms)}
        r = self.emit(st, "int", template, out, {"template": template, "context": sorted(ctx.atoms)},
                      lambda o: {"clarified": re.sub(r"\s+", " ", str(o["clarified"])).strip() or None}, norm=lambda o: {"clarified": o["clarified"]})
        return r["clarified"] or clar

    # ------------------------------------------------------------------ c_pln
    def _exemplars(self, step, dom, eco, extra):
        key = (step, dom, eco)
        if key not in self._ex:
            ks = {"elig": ["elig_first", "elig_renew"], "docs": ["docs"], "extra": ["extra"], "appt": ["appt", "appt_prio"], "sub": ["sub"]}[step]
            tx = [p.format(dom=dom, eco=eco, extra=extra, channel="<CHANNEL>") for k in ks for p in lex.PHRASES[k][:3]]
            self._ex[key] = self.senc.encode(tx)
        return self._ex[key]

    @staticmethod
    def _lex_cues(c, extras):
        low = c.lower()
        cues = {
            "elig": bool(re.search(r"eligib|qualif|entitle|entitlement|right to|can be renewed|can renew|meet the conditions", low)),
            "docs": bool(re.search(r"documen|paperwork|proofs|supporting|papers|formalit|dossier|assemble the|gather|pull together", low)),
            "extra": any(x.lower() in low for x in extras),
            "appt": bool(re.search(r"appointment|slot|book|schedul|reserve|meeting|set up|\bfee|settle|\bpay\b|charge", low)),
            "sub": bool(re.search(r"submit|send in|hand in|lodge|\bfile the|\bfile my|notif|let me know|confirm by|confirm via|inform me|send me a", low)),
        }
        return cues

    def plan(self, clarified, ctx, jur_name, st):
        """Decompose the clarified prompt into typed sub-goals grouped by procedure."""
        verbs = r"(?:verify|check|collect|gather|upload|prepare|assess|confirm|validate|have|pay|settle|book|schedule|arrange|obtain|reserve|submit|send|file|lodge|notify|let|inform)"
        clauses = [c.strip(" ,") for c in re.split(r"[.;!?]|,\s*(?:and\s+|then\s+)?|\s+(?:and\s+)?then\s+|\s+and\s+(?=" + verbs + r"\b)", clarified) if c and c.strip(" ,")]
        cur_slug = None
        found = []  # (slug, step, qual, text)
        for c in clauses:
            c2 = re.sub(r"^(?:first|next|also|finally|please|and|then|after that|i need to|i want to|i would like to|we have to|make sure to|help me|can you)[, ]*", "", c, flags=re.I).strip()
            c2 = re.sub(r"^(?:first|next|also|finally|please|and|then)[, ]*", "", c2, flags=re.I).strip()
            m = self.dom_rx.search(c2)
            if m is None and re.search(lex.OFFTOPIC, c2, re.I):
                continue
            slug = self.dom_slug[m.group(0).lower()] if m else cur_slug
            if slug is None:
                continue
            cur_slug = slug
            dom, kind, ea, eb = lex.DOMAINS[slug]
            cues = self._lex_cues(c2, (ea, eb))
            chx = r"(SMS|email|postal letter|mobile app|post)"
            mch = re.search(r"\b(?:by|via|through)\s+" + chx + r"\b", c2) or re.search(r"\bsend me an?\s+" + chx + r"\s+notification", c2)
            text = re.sub(r"\b(?:by|via|through)\s+" + chx + r"\b", "by <CHANNEL>", c2)
            text = re.sub(r"\bsend me an?\s+" + chx + r"\s+notification", "send me a <CHANNEL> notification", text)
            if m is None:
                text = "%s for the %s in %s" % (text, dom, jur_name)
            v = self.senc.encode(text)[0]
            best, bs, bsem = None, -1.0, 0.0
            for step in ("extra", "elig", "sub", "appt", "docs"):
                sem = float((self._exemplars(step, dom, jur_name, ea) @ v).max())
                sc = self.w_lex * (1.0 if cues[step] else 0.0) + (1 - self.w_lex) * sem
                if sc > bs + 1e-9:
                    best, bs, bsem = step, sc, sem
            # a clause is a goal only with an action cue or a very close semantic match to a step exemplar
            if bs < self.theta or not (cues[best] or bsem >= self.p_sem):
                continue
            low = text.lower()
            qual = "renew" if re.search(r"renew|extend|extension", low) else ("first" if re.search(r"first|new |apply for", low) else "")
            found.append((slug, best, qual, text, bs, mch.group(1) if mch else None))
        g = self._graph(found, ctx)
        rule_out = {"goals": [{"procedure": x.obj, "step": x.act, "qualifier": x.key[2] if x.act == "elig" else "",
                               "text": x.text, "channel": (x.slots or {}).get("channel") or None} for x in g.goals]}

        def parse(o):
            fd = []
            for it in o["goals"]:
                slug, step = str(it["procedure"]), str(it["step"])
                if slug not in lex.DOMAINS or step not in ("elig", "docs", "extra", "appt", "sub"):
                    continue
                qual = str(it.get("qualifier") or "")
                ch = it.get("channel") or None
                fd.append((slug, step, qual if qual in ("first", "renew") else "", str(it.get("text") or ""), 1.0, ch))
            return self._graph(fd, ctx) if fd else None
        payload = {"prompt": clarified, "jurisdiction": jur_name,
                   "procedures": {s: lex.DOMAINS[s][0] for s in sorted({x[0] for x in found})} or {s: v[0] for s, v in lex.DOMAINS.items()}}
        return self.emit(st, "pln", clarified, g, payload, parse, norm=lambda _g: rule_out,
                         acct={"goals": [(x.key, x.text) for x in g.goals], "edges": g.edges})

    def _graph(self, found, ctx):
        """Build the goal hypergraph from (slug, step, qualifier, text, score, channel) tuples."""
        # goals grouped by procedure, canonical order within a block
        order = ["elig", "docs", "extra", "appt", "sub"]
        blocks = {}
        for slug, step, qual, text, sc, ch in found:
            b = blocks.setdefault(slug, {})
            if step not in b or sc > b[step][2]:
                b[step] = (qual, text, sc, ch)
        goals, edges = [], []
        gid = 0
        for slug, steps in blocks.items():
            P = "%s.%s" % (ctx.jur, slug)
            ids = {}
            for step in order:
                if step not in steps:
                    continue
                qual, text, _, ch = steps[step]
                if step == "elig":
                    qual = qual or ("renew" if "valid_permit_held" in ctx.atoms else "first")
                key = (P, step, qual if step == "elig" else "")
                g = Goal(gid, key, text, build.goal_omega(P, step), {"channel": ch} if ch else ({"channel": ""} if "<CHANNEL>" in text else {}), step, slug)
                goals.append(g)
                ids[step] = gid
                gid += 1
            pre = [ids[s] for s in ("elig", "docs", "extra") if s in ids]
            if "appt" in ids and pre:
                edges.append((tuple(pre), (ids["appt"],), "seq" if len(pre) == 1 else "and"))
            if "sub" in ids and "appt" in ids:
                edges.append(((ids["appt"],), (ids["sub"],), "seq"))
        g = GoalGraph(goals, edges)
        return g

    # ------------------------------------------------------------------ c_ret
    def reformulate(self, graph, st):
        qs = [g.text for g in graph.goals]
        return self.emit(st, "ret", " | ".join(qs), qs, {"sub_goals": qs},
                         lambda o: [str(q) for q in o["queries"]] if len(o["queries"]) == len(qs) else None,
                         norm=lambda o: {"queries": o})

    # ------------------------------------------------------------------ c_sel / c_com / c_ver / c_exp
    def select(self, goal, cands, ctx_text, st):
        """Pick among compatible candidates: semantic fit to the sub-goal, tie-broken by the composition score."""
        if not cands:
            return self.emit(st, "sel", ctx_text, None, {"sub_goal": goal.text, "candidates": []}, lambda o: None,
                             norm=lambda o: {"choice": None}, acct={"choice": None})
        v = self.senc.encode(goal.text)[0]
        tx = self.senc.encode([c["text"] for c in cands])
        sc = [0.5 * float(tx[i] @ v) + 0.5 * c["eta"] for i, c in enumerate(cands)]
        k = int(np.argmax(sc))
        ids = [c["id"] for c in cands]
        payload = {"sub_goal": goal.text, "context": ctx_text[:2000], "candidates": [{"id": c["id"], "text": c["text"]} for c in cands]}
        return self.emit(st, "sel", ctx_text, k, payload, lambda o: ids.index(o["choice"]) if o.get("choice") in ids else None,
                         norm=lambda i: {"choice": ids[i]}, acct={"choice": cands[k]["id"]})

    def compose(self, ctx_text, plan_obj, st):
        self.emit(st, "com", ctx_text, plan_obj, {"context": ctx_text[:2000], "plan": plan_obj}, lambda o: o,
                  norm=lambda o: {"plan": o})

    def repair_note(self, defects, ctx_text, st):
        self.emit(st, "ver", ctx_text + " " + json.dumps(defects, default=str), {"repair": defects},
                  {"context": ctx_text[:2000], "defects": defects}, lambda o: o, norm=lambda o: json.loads(json.dumps(o, default=str)))

    def explain(self, witnesses, st):
        lines = []
        for gid, w in witnesses.items():
            lines.append(w.get("line", ""))
        txt = "\n".join(l for l in lines if l)
        return self.emit(st, "exp", json.dumps(witnesses, default=str)[:4000], txt, {"witnesses": [l for l in lines if l]},
                         lambda o: str(o["explanation"]) or None, norm=lambda o: {"explanation": o})
