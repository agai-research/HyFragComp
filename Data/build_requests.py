"""Construct requests with known sub-goal structure, composition records and requester populations.

Everything here is built from construction logic (procedure keys, step order, phrase templates); no method output is used.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfc import build, lex
from hfc.data import main_data

STEPS = ["elig", "docs", "extra", "appt", "sub"]
SIZE = {"light": 3, "std": 4, "ext": 5}
DATA_ITEMS = ["id_scan", "proofs", "payment_method", "extra_doc", "application_form"]
BASE_ATOMS = ["applicant_present"]


def zipf_weights(n, s=1.1):
    w = 1.0 / np.arange(1, n + 1) ** s
    return w / w.sum()


def kind_split(rng, n):
    """Block kinds whose goal counts add up to n (3 = light, 4 = std, 5 = ext)."""
    opts = []

    def rec(rest, cur):
        if rest == 0:
            opts.append(list(cur))
            return
        for k in (3, 4, 5):
            if rest - k >= 0 and (rest - k == 0 or rest - k >= 3):
                rec(rest - k, cur + [k])
    rec(n, [])
    pick = opts[int(rng.integers(len(opts)))]
    inv = {3: "light", 4: "std", 5: "ext"}
    return [inv[k] for k in pick]


def steps_of(kind):
    return list(build.KIND_GOALS[kind])


def route_of_steps(steps, qual):
    s = set(steps)
    if "extra" in s:
        return "X" + ("f" if qual == "first" else "r")
    if "elig" in s:
        return "S" + ("f" if qual == "first" else "r")
    return "L"


class Gen:
    """Request and record constructor over a set of procedures (a repository instance)."""

    def __init__(self, cfg, pids, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        m = main_data()
        self.eco = {e["iso3"]: e for e in m["economies"]}
        pr = {p["pid"]: p for p in m["procedures"]}
        self.procs = [pr[p] for p in pids]
        self.by_eco = {}
        for p in self.procs:
            self.by_eco.setdefault(p["iso3"], []).append(p)
        self.ecos = list(self.by_eco)
        self.rank = {p["pid"]: i for i, p in enumerate(self.procs)}

    # ------------------------------------------------------------------ choices
    def pick_blocks(self, kinds, eco=None, zipf=1.1, exclude=()):
        rng = self.rng
        if eco is None:
            w = zipf_weights(len(self.ecos), zipf)
            eco = self.ecos[int(rng.choice(len(self.ecos), p=w))]
        pool = [p for p in self.by_eco[eco] if p["pid"] not in exclude]
        if len(pool) < len(kinds):
            pool = self.by_eco[eco]
        w = zipf_weights(len(pool), zipf)
        idx = rng.choice(len(pool), size=min(len(kinds), len(pool)), replace=False, p=w)
        return eco, [(pool[i], k) for i, k in zip(idx, kinds)]

    # ------------------------------------------------------------------ text
    def _phrase(self, key, p, slot=None, unseen=False):
        rng = self.rng
        dom, kind, ea, eb = lex.DOMAINS[p["slug"]]
        eco = self.eco[p["iso3"]]["name"]
        t = lex.PHRASES[key][int(rng.integers(len(lex.PHRASES[key])))]
        t = t.format(dom=dom, eco=eco, extra=ea, channel=slot or "SMS")
        if unseen:
            t = substitute(t, rng, 0.8)
        return t

    def clauses(self, steps_by_block, qual, prio, channel, unseen_p):
        """One sentence per goal, in block order."""
        rng = self.rng
        out = []
        for p, steps in steps_by_block:
            for s in STEPS:
                if s not in steps:
                    continue
                key = {"elig": "elig_first" if qual == "first" else "elig_renew", "docs": "docs", "extra": "extra",
                       "appt": "appt_prio" if prio else "appt", "sub": "sub"}[s]
                out.append(self._phrase(key, p, channel if s == "sub" else None, rng.random() < unseen_p))
        return out

    def person(self):
        rng = self.rng
        n = "%s %s" % (lex.FIRST[int(rng.integers(len(lex.FIRST)))], lex.LAST[int(rng.integers(len(lex.LAST)))])
        mail = re.sub(r"[^a-z]", "", n.lower()) + str(int(rng.integers(10, 99))) + "@mail.com"
        return n, mail

    def render(self, clauses, words=None, pii=True, n_trap=0, n_fill=None):
        """Assemble the prompt text: personal intro, goal clauses with connectors, filler and trap sentences."""
        rng = self.rng
        intro = []
        if pii:
            n, mail = self.person()
            city = lex.CITIES[int(rng.integers(len(lex.CITIES)))]
            intro = ["Hello, my name is %s and my email is %s." % (n, mail), "I live in %s and my phone is +216 %d." % (city, int(rng.integers(20000000, 99999999)))]
        conn = ["First, ", "Next, ", "Also, ", "Then ", "Finally, "]
        body = []
        for i, c in enumerate(clauses):
            lead = conn[min(i, len(conn) - 1)] if len(clauses) > 1 else "Please "
            body.append(lead + c[0].lower() + c[1:] + ".")
        extras = []
        for _ in range(n_trap):
            extras.append(lex.TRAP_FILLER[int(rng.integers(len(lex.TRAP_FILLER)))])
        if words is not None:
            cur = len(" ".join(intro + body + extras).split())
            while cur < words:
                s = lex.FILLER[int(rng.integers(len(lex.FILLER)))]
                extras.append(s)
                cur += len(s.split())
        elif n_fill:
            for _ in range(n_fill):
                extras.append(lex.FILLER[int(rng.integers(len(lex.FILLER)))])
        parts = intro + body
        for s in extras:
            parts.insert(int(rng.integers(len(intro), len(parts) + 1)), s)
        return " ".join(parts)

    def terse(self, blocks):
        """Ten-word style prompt: one domain mention followed by a bare list of goals."""
        p, steps = blocks[0]
        dom = lex.DOMAINS[p["slug"]][0]
        eco = self.eco[p["iso3"]]["name"]
        verbs = {"elig": "verify eligibility", "docs": "verify documents", "extra": "validate extra document", "appt": "book appointment", "sub": "submit application"}
        return "%s %s: %s." % (dom, eco, ", ".join(verbs[s] for s in STEPS if s in steps))

    # ------------------------------------------------------------------ requests
    def ctx(self, iso, qual, prio, deadline=90):
        atoms = list(BASE_ATOMS) + ["jur:" + iso, "valid_permit_held" if qual == "renew" else "no_valid_permit"]
        if prio:
            atoms.append("priority_status")
        return {"atoms": sorted(atoms), "data": sorted(DATA_ITEMS), "roles": ["agency", "citizen"], "deadline": deadline, "jur": iso}

    def request(self, rid, t, user, blocks, qual, prio=False, words=None, unseen_p=0.0, n_trap=0, n_fill=0, pii=True,
                terse=False, channel=None, branch="", steps_override=None):
        """blocks: list of (procedure dict, kind or list of steps)."""
        rng = self.rng
        sb = [(p, steps_of(k) if isinstance(k, str) else list(k)) for p, k in blocks]
        iso = sb[0][0]["iso3"]
        channel = channel or lex.CHANNELS[int(rng.integers(len(lex.CHANNELS)))]
        if terse:
            text = self.terse(sb)
            if pii:
                n, mail = self.person()
                text = "Hello, I am %s. %s" % (n, text)
        else:
            text = self.render(self.clauses(sb, qual, prio, channel, unseen_p), words=words, pii=pii, n_trap=n_trap, n_fill=n_fill)
        goals = []
        for p, steps in sb:
            for s in STEPS:
                if s in steps:
                    goals.append([p["pid"], s, qual if s == "elig" else ""])
        truth = {"blocks": [{"pid": p["pid"], "steps": steps, "qual": qual, "prio": bool(prio)} for p, steps in sb],
                 "goals": goals, "channel": channel if any("sub" in s for _, s in sb) else None, "branch": branch,
                 "n_words": len(text.split())}
        return {"rid": rid, "text": text, "t": float(t), "user": user, "ctx": self.ctx(iso, qual, prio), "truth": truth}

    # ------------------------------------------------------------------ records
    def record(self, cid, t, user, blocks, qual, prio=False, scope="shared", users=None, hits=1):
        """Stored composition record: goal graph with canonical texts and the reference fragments covering each goal."""
        rng = self.rng
        sb = [(p, steps_of(k) if isinstance(k, str) else list(k)) for p, k in blocks]
        iso = sb[0][0]["iso3"]
        goals, edges, phi, gid = [], [], {}, 0
        channel = lex.CHANNELS[int(rng.integers(len(lex.CHANNELS)))]
        for p, steps in sb:
            ids = {}
            P = p["pid"]
            for s in STEPS:
                if s not in steps:
                    continue
                key = {"elig": "elig_first" if qual == "first" else "elig_renew", "docs": "docs", "extra": "extra",
                       "appt": "appt_prio" if prio else "appt", "sub": "sub"}[s]
                text = self._phrase(key, p, "<CHANNEL>" if s == "sub" else None)
                if s == "sub":
                    text = text.replace(channel, "<CHANNEL>")
                f = {"elig": P + (".elig_f" if qual == "first" else ".elig_r"), "docs": P + ".docs", "extra": P + ".extra_a",
                     "appt": None, "sub": P + ".sub"}[s]
                if s == "appt":
                    phi[gid] = [P + ".fee", P + (".appt_p" if prio else ".appt_s")]
                else:
                    phi[gid] = [f]
                goals.append({"gid": gid, "key": [P, s, qual if s == "elig" else ""], "text": text,
                              "slots": {"channel": channel} if s == "sub" else {}})
                ids[s] = gid
                gid += 1
            pre = [ids[s] for s in ("elig", "docs", "extra") if s in ids]
            if "appt" in ids and pre:
                edges.append([pre, [ids["appt"]], "seq" if len(pre) == 1 else "and"])
            if "sub" in ids and "appt" in ids:
                edges.append([[ids["appt"]], [ids["sub"]], "seq"])
        frags = [iso + ".ent"] + sorted({f for v in phi.values() for f in v})
        return {"cid": cid, "t": float(t), "user": user, "scope": scope, "users": list(users or [user]), "hits": hits,
                "iso": iso, "qual": qual, "prio": bool(prio), "goals": goals, "edges": edges, "phi": {str(k): v for k, v in phi.items()},
                "frags": frags, "blocks": [{"pid": p["pid"], "steps": s} for p, s in sb]}


def substitute(text, rng, p):
    """Replace known step words by synonyms the sentence encoder never saw in training."""
    out = text
    for w, s in sorted(lex.SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b%s\b" % re.escape(w), out) and rng.random() < p:
            out = re.sub(r"\b%s\b" % re.escape(w), s, out, count=1)
    return out
