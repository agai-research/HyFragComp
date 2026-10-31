"""Run the HyFragComp prototype on a request.

    python HyFragComp.py --selftest
    python HyFragComp.py --prompts                      # serve Prompts/prompts.json on the scenario repository
    python HyFragComp.py --text "..." --jur MAR [--renew] [--priority] [--out out/run]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hfc.util import ensure_hash_seed  # noqa: E402

ensure_hash_seed()
from hfc import config, report  # noqa: E402
from hfc.llm import ROLES
from hfc.methods import make
from hfc.model import Ctx
from hfc.repo import Models, Repo
from hfc.req import Request, request_from
from hfc.util import Stats, path

BASE_ATOMS = ["applicant_present"]
ITEMS = ["application_form", "extra_doc", "id_scan", "payment_method", "proofs"]


def check_paths():
    """Fail early when the project layout is wrong."""
    for p in ("configs/default.yaml", "Data/main/fragments.json", "Prompts/prompts.json"):
        if not (ROOT / p).exists():
            raise SystemExit("missing project file: %s" % p)


def selftest(models, cfg):
    """Call each of the eight role configurations once and print the outcome."""
    from hfc.llm import Roles
    r = Roles(cfg, models.senc)
    st = Stats()
    ctx = Ctx(frozenset(["applicant_present", "jur:MAR", "no_valid_permit"]), frozenset(ITEMS), frozenset(["citizen", "agency"]), 90, "MAR")
    a = r.abstract("Hello, I am Sara Benali, email sara@mail.com. Please submit my long-stay residence permit application in Morocco.", "Morocco", st)
    clar = r.interpret(a["template"], ctx, st)
    g = r.plan(clar, ctx, "Morocco", st)
    r.reformulate(g, st)
    r.select(g.goals[0], [{"id": "x", "text": g.goals[0].text, "eta": 0.5}], clar, st) if g.goals else None
    r.compose(clar, {"goals": len(g.goals)}, st)
    r.repair_note([{"check": "V4"}], clar, st)
    r.explain({0: {"line": "reused from a stored record"}}, st)
    seen = {k for k, v in st.calls.items() if v}
    for k in ROLES:
        print("  %-4s calls=%d tokens_in=%d tokens_out=%d" % (k, st.calls.get(k, 0), st.tok_in.get(k, 0), st.tok_out.get(k, 0)))
    ok = seen == set(ROLES)
    print("self-test %s: %d of %d roles called" % ("passed" if ok else "FAILED", len(seen), len(ROLES)))
    return ok


def build_request(a, n=0):
    atoms = BASE_ATOMS + ["jur:" + a.jur, "valid_permit_held" if a.renew else "no_valid_permit"] + (["priority_status"] if a.priority else [])
    ctx = Ctx(frozenset(atoms), frozenset(ITEMS), frozenset(["citizen", "agency"]), a.deadline, a.jur)
    return Request("cli%d" % n, a.text, a.t, a.user, ctx, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--jur", default="MAR")
    ap.add_argument("--renew", action="store_true")
    ap.add_argument("--priority", action="store_true")
    ap.add_argument("--deadline", type=int, default=90)
    ap.add_argument("--t", type=float, default=10.0)
    ap.add_argument("--user", default="cli_user")
    ap.add_argument("--prompts", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--method", default="ours", choices=["ours", "scratch", "whole", "nore", "cbr", "cafc"])
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--out", default="out/run")
    a = ap.parse_args()
    check_paths()
    cfg = config.load(*a.config)
    models = Models(cfg, cfg["seed"])
    if a.selftest and not a.prompts and not a.text:
        return 0 if selftest(models, cfg) else 1
    repo = Repo(cfg, models, a.size)
    m = make(a.method, repo, cfg, models, cfg["seed"])
    if a.selftest and not selftest(models, cfg):
        return 1
    reqs = []
    if a.prompts:
        reqs += [request_from(d) for d in json.loads((ROOT / "Prompts" / "prompts.json").read_text())["queries"]]
    if a.text:
        reqs.append(build_request(a))
    out = path(*a.out.split("/"))
    for q in reqs:
        r = m.serve(q)
        print(report.summary(r, q.rid))
        report.write(r, out, q.rid)
    print("results in %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
