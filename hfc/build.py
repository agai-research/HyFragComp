"""Constructors for procedures, fragments and hyperedges of the repository."""
from .model import Fragment, Edge, INF_DAY
from . import lex

BASE_DUR = {"ent": 1, "elig": 3, "docs": 7, "fee": 1, "extra": 4, "appt_s": 14, "appt_p": 3, "sub": 2}
ROUTES = ["L", "Sf", "Sr", "Xf", "Xr", "Xfb", "Xrb"]
ROLES_STD = frozenset({"citizen", "agency"})


def qos_factor(pillar2):
    """Faster procedures in economies with a better public-service score."""
    return 0.6 + 0.8 * (1.0 - pillar2 / 100.0)


def pid(iso3, slug):
    return "%s.%s" % (iso3, slug)


def ent_id(iso3):
    return "%s.ent" % iso3


def make_entry(iso3, eco_name, factor):
    d = max(1, round(BASE_DUR["ent"] * factor))
    return Fragment(
        fid=ent_id(iso3), text=lex.FRAG_TEXT["ent"].format(eco=eco_name),
        pre=frozenset({"applicant_present", "jur:" + iso3}), post=frozenset({"dossier_open:" + iso3}),
        inp=frozenset({"id_scan"}), out=frozenset({"dossier"}), roles=ROLES_STD, dmin=d, dmax=d * 3,
        ftype="ENT", nu=(0, INF_DAY), proc=iso3, step="ent")


def make_procedure(iso3, eco_name, slug, factor, window=(0, INF_DAY)):
    """Return (fragments, edges) of one administrative procedure."""
    dom, kind, ex_a, ex_b = lex.DOMAINS[slug]
    P = pid(iso3, slug)
    ent = ent_id(iso3)
    jur = "jur:" + iso3
    base = frozenset({"dossier_open:" + iso3, jur})

    def dur(step):
        d = max(1, round(BASE_DUR[step] * factor))
        return d, d * 3

    def frag(name, step, pre, post, inp, out, roles=ROLES_STD, ftype="INT", guard="", prio=0, dstep=None, variant=""):
        a, b = dur(dstep or step)
        return Fragment(
            fid="%s.%s" % (P, name), text=lex.FRAG_TEXT[name].format(dom=dom, eco=eco_name, extra=ex_a, extra_alt=ex_b),
            pre=frozenset(pre), post=frozenset(post), inp=frozenset(inp), out=frozenset(out), roles=roles,
            dmin=a, dmax=b, ftype=ftype, nu=window, proc=P, step=step, guard=guard, prio=prio, variant=variant)

    fr = {
        "elig_f": frag("elig_f", "elig", base | {"no_valid_permit"}, {"eligible:" + P}, set(), {"eligibility_decision"}, variant="first"),
        "elig_r": frag("elig_r", "elig", base | {"valid_permit_held"}, {"eligible:" + P}, set(), {"eligibility_decision"}, variant="renew"),
        "docs": frag("docs", "docs", base, {"docs_ok:" + P}, {"proofs"}, {"verified_dossier"}),
        "fee": frag("fee", "fee", base, {"fee_paid:" + P}, {"payment_method"}, {"receipt"}),
        "extra_a": frag("extra_a", "extra", base, {"extra_ok:" + P}, {"extra_doc"}, {"extra_review"}),
        "extra_b": frag("extra_b", "extra", base, {"extra_ok:" + P}, {"extra_doc"}, {"extra_review"},
                        roles=frozenset({"employer_agent"})),
        "appt_s": frag("appt_s", "appt", base | {"docs_ok:" + P, "fee_paid:" + P}, {"appt_booked:" + P}, set(), {"appointment"},
                       ftype="VAR", dstep="appt_s", variant="standard"),
        "appt_p": frag("appt_p", "appt", base | {"docs_ok:" + P, "fee_paid:" + P}, {"appt_booked:" + P}, set(), {"appointment"},
                       ftype="VAR", guard="priority_status", prio=1, dstep="appt_p", variant="priority"),
        "sub": frag("sub", "sub", base | {"appt_booked:" + P}, {"submitted:" + P, "notified:" + P}, {"application_form"},
                    {"receipt_of_submission"}, ftype="TER"),
    }
    F = {f"{P}.{k}": v for k, v in fr.items()}
    edges = []
    routes = {
        "L": ["docs", "fee"],
        "Sf": ["elig_f", "docs", "fee"], "Sr": ["elig_r", "docs", "fee"],
        "Xf": ["elig_f", "docs", "fee", "extra_a"], "Xr": ["elig_r", "docs", "fee", "extra_a"],
        "Xfb": ["elig_f", "docs", "fee", "extra_b"], "Xrb": ["elig_r", "docs", "fee", "extra_b"],
    }
    for r, mem in routes.items():
        ids = tuple(f"{P}.{m}" for m in mem)
        edges.append(Edge(f"{P}.e1.{r}", (ent,), ids, "and", route=r))
        edges.append(Edge(f"{P}.e2.{r}", ids, (f"{P}.appt_s", f"{P}.appt_p"), "xor", guard="priority_status", route=r))
    edges.append(Edge(f"{P}.e3", (f"{P}.appt_s",), (f"{P}.sub",), "seq"))
    edges.append(Edge(f"{P}.e4", (f"{P}.appt_p",), (f"{P}.sub",), "seq"))
    if kind == "company":
        edges.append(Edge(f"{P}.e5", (f"{P}.sub",), (f"{P}.docs",), "loop", guard="rejected"))
    return list(F.values()), edges


def goal_omega(P, step):
    return {
        "elig": frozenset({"eligible:" + P}),
        "docs": frozenset({"docs_ok:" + P}),
        "extra": frozenset({"extra_ok:" + P}),
        "appt": frozenset({"fee_paid:" + P, "appt_booked:" + P}),
        "sub": frozenset({"submitted:" + P, "notified:" + P}),
    }[step]


KIND_GOALS = {"light": ["docs", "appt", "sub"], "std": ["elig", "docs", "appt", "sub"], "ext": ["elig", "docs", "extra", "appt", "sub"]}


def route_name(kind, qual):
    if kind == "light":
        return "L"
    return ("S" if kind == "std" else "X") + ("f" if qual == "first" else "r")


def reference_fragments(P, iso3, kind, qual, prio):
    """Fragments of the reference workflow of one block (without the shared entry)."""
    out = [f"{P}.docs", f"{P}.fee", f"{P}.appt_p" if prio else f"{P}.appt_s", f"{P}.sub"]
    if kind != "light":
        out.append(f"{P}.elig_f" if qual == "first" else f"{P}.elig_r")
    if kind == "ext":
        out.append(f"{P}.extra_a")
    return out


DATA_ITEMS = ("id_scan", "proofs", "payment_method", "extra_doc", "application_form")
