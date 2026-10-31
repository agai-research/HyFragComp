"""Construct the main dataset from the B-READY seed package and the domain lexicon."""
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfc import build, lex
from hfc.data import MAIN, SEED, ROOT, ensure_seed, jload, jsave, frag_to_dict, edge_to_dict, sha

HOT_ECO = ["MAR", "PRT", "EST", "GRC", "ROU", "MEX", "COL", "NZL", "SGP", "GEO"]
HOT_DOM = ["longstay", "studperm", "workperm", "llc", "famperm", "sole", "branch", "taxid", "buslic", "socaff", "drivlic"]


def economies():
    ensure_seed()
    base = {e["iso3"]: e for e in jload(SEED / "final_json" / "economies.json")}
    rows = {r["iso3"]: r for r in csv.DictReader(open(SEED / "raw" / "business_entry_pillar_scores_2024.csv", encoding="utf-8"))}
    out = []
    for iso, e in sorted(base.items()):
        r = rows[iso]
        p1, p2, p3 = float(r["pillar_1"]), float(r["pillar_2"]), float(r["pillar_3"])
        out.append({"iso3": iso, "name": e["name"], "region": e["region"], "income_group": e["income_group"], "year": 2024,
                    "pillar_1": round(p1, 3), "pillar_2": round(p2, 3), "pillar_3": round(p3, 3),
                    "qos_factor": round(build.qos_factor(p2), 4), "provenance": "source+derived"})
    return out


def main():
    MAIN.mkdir(parents=True, exist_ok=True)
    eco = economies()
    jsave(eco, MAIN / "economies.json", indent=1)
    # business-entry seed layers kept as source-derived reference tables
    for name in ("services", "resources", "requirements", "constraints", "fragments"):
        src = jload(SEED / "final_json" / (name + ".json"))
        jsave(src, MAIN / ("seed_" + name + ".json"), indent=1)
    shutil.copy(SEED / "raw" / "business_entry_pillar_scores_2024.csv", MAIN / "seed_pillar_scores_2024.csv")
    jsave({"generic_atoms": "an atom 'a:p' specialises the generic atom 'a'", "conflicts": [list(c) for c in __import__("hfc.model", fromlist=["x"]).CONFLICTS]},
          MAIN / "ontology.json", indent=1)
    jsave({"domains": {k: list(v) for k, v in lex.DOMAINS.items()}, "phrases": lex.PHRASES, "frag_text": lex.FRAG_TEXT,
           "channels": lex.CHANNELS, "filler": lex.FILLER, "trap_filler": lex.TRAP_FILLER}, MAIN / "lexicon.json", indent=1)

    # procedure order: hot set first (economy-major), then a seeded shuffle of the rest
    hot = [(e, d) for e in HOT_ECO for d in HOT_DOM]
    hot_set = set(hot)
    rest = [(e["iso3"], d) for e in eco for d in lex.DOMAINS if (e["iso3"], d) not in hot_set]
    random.Random(7).shuffle(rest)
    order = hot + rest
    name_of = {e["iso3"]: e["name"] for e in eco}
    fac = {e["iso3"]: e["qos_factor"] for e in eco}

    frags, edges, procs = [], [], []
    for iso in sorted(name_of):
        frags.append(frag_to_dict(build.make_entry(iso, name_of[iso], fac[iso])))
    for i, (iso, slug) in enumerate(order):
        fs, es = build.make_procedure(iso, name_of[iso], slug, fac[iso])
        dom, kind, _, _ = lex.DOMAINS[slug]
        procs.append({"pid": build.pid(iso, slug), "iso3": iso, "slug": slug, "domain": dom, "kind": kind, "rank": i,
                      "hot": (iso, slug) in hot_set,
                      "fragments": [f.fid for f in fs], "edges": [e.eid for e in es]})
        frags += [frag_to_dict(f) for f in fs]
        edges += [edge_to_dict(e) for e in es]
    # provenance links between business-entry seed services and company-lane fragments
    link = {"ent": ["svc_identity"], "elig_f": ["svc_domestic_rules"], "elig_r": ["svc_domestic_rules"], "docs": ["svc_company_info", "svc_shareholder_info", "svc_bo_filing"],
            "fee": ["svc_simplified_registration"], "extra_a": ["svc_risk_license"], "appt_s": ["svc_exchange"], "appt_p": ["svc_exchange"], "sub": ["svc_ubi", "svc_domestic_efficiency"]}
    jsave(link, MAIN / "seed_service_links.json", indent=1)
    jsave(procs, MAIN / "procedures.json")
    jsave(frags, MAIN / "fragments.json")
    jsave(edges, MAIN / "hyperedges.json")
    jsave({"F1": "MAR.ent", "F2": "MAR.longstay.elig_f", "F2r": "MAR.longstay.elig_r", "F3": "MAR.longstay.docs", "F4": "MAR.longstay.fee",
           "F5": "MAR.longstay.appt_s", "F7": "MAR.longstay.appt_p", "F6": "MAR.longstay.sub", "F16": "MAR.longstay.extra_a",
           "F17": "MAR.longstay.extra_b", "e1": "MAR.longstay.e1.Sf", "e2": "MAR.longstay.e2.Sf", "e3": "MAR.longstay.e3", "e4": "MAR.longstay.e4",
           "company_lane": "MAR.llc", "note": "reference scenario of the paper instantiated on the Morocco jurisdiction"}, MAIN / "scenario_map.json", indent=1)
    manifest = {"economies": len(eco), "procedures": len(procs), "hot_procedures": sum(p["hot"] for p in procs),
                "fragments": len(frags), "hyperedges": len(edges),
                "edges_by_pattern": {k: sum(1 for e in edges if e["pat"] == k) for k in ("seq", "and", "xor", "loop")}}
    manifest["checksums"] = {p.name: sha(p) for p in sorted(MAIN.glob("*.json"))}
    jsave(manifest, MAIN / "manifest.json", indent=1)
    print({k: v for k, v in manifest.items() if k != "checksums"})


if __name__ == "__main__":
    main()
