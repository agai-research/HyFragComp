"""Build the fragment hypergraph from the dataset and report its statistics.

    python hypergraph.py                    # rebuild Data/main when missing, print the statistics of the full repository
    python hypergraph.py --rebuild          # rebuild the dataset from the B-READY seed first
    python hypergraph.py --size 1000 --export out/graph.json
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hfc import config, data
from hfc.repo import Models, Repo


def stats(hg):
    pats = Counter(e.pat for e in hg.edges.values())
    kinds = Counter(f.kind for f in hg.frags.values())
    return {"fragments": len(hg.frags), "hyperedges": len(hg.edges), "by_pattern": dict(pats), "by_kind": dict(kinds),
            "jurisdictions": len({f.split(".")[0] for f in hg.frags})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--size", type=int, default=None, help="repository size; the whole dataset when omitted")
    ap.add_argument("--export", default=None)
    a = ap.parse_args()
    if a.rebuild or not (data.MAIN / "fragments.json").exists():
        subprocess.check_call([sys.executable, str(ROOT / "Data" / "build_base.py")])
    if a.size:
        cfg = config.load()
        repo = Repo(cfg, Models(cfg, cfg["seed"]), a.size)
        hg = repo.base
    else:
        m = data.main_data()
        hg = data.load_repo([p["pid"] for p in m["procedures"]])
    s = stats(hg)
    print(json.dumps(s, indent=1))
    if a.export:
        out = Path(a.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"stats": s, "fragments": [data.frag_to_dict(f) for f in hg.frags.values()],
                                   "hyperedges": [data.edge_to_dict(e) for e in hg.edges.values()]}))
        print("exported to", out)


if __name__ == "__main__":
    main()
