"""Run experiment series: python main-exp.py --exp e1 e5 e6 e7 x1 x2 x3 --runs 5 --profile full

e1 Fig. 4, e5 Fig. 5, e7 Fig. 6, e6 Fig. 7 of the paper; x1, x2, x3 the additional experiments (RQ2, RQ3, RQ5).
See REPRODUCE.md for the command, run time and outputs of each figure and table."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hfc.util import ensure_hash_seed  # noqa: E402

ensure_hash_seed()
from Exp import core  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["e1", "e5", "e6", "e7", "x1", "x2", "x3"])
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--profile", default="full", choices=["smoke", "full"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--no-tuned", action="store_true")
    ap.add_argument("--only", choices=["run", "aggregate", "figures"], default=None, help="run a single stage from stored results")
    a = ap.parse_args()
    for e in a.exp:
        t0 = time.time()
        if a.only in (None, "run"):
            core.run_series(e, a.runs, a.profile, a.methods, a.jobs, tuned=not a.no_tuned)
        if a.only in (None, "run", "aggregate", "figures"):
            agg = core.aggregate(e, a.profile, a.methods, a.runs)
            core.write_tables(e, agg, a.profile)
            core.render_figures(e, agg, a.profile)
        print("%s finished in %.0fs" % (e, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
