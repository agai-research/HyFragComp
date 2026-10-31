"""Dataset loading and (de)serialisation of fragments and hyperedges."""
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .model import Fragment, Edge, Hypergraph

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "Data" / "main"
INST = ROOT / "Data" / "instances"
SEED = ROOT / "Data" / "seed" / "HyFragComp_BREADY_BE_Dataset"
SEED_ZIP = ROOT / "Data" / "raw" / "bready_dataset_v2.zip"


def ensure_seed():
    """Extract the B-READY seed package (Data/raw/bready_dataset_v2.zip) into Data/seed when it is missing."""
    if not (SEED / "final_json" / "economies.json").exists():
        import zipfile
        with zipfile.ZipFile(SEED_ZIP) as z:
            z.extractall(SEED.parent)
    return SEED

_SETS = ("pre", "post", "inp", "out", "roles")


def frag_to_dict(f):
    d = asdict(f)
    for k in _SETS:
        d[k] = sorted(d[k])
    d["nu"] = list(d["nu"])
    d["supp"] = list(d["supp"])
    return d


def frag_from_dict(d):
    d = dict(d)
    for k in _SETS:
        d[k] = frozenset(d[k])
    d["nu"] = tuple(d["nu"])
    d["supp"] = tuple(d.get("supp", ()))
    return Fragment(**d)


def edge_to_dict(e):
    return {"eid": e.eid, "tail": list(e.tail), "head": list(e.head), "pat": e.pat, "guard": e.guard, "route": e.route}


def edge_from_dict(d):
    return Edge(d["eid"], tuple(d["tail"]), tuple(d["head"]), d["pat"], d.get("guard", ""), d.get("route", ""))


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def jsave(obj, path, indent=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_cache = {}


def main_data():
    """Parsed main dataset (cached per process)."""
    if "m" not in _cache:
        m = {k: jload(MAIN / (k + ".json")) for k in ("economies", "procedures", "lexicon")}
        m["fragments"] = {d["fid"]: d for d in jload(MAIN / "fragments.json")}
        m["edges"] = {d["eid"]: d for d in jload(MAIN / "hyperedges.json")}
        _cache["m"] = m
    return _cache["m"]


def load_repo(proc_ids):
    """Build the shared base hypergraph from the main dataset for the listed procedures."""
    m = main_data()
    hg = Hypergraph()
    ecos = set()
    procs = {p["pid"]: p for p in m["procedures"]}
    for pid in proc_ids:
        p = procs[pid]
        ecos.add(p["iso3"])
        for fid in p["fragments"]:
            hg.add_fragment(frag_from_dict(m["fragments"][fid]))
        for eid in p["edges"]:
            hg.add_edge(edge_from_dict(m["edges"][eid]))
    for iso in sorted(ecos):
        hg.add_fragment(frag_from_dict(m["fragments"][iso + ".ent"]))
    return hg
