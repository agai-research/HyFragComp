"""Train the structural encoder and embed the whole repository (run once; results are cached)."""
import json
import sys
import time

import numpy as np

from . import config, data, enc
from .allset import AllSet
from .util import path

CK = path("results", "ckpt")
CA = path("results", "cache")


def run(cfg=None, seed=0, force=False, log=print):
    cfg = cfg or config.load()
    f = CA / ("emb_%d.npz" % seed)
    if f.exists() and not force:
        return
    m = data.main_data()
    tenc = enc.sentence_encoder(cfg, seed)
    fids, h0 = enc.all_h0(tenc, m)
    row = {x: i for i, x in enumerate(fids)}
    hg = data.load_repo([p["pid"] for p in m["procedures"]])
    procs = m["procedures"]
    isos = sorted({p["iso3"] for p in procs})
    val = tuple(isos[-cfg["encoder"]["val_jurs"]:])
    t0 = time.time()
    model, rep = enc.train_allset(cfg, h0, row, procs, hg, log=log, seed=seed, val_isos=val)
    rep["train_seconds"] = round(time.time() - t0, 1)
    rep["val_economies"] = list(val)
    CK.mkdir(parents=True, exist_ok=True)
    model.save(CK / ("allset_%d.npz" % seed))
    z = enc.embed_all(model, procs, hg, h0, row)
    CA.mkdir(parents=True, exist_ok=True)
    np.savez(f, fids=np.array(fids), h0=h0, zstr=z)
    (CK / ("allset_%d_report.json" % seed)).write_text(json.dumps(rep))
    log("encoder report: %s" % {k: v for k, v in rep.items() if k != "curve"})


if __name__ == "__main__":
    run(force="--force" in sys.argv)
