"""Text encoder fitting, AllSet training by masked hyperedge completion and fragment/goal embeddings."""
import pickle
import time

import numpy as np

from . import nn, data, lex
from .allset import AllSet, Packed, PATTERNS
from .emb import SentEnc
from .util import path

CACHE = path("results", "cache")


def _corpus(seed=0, n_frag=9000, n_req=3000):
    """Fragment descriptions plus request-side phrasings over random domain/jurisdiction pairs."""
    rng = np.random.default_rng(seed)
    m = data.main_data()
    fr = [d["text"] for d in m["fragments"].values()]
    idx = rng.choice(len(fr), min(n_frag, len(fr)), replace=False)
    docs = [fr[i] for i in idx]
    slugs = list(lex.DOMAINS)
    ecos = [e["name"] for e in m["economies"]]
    for _ in range(n_req):
        slug = slugs[rng.integers(len(slugs))]
        dom, _, ea, eb = lex.DOMAINS[slug]
        key = list(lex.PHRASES)[rng.integers(len(lex.PHRASES))]
        ph = lex.PHRASES[key][rng.integers(4)]
        docs.append(ph.format(dom=dom, eco=ecos[rng.integers(len(ecos))], extra=ea, channel="<CHANNEL>"))
    docs += lex.FILLER + lex.TRAP_FILLER
    return docs


def concept_groups(seed=0, hold_out=False):
    """Paraphrase groups: one concept = (domain, jurisdiction, step variant); texts = request phrasings + fragment text."""
    rng = np.random.default_rng(seed)
    m = data.main_data()
    ecos = [e["name"] for e in m["economies"]]
    groups, meta = [], []
    ch = "<CHANNEL>"
    for slug, (dom, kind, ea, eb) in lex.DOMAINS.items():
        for eco in ecos:
            fmt = dict(dom=dom, eco=eco, extra=ea, extra_alt=eb, channel=ch)
            spec = [("elig_first", "elig_first", "elig_f"), ("elig_renew", "elig_renew", "elig_r"), ("docs", "docs", "docs"),
                    ("extra", "extra", "extra_a"), ("appt", "appt", "appt_s"), ("appt_prio", "appt_prio", "appt_p"),
                    ("sub", "sub", "sub"), (None, None, "fee")]
            for var, pk, fk in spec:
                texts = [lex.FRAG_TEXT[fk].format(**fmt)]
                if fk == "fee":
                    texts += [x.format(**fmt) for x in ("Pay the fee for the {dom} in {eco}", "settle the administrative fees of the {dom} in {eco}",
                                                        "Fee receipt of the {dom} in {eco}", "payment of the application fee for the {dom} in {eco}")]
                if pk:
                    ph = lex.PHRASES[pk]
                    texts += [x.format(**fmt) for x in (ph[:3] if hold_out else ph)]
                groups.append(texts)
                meta.append((slug, eco, var or "fee"))
    return groups, meta


def sentence_encoder(cfg=None, seed=0, refit=False, log=print):
    f = CACHE / f"sent_{seed}.pkl"
    if f.exists() and not refit:
        return pickle.loads(f.read_bytes())
    groups, meta = concept_groups(seed, hold_out=False)
    enc = SentEnc(256, 192, 96, seed)
    corp = [t for g in groups[::3] for t in g] + lex.FILLER + lex.TRAP_FILLER
    enc.fit_text(corp)
    by_dom = {}
    for i, (sl, eco, var) in enumerate(meta):
        by_dom.setdefault((sl, eco), []).append(i)
    by_var = {}
    for i, (sl, eco, var) in enumerate(meta):
        by_var.setdefault(var, []).append(i)
    keys = list(by_dom)
    vars_ = list(by_var)

    def hard(rng, batch):
        # half of the batch: all step variants of a few (domain, jurisdiction) pairs; half: one step across domains
        ids = []
        for k in rng.choice(len(keys), batch // 16, replace=False):
            ids += by_dom[keys[k]]
        v = vars_[rng.integers(len(vars_))]
        ids += list(rng.choice(by_var[v], batch - len(ids), replace=False))
        return np.array(list(dict.fromkeys(ids))[:batch])

    enc.fit_contrastive(groups, steps=cfg["encoder"].get("sent_steps", 1200) if cfg else 1200, batch=96, seed=seed, log=log, hard=hard)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(enc))
    return enc


def text_encoder(*a, **k):
    return sentence_encoder(**k) if not a else sentence_encoder(a[0], **k)


# ---------------------------------------------------------------- hypergraph packing
def pack_edges(fids, edges):
    """Index nodes of one jurisdiction subgraph and pack its pattern-typed hyperedges."""
    ix = {f: i for i, f in enumerate(fids)}
    el = []
    for e in edges:
        tl = [ix[v] for v in e.tail if v in ix]
        hd = [ix[v] for v in e.head if v in ix]
        if tl and hd:
            el.append((tl, hd, PATTERNS[e.pat]))
    return ix, Packed(len(fids), el, cap_set=6, cap_inc=10)


def iso_graph(repo_hg, iso, procs):
    """Fragments and edges of one jurisdiction (entry fragment plus its procedures)."""
    fids = [iso + ".ent"]
    edges = []
    for p in procs:
        fids += [f for f in p["fragments"]]
        edges += [repo_hg.edges[e] for e in p["edges"]]
    return fids, edges


# ---------------------------------------------------------------- training
def all_h0(tenc, m=None):
    """Semantic vectors of every fragment of the main dataset, in dataset order."""
    m = m or data.main_data()
    fids = list(m["fragments"])
    h0 = tenc.encode([m["fragments"][f]["text"] for f in fids]).astype(np.float32)
    return fids, h0


def proc_union(procs, hg, h0, row):
    """Disjoint union of procedure subgraphs, each with its own copy of the jurisdiction entry fragment."""
    nodes, edges, off = [], [], 0
    spans = []
    for p in procs:
        fids = [p["iso3"] + ".ent"] + list(p["fragments"])
        ix = {f: i for i, f in enumerate(fids)}
        for eid in p["edges"]:
            e = hg.edges[eid]
            tl = [off + ix[v] for v in e.tail if v in ix]
            hd = [off + ix[v] for v in e.head if v in ix]
            if tl and hd:
                edges.append((tl, hd, PATTERNS[e.pat]))
        nodes += [row[f] for f in fids]
        spans.append((off, len(fids)))
        off += len(fids)
    P = Packed(off, edges, cap_set=6, cap_inc=12)
    return P, h0[np.array(nodes)], spans


def train_allset(cfg, h0, row, procs, hg, log=print, seed=0, val_isos=()):
    """Masked hyperedge completion with InfoNCE over procedure subgraphs; returns the model and a report."""
    ec = cfg["encoder"]
    d = h0.shape[1]
    model = AllSet(d=d, dp=16, layers=ec["L"], seed=seed)
    rng = np.random.default_rng(seed)
    tr = [p for p in procs if p["iso3"] not in val_isos]
    va = [p for p in procs if p["iso3"] in val_isos]
    opt = nn.Adam(model.params(), lr=ec["lr"])
    tgt_w = nn.param((d, d), None, np.random.default_rng(seed + 1))
    opt.p.append(tgt_w)
    opt.m.append(np.zeros_like(tgt_w.d))
    opt.v.append(np.zeros_like(tgt_w.d))
    t0 = time.time()
    losses = []
    for step in range(ec["steps"]):
        sel = [tr[i] for i in rng.choice(len(tr), ec["procs_per_step"], replace=False)]
        P, x, spans = proc_union(sel, hg, h0, row)
        n = P.n
        m = rng.random(n) < ec["mask_rate"]
        for o, _ in spans:
            m[o] = False
        if not m.any():
            m[spans[0][0] + 1] = True
        h = model.forward(x, P, masked=m)
        z = nn.l2norm(model.zstr(h))
        zm = nn.gather(z, np.where(m)[0])
        tg = nn.l2norm(nn.matmul(nn.T(x[m]), tgt_w))
        logits = nn.scale(nn.matmul_t(zm, tg), 1.0 / ec["temp"])
        loss = nn.ce_logits(logits, np.arange(int(m.sum())))
        opt.zero()
        loss.backward()
        opt.step()
        losses.append(float(loss.d))
        if (step + 1) % 50 == 0:
            log("  allset step %d loss %.3f (%.0fs)" % (step + 1, np.mean(losses[-50:]), time.time() - t0))
    rep = {"loss_first": float(np.mean(losses[:20])), "loss_last": float(np.mean(losses[-20:])), "steps": len(losses), "curve": losses[::5]}
    rep.update(_hit_at_k(model, tgt_w, va, hg, h0, row, rng, ec))
    return model, rep


def _hit_at_k(model, tgt_w, va, hg, h0, row, rng, ec, k=(1, 5)):
    hits = {x: 0 for x in k}
    tot = 0
    for _ in range(12):
        sel = [va[i] for i in rng.choice(len(va), min(ec["procs_per_step"], len(va)), replace=False)]
        P, x, spans = proc_union(sel, hg, h0, row)
        n = P.n
        m = rng.random(n) < ec["mask_rate"]
        for o, _l in spans:
            m[o] = False
        if not m.any():
            continue
        with nn.no_grad():
            tg = nn.l2norm(nn.matmul(nn.T(x), tgt_w)).d
            z = nn.l2norm(model.zstr(model.forward(x, P, masked=m))).d
        sc = z[m] @ tg.T
        for r, i in enumerate(np.where(m)[0]):
            rank = int((sc[r] > sc[r, i]).sum())
            for q in k:
                hits[q] += rank < q
            tot += 1
    return {"hit@%d" % q: round(hits[q] / max(tot, 1), 4) for q in k} | {"val_masks": tot}


# ---------------------------------------------------------------- embedding
def fuse(zsem, zstr, alpha):
    """z = normalise(alpha * z_sem ++ (1 - alpha) * z_str); both parts are unit vectors of equal size."""
    a = np.concatenate([alpha * zsem, (1.0 - alpha) * zstr], axis=-1)
    return (a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)).astype(np.float32)


def project_sem(x, proj):
    """Project semantic vectors to the structural dimensionality."""
    y = x @ proj
    return y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)


def encode_graph(model, h0, P, chunk=4096):
    with nn.no_grad():
        z = model.zstr(model.forward(h0, P, chunk=chunk)).d
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9)


def embed_all(model, procs, hg, h0, row, chunk_procs=150):
    """Structural embeddings of every fragment, one disjoint union of procedure subgraphs at a time."""
    z = np.zeros((h0.shape[0], model.d), dtype=np.float32)
    seen = set()
    for s in range(0, len(procs), chunk_procs):
        sel = procs[s:s + chunk_procs]
        P, x, spans = proc_union(sel, hg, h0, row)
        zz = encode_graph(model, x, P)
        o = 0
        for p, (off, n) in zip(sel, spans):
            fids = [p["iso3"] + ".ent"] + list(p["fragments"])
            for i, f in enumerate(fids):
                if f.endswith(".ent") and f in seen:
                    continue
                z[row[f]] = zz[off + i]
                seen.add(f)
    return z


def save_model(model, f):
    f.parent.mkdir(parents=True, exist_ok=True)
    model.save(f)
