"""Shared helpers: paths, config, timers, logging, counters."""
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def path(*parts):
    return ROOT.joinpath(*parts)


def load_cfg(name="default", overrides=None):
    cfg = yaml.safe_load(path("configs", "default.yaml").read_text())
    if name != "default":
        extra = yaml.safe_load(path("configs", f"{name}.yaml").read_text()) or {}
        deep_update(cfg, extra)
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def get_logger(name, logfile=None):
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fmt = logging.Formatter("%(message)s")
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            h = logging.FileHandler(logfile, mode="a")
            h.setFormatter(fmt)
            lg.addHandler(h)
    return lg


def jlog(lg, **kw):
    lg.info(json.dumps(kw, default=str))


class Stats:
    """Per-request counters: stage timers, LLM calls/tokens, predicate evaluations."""

    def __init__(self):
        self.t = defaultdict(float)
        self.calls = defaultdict(int)
        self.tok_in = defaultdict(int)
        self.tok_out = defaultdict(int)
        self.pred = defaultdict(int)
        self.llm_ms = defaultdict(float)
        self.extra = {}

    def tic(self, stage):
        return (stage, time.perf_counter())

    def toc(self, h):
        self.t[h[0]] += time.perf_counter() - h[1]

    def llm(self, role, nin, nout, ms=0.0):
        self.calls[role] += 1
        self.tok_in[role] += nin
        self.tok_out[role] += nout
        self.llm_ms[role] += ms

    def total_calls(self):
        return sum(self.calls.values())

    def to_dict(self):
        return {"t": dict(self.t), "calls": dict(self.calls), "tok_in": dict(self.tok_in),
                "tok_out": dict(self.tok_out), "pred": dict(self.pred), "llm_ms": dict(self.llm_ms), **self.extra}

    def total_llm_ms(self):
        return sum(self.llm_ms.values())

    def total_ms(self):
        """measured compute plus modelled LLM latency, in milliseconds"""
        return 1000.0 * sum(self.t.values()) + self.total_llm_ms()


def ntok(text):
    return max(1, (len(text) + 3) // 4)


def ensure_hash_seed():
    """Python randomises the iteration order of sets of strings per process; results are reproducible only with a
    fixed PYTHONHASHSEED. Re-execute the current script with PYTHONHASHSEED=0 when it is not set."""
    import os
    import sys
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
