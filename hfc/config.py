"""Configuration loading with dotted-key overrides."""
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


class Cfg(dict):
    # dict with attribute access for nested sections
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def set(self, dotted, value):
        cur = self
        parts = dotted.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
        return self

    def get_path(self, dotted, default=None):
        cur = self
        for p in dotted.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def copy_with(self, **dotted):
        c = Cfg(copy.deepcopy(dict(self)))
        for k, v in dotted.items():
            c.set(k.replace("__", "."), v)
        return c


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load(*overrides):
    base = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    for o in overrides:
        p = Path(o)
        if not p.is_absolute():
            p = ROOT / "configs" / o
        _merge(base, yaml.safe_load(p.read_text()) or {})
    return Cfg(base)
