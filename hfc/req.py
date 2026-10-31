"""Request, goal and result structures shared by all methods."""
from dataclasses import dataclass, field
from typing import Optional

from .model import Ctx, Goal, GoalGraph


@dataclass
class Request:
    """A request (P, D, t, u) with the construction-time truth kept for the oracle only."""
    rid: str
    text: str
    t: int
    user: str
    ctx: Ctx
    truth: dict = field(default_factory=dict)


@dataclass
class Result:
    """Outcome of serving one request with one method."""
    rid: str
    mode: str = "scratch"
    ok: bool = False
    frags: tuple = ()
    conns: tuple = ()
    cover: dict = field(default_factory=dict)      # gid -> tuple of fids
    status: dict = field(default_factory=dict)     # gid -> reuse | switch | new
    graph: Optional[GoalGraph] = None
    rho: float = 0.0
    delta: int = 0
    n_goals: int = 0
    junctions: tuple = ()
    witnesses: dict = field(default_factory=dict)
    explanation: str = ""
    repairs: int = 0
    stats: dict = field(default_factory=dict)
    align: dict = field(default_factory=dict)      # request gid -> (cid, gid') produced by the alignment
    info: dict = field(default_factory=dict)


def ctx_from(d):
    return Ctx(frozenset(d["atoms"]), frozenset(d["data"]), frozenset(d["roles"]), int(d["deadline"]), d["jur"])


def request_from(d):
    """Rebuild a Request from its JSON form."""
    return Request(d["rid"], d["text"], d["t"], d["user"], ctx_from(d["ctx"]), d.get("truth", {}))
