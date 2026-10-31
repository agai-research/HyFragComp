"""Factory of the compared methods; all expose preload(records), serve(request) and handle(event)."""
from .system import Core, HyFragComp

NAMES = ("ours", "scratch", "whole", "nore", "cbr", "cafc")


class Scratch(Core):
    """No memory: every request goes through the shared scratch composer."""
    name = "scratch"

    def serve(self, req):
        from .util import Stats
        st = Stats()
        self.w.t = max(self.w.t, req.t)
        self.served += 1
        fr = self.front(req, st)
        if fr is None:
            return self.fail(req, st, "pii")
        graph, gv, _ = fr
        if not graph.goals:
            return self.fail(req, st, "no-goals", graph)
        out = self.scratch(req, graph, gv, st, st.pred)
        res = self.assemble(req, st, graph, out, "scratch", None, False, "scratch")
        res.stats = self.pack(st)
        return res

    def preload(self, recs, t=0.0):
        pass


def make(name, repo, cfg, models, seed=0):
    if name == "ours":
        return HyFragComp(repo, cfg, models, seed)
    if name == "whole":
        return HyFragComp(repo, cfg, models, seed, delta=False, single=True, name="whole")
    if name == "nore":
        return HyFragComp(repo, cfg, models, seed, revalidate=False, invalidate=False, name="nore")
    if name == "scratch":
        return Scratch(repo, cfg, models, seed)
    if name == "cbr":
        from Baselines.cbr import CBR
        return CBR(repo, cfg, models, seed)
    if name == "cafc":
        from Baselines.cafc import CAFC
        return CAFC(repo, cfg, models, seed)
    raise ValueError(name)
