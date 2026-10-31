"""Client for the hosted language model (OpenAI GPT-5.6 Luna by default).

Every call is logged to results/logs/llm_calls.jsonl with its role, token counts and measured latency, and
cached in results/llm_cache so that a rerun with the same prompts returns the same answers and the same
recorded latencies (set llm.api.cache: false to disable). The client uses the Chat Completions endpoint in
JSON mode through the standard library only.

Configuration (configs/default.yaml, section llm.api): model, base_url, timeout_s, max_retries, cache,
max_output_tokens. The key is read from the environment variable OPENAI_API_KEY.

HFC_LLM_MOCK=1 replaces the network call by a local echo of the rule answer with a synthetic latency; it is
used only by Test/test_llm_backend.py to exercise the code paths without network access.
"""
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from .util import ntok, path


class LLMError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, cfg):
        api = cfg["llm"].get("api", {})
        self.model = api.get("model", "gpt-5.6-luna")
        self.base_url = api.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.timeout = float(api.get("timeout_s", 60))
        self.retries = int(api.get("max_retries", 4))
        self.max_out = int(api.get("max_output_tokens", 1024))
        self.use_cache = bool(api.get("cache", True))
        self.mock = os.environ.get("HFC_LLM_MOCK") == "1"
        self.key = os.environ.get("OPENAI_API_KEY", "")
        if not self.key and not self.mock:
            raise LLMError("llm.backend requires the OPENAI_API_KEY environment variable")
        self.cache_dir = path("results", "llm_cache")
        self.log_file = path("results", "logs", "llm_calls.jsonl")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.context = {}  # set by the caller: request id, method, backend

    # ------------------------------------------------------------------ public
    def complete(self, role, system, user, decoding, mock_answer=None):
        """Return (parsed JSON object or None, raw text, tokens in, tokens out, measured ms, cached)."""
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "max_completion_tokens": self.max_out}
        if "temperature" in decoding:
            body["temperature"] = decoding["temperature"]
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cf = self.cache_dir / (key + ".json")
        cached = False
        if self.use_cache and cf.exists():
            rec = json.loads(cf.read_text())
            cached = True
        else:
            rec = self._mock(user, mock_answer) if self.mock else self._post(body)
            if self.use_cache:
                cf.write_text(json.dumps(rec))
        obj = None
        try:
            obj = json.loads(rec["text"])
        except (ValueError, TypeError):
            obj = None
        self._log(role, rec, cached)
        return obj, rec["text"], rec["tok_in"], rec["tok_out"], rec["ms"], cached

    # ------------------------------------------------------------------ transport
    def _post(self, body):
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base_url + "/chat/completions", data=data, method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode())
                ms = 1000.0 * (time.perf_counter() - t0)
                use = payload.get("usage", {})
                text = payload["choices"][0]["message"]["content"] or ""
                return {"text": text, "tok_in": int(use.get("prompt_tokens", 0)), "tok_out": int(use.get("completion_tokens", 0)),
                        "ms": ms, "model": payload.get("model", self.model), "t": time.time()}
            except urllib.error.HTTPError as e:
                last = "HTTP %d: %s" % (e.code, e.read()[:300])
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = repr(e)
            time.sleep(min(30.0, 2 ** attempt + random.random()))
        raise LLMError("API call failed after %d attempts: %s" % (self.retries + 1, last))

    def _mock(self, user, mock_answer):
        text = json.dumps(mock_answer if mock_answer is not None else {}, default=list)
        tin, tout = ntok(user) + 40, ntok(text)
        return {"text": text, "tok_in": tin, "tok_out": tout, "ms": 300.0 + tin / 3.0 + tout * 20.0, "model": "mock", "t": time.time()}

    def _log(self, role, rec, cached):
        line = {"role": role, "model": rec.get("model"), "tok_in": rec["tok_in"], "tok_out": rec["tok_out"],
                "ms": round(rec["ms"], 2), "cached": cached, **self.context}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(line) + "\n")
