"""FastAPI service: the two tiers + metrics behind HTTP.

POST /store   — put a prompt/response pair in both tiers
POST /lookup  — exact first, semantic second, model call only on a miss
GET  /stats   — the day-4 metrics summary, plus live entry counts
GET  /health  — liveness for Render and smoke tests

The model call is a seam, like everything else here: by default a
deterministic mock (no keys, no network), swapped for a real call by
passing model_fn= to create_app(). The service never caches a miss it
didn't store — a lookup that misses stores what the model said, so the
next identical prompt is an exact hit.
"""

import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .exact import ExactCache
from .metrics import EXACT_HIT, MISS, SEMANTIC_HIT, CacheMetrics
from .semantic import SemanticCache

# what a model call costs in the mock world — a hit saves roughly this
DEFAULT_MODEL_LATENCY_MS = 2000.0


def _mock_model(prompt):
    # SWAP: pass a real model call as model_fn= — the tiers never change
    return f"[mock] answer to: {prompt}"


class StoreRequest(BaseModel):
    prompt: str
    response: str
    model: str = "unknown"


class LookupRequest(BaseModel):
    prompt: str


def create_app(exact=None, semantic=None, metrics=None, model_fn=None,
               cost_per_call=None, model_latency_ms=DEFAULT_MODEL_LATENCY_MS):
    exact = exact or ExactCache(path=os.environ.get("CACHE_DB_PATH", "cache.db"))
    semantic = semantic or SemanticCache(
        path=os.environ.get("CACHE_DB_PATH", "cache.db"))
    metrics = metrics or CacheMetrics()
    model_fn = model_fn or _mock_model

    app = FastAPI(title="semantic-cache")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.post("/store")
    def store(req: StoreRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is empty")
        key = exact.store(req.prompt, req.response, model=req.model)
        semantic.store(req.prompt)
        return {"stored": True, "key": key}

    @app.post("/lookup")
    def lookup(req: LookupRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is empty")
        started = time.perf_counter()
        # tier 1: byte-identical after normalization
        response = exact.lookup(req.prompt)
        if response is not None:
            metrics.record(EXACT_HIT, latency_saved_ms=_saved(started, model_latency_ms))
            return {"response": response, "tier": "exact", "similarity": None}
        # tier 2: close enough to trust, at or above the threshold
        match = semantic.lookup(req.prompt)
        if match is not None:
            response = exact.lookup(match["prompt"])
            if response is not None:
                metrics.record(SEMANTIC_HIT,
                               latency_saved_ms=_saved(started, model_latency_ms))
                return {"response": response, "tier": "semantic",
                        "similarity": match["similarity"]}
        # miss in both: pay for the model call once, then remember it
        response = model_fn(req.prompt)
        exact.store(req.prompt, response)
        semantic.store(req.prompt)
        metrics.record(MISS)
        return {"response": response, "tier": "miss", "similarity": None}

    @app.get("/stats")
    def stats():
        summary = metrics.summary(cost_per_call=cost_per_call)
        summary["exact_entries"] = exact.count()
        summary["semantic_entries"] = semantic.count()
        return summary

    return app


def _saved(started, model_latency_ms):
    # a hit saves the model call minus the milliseconds the lookup took
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return max(0.0, model_latency_ms - elapsed_ms)


app = create_app()
