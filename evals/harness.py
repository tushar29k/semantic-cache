"""Shared harness: run adversarial cases through the real service.

Every case runs against a fresh app (fresh in-memory tiers), so cases
cannot interfere. Writes go through the public API only — POST /store
for legitimate and planted entries, POST /lookup for queries — and a
"miss plant" first asks the attacker's question with a rigged model
answer, which the service then caches exactly as it caches any miss.

A miss is unmistakable: the default model answer carries FRESH_PREFIX,
which no stored response uses. CACHE_DB_PATH is pinned to :memory:
before the service module is imported (its module-level app would
otherwise build a cache.db in the repo), same as evals/run_evals.py.
"""

import json
import os
import sys

os.environ["CACHE_DB_PATH"] = ":memory:"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRESH_PREFIX = "FRESH MODEL ANSWER: "


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def make_client(model_answers=None, threshold=None):
    from fastapi.testclient import TestClient

    from semantic_cache import CacheMetrics, ExactCache, SemanticCache
    from semantic_cache.service import create_app

    def model_fn(prompt):
        if model_answers and prompt in model_answers:
            return model_answers[prompt]
        return FRESH_PREFIX + prompt

    semantic = SemanticCache() if threshold is None \
        else SemanticCache(threshold=threshold)
    app = create_app(ExactCache(), semantic, CacheMetrics(),
                     model_fn=model_fn)
    return TestClient(app)


def _lookup(client, prompt):
    resp = client.post("/lookup", json={"prompt": prompt})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    served = None if body["tier"] == "miss" else body["response"]
    return {"served": served, "tier": body["tier"],
            "similarity": body["similarity"]}


def run_attack_case(case, threshold=None):
    """One attack case -> [(query, outcome)] with one entry per query."""
    model_answers = {mp["prompt"]: mp["model_answer"]
                     for mp in case.get("miss_plants", [])}
    client = make_client(model_answers=model_answers, threshold=threshold)
    for entry in case.get("stores", []) + case.get("plants", []):
        resp = client.post("/store", json={
            "prompt": entry["prompt"], "response": entry["response"]})
        assert resp.status_code == 200, resp.text
    for mp in case.get("miss_plants", []):
        out = _lookup(client, mp["prompt"])
        assert out["tier"] == "miss", f"miss plant did not miss: {out}"
    results = []
    for query in case["queries"]:
        out = _lookup(client, query["prompt"])
        outcome = {"served": out["served"], "tier": out["tier"],
                   "similarity": out["similarity"],
                   "poison_response": query.get("poison_response"),
                   "legit_response": query.get("legit_response")}
        results.append((query, outcome))
    return results


def run_control_case(case, threshold=None):
    """One benign control case -> (query, outcome)."""
    client = make_client(threshold=threshold)
    for entry in case.get("stores", []):
        resp = client.post("/store", json={
            "prompt": entry["prompt"], "response": entry["response"]})
        assert resp.status_code == 200, resp.text
    query = case["query"]
    out = _lookup(client, query["prompt"])
    return query, {"served": out["served"], "tier": out["tier"],
                   "similarity": out["similarity"]}


def run_paws_pair(pair, threshold=None):
    """One PAWS-style pair -> True when the cache predicted paraphrase
    (a hit), with the similarity the semantic tier reported."""
    client = make_client(threshold=threshold)
    resp = client.post("/store", json={
        "prompt": pair["sentence1"], "response": "stored answer"})
    assert resp.status_code == 200, resp.text
    out = _lookup(client, pair["sentence2"])
    return out["served"] is not None, out
