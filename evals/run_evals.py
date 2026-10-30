"""Evals: hit-rate on a synthetic workload, cost-saved estimate.

The workload is fixed, so the numbers are reproducible: 12 seed
questions, each asked four ways (the seed itself, an exact repeat, a
casing/spacing variant, and a near-duplicate with one word changed),
plus 6 unrelated questions that must miss. Expected shape: 54 lookups,
36 hits (24 exact, 12 semantic), 18 misses — hit-rate 0.667.

Run from the repo root: python evals/run_evals.py
"""

import os
import sys

# importing the service module builds its module-level app; keep that app
# in-memory so an eval run never leaves a cache.db in the repo
os.environ["CACHE_DB_PATH"] = ":memory:"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 12-word seeds on distinct topics, so a near-duplicate's best match is
# its own seed and never a neighbour — one word changed in twelve
# scores 11/12 ≈ 0.917, just over the 0.90 threshold, by design
SEED_PROMPTS = [
    "how do i reset my account password if i forgot it today",
    "what is the refund window for a damaged product return request",
    "how long does standard shipping usually take to arrive here",
    "can i change my billing address before my next payment cycle",
    "what is the warranty period for this electronic device purchase",
    "how do i cancel my subscription before the renewal date arrives",
    "where can i track my order after the shipment leaves warehouse",
    "how do i return an item that arrived broken or damaged today",
    "is there a student discount available for annual plan purchase",
    "how do i delete my account and all my personal data permanently",
    "can i update my payment method without contacting support staff",
    "when will my delivery arrive if i ordered it yesterday morning",
]

NOVEL_PROMPTS = [
    "explain quantum entanglement in simple everyday language please",
    "what is the capital city of australia and its population size",
    "how do i bake sourdough bread at home without a dutch oven",
    "translate good morning into japanese with polite formal phrasing",
    "what causes the northern lights to appear in the night sky",
    "how many minutes are there in a standard leap year exactly",
]


def build_workload():
    """Seed, repeat, variant, near-duplicate, then novels — in that order."""
    workload = []
    for seed in SEED_PROMPTS:
        workload.append(seed)
    for seed in SEED_PROMPTS:
        workload.append(seed)  # exact repeat — tier 1
    for seed in SEED_PROMPTS:
        workload.append("  " + seed.upper() + " ")  # variant — tier 1
    for seed in SEED_PROMPTS:
        words = seed.split()
        words[-1] = "now" if words[-1] != "now" else "today"
        workload.append(" ".join(words))  # one word changed — tier 2
    workload.extend(NOVEL_PROMPTS)  # unrelated — must miss
    return workload


def evaluate(cost_per_call=0.02, model_latency_ms=2000.0):
    """Run the workload through the real service and report the numbers."""
    from fastapi.testclient import TestClient

    from semantic_cache import CacheMetrics, ExactCache, SemanticCache
    from semantic_cache.service import create_app

    calls = {"n": 0}

    def mock_model(prompt):
        calls["n"] += 1
        return "model says: " + prompt

    app = create_app(ExactCache(), SemanticCache(), CacheMetrics(),
                     model_fn=mock_model, cost_per_call=cost_per_call,
                     model_latency_ms=model_latency_ms)
    client = TestClient(app)
    for prompt in build_workload():
        response = client.post("/lookup", json={"prompt": prompt})
        assert response.status_code == 200, response.text
    stats = client.get("/stats").json()
    stats["model_calls"] = calls["n"]
    return stats


def main():
    stats = evaluate()
    print("semantic-cache evals — synthetic workload")
    print(f"  lookups:            {stats['total_lookups']}")
    print(f"  exact hits:         {stats['exact_hits']}")
    print(f"  semantic hits:      {stats['semantic_hits']}")
    print(f"  misses:             {stats['misses']}")
    print(f"  hit rate:           {stats['hit_rate']:.3f}")
    print(f"  model calls:        {stats['model_calls']}")
    print(f"  latency saved (ms): {stats['latency_saved_ms']:.0f}")
    print(f"  cost saved (@$0.02/call): ${stats['estimated_cost_saved']:.2f}")
    if stats["model_calls"] != stats["misses"]:
        print("FAIL: model calls must equal misses")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
