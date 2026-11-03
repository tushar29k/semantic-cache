"""Threshold probe: the similarity ladder and the threshold sweep.

Two measurements the day-6 hit rate cannot make:

1. The ladder. For stored prompts of 6, 12 and 20 tokens, queries
   sharing exactly k of the stored tokens (the rest replaced from a
   disjoint vocabulary, so only overlap moves). This shows which
   similarity values are even attainable: with integer token counts
   the cosine takes discrete steps, so whole intervals of the
   threshold behave identically — tuning inside a step does nothing.
   An append ladder (stored prompt + m arbitrary tokens) shows how
   many unexamined tokens a query may carry and still hit.

2. The sweep. The seed attacks and benign controls replayed through
   the real service at thresholds 0.80-0.99. ASR, false-hit rate and
   benign completion per threshold: the trade-off curve, measured
   instead of argued.

Similarities below the default threshold are read with a per-call
threshold override of 0.0 (measurement only; hit/miss is always
judged at the threshold under test). Report-only, deterministic.

Run from the repo root: python evals/run_threshold_probe.py
"""

import os
import sys

os.environ["CACHE_DB_PATH"] = ":memory:"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import scoring  # noqa: E402
from semantic_cache import SemanticCache  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

CONTENT = ["account", "billing", "refund", "window", "order", "tracking",
           "delivery", "warranty", "password", "reset", "cancel", "renewal",
           "shipping", "address", "payment", "invoice", "support", "ticket",
           "plan", "discount"]
FILLER = ["zebra", "quasar", "melon", "cobalt", "velvet", "harbor",
          "cinder", "topaz", "willow", "falcon", "marble", "saffron",
          "tundra", "onyx", "basil", "comet", "dune", "ember",
          "fjord", "garnet"]


def _raw_similarity(cache, prompt):
    hit = cache.lookup(prompt, threshold=0.0)
    return 0.0 if hit is None else hit["similarity"]


def overlap_ladder(length):
    """[(shared_tokens, similarity, hit_at_default)] for k = L..0."""
    stored = " ".join(CONTENT[:length])
    cache = SemanticCache()
    cache.store(stored)
    rows = []
    for k in range(length, -1, -1):
        query = " ".join(CONTENT[:k] + FILLER[:length - k])
        sim = _raw_similarity(cache, query)
        rows.append((k, sim, sim >= cache.threshold))
    cache.close()
    return rows


def append_ladder():
    """[(extra_tokens, similarity, hit_at_default)] for m = 0..6."""
    stored = " ".join(CONTENT[:12])
    cache = SemanticCache()
    cache.store(stored)
    rows = []
    for m in range(0, 7):
        query = stored + (" " + " ".join(FILLER[:m]) if m else "")
        sim = _raw_similarity(cache, query)
        rows.append((m, sim, sim >= cache.threshold))
    cache.close()
    return rows


def sweep(thresholds=(0.80, 0.85, 0.90, 0.95, 0.99)):
    """Replay seeds + controls at each threshold -> [metrics rows]."""
    seeds = harness.load_jsonl(os.path.join(HERE, "adversarial_seeds.jsonl"))
    controls = harness.load_jsonl(os.path.join(HERE, "benign_controls.jsonl"))
    rows = []
    for t in thresholds:
        attacks, benign = [], []
        for case in seeds:
            for query, outcome in harness.run_attack_case(case, threshold=t):
                attacks.append((query, outcome,
                                scoring.classify_attack(outcome)))
        for case in controls:
            query, outcome = harness.run_control_case(case, threshold=t)
            benign.append((case["query"], outcome,
                           scoring.classify_benign(case["query"], outcome)))
        m = scoring.aggregate(attacks, benign)
        rows.append((t, m))
    return rows


def main():
    for length in (6, 12, 20):
        print(f"overlap ladder — stored prompt of {length} tokens "
              f"(threshold 0.90):")
        for k, sim, hit in overlap_ladder(length):
            print(f"  shared {k:2d}/{length}: similarity {sim:.4f} "
                  f"{'HIT' if hit else 'miss'}")
    print("append ladder — 12-token stored prompt + m arbitrary tokens:")
    for m, sim, hit in append_ladder():
        print(f"  +{m} tokens: similarity {sim:.4f} {'HIT' if hit else 'miss'}")
    print("threshold sweep — seed attacks + benign controls:")
    print("  threshold | ASR | block rate | false-hit rate | completion")
    for t, m in sweep():
        a, b = m["attacks"], m["benign"]
        print(f"  {t:.2f}      | {scoring.fmt_rate(a['asr'], a['n'])} | "
              f"{scoring.fmt_rate(a['block_rate'], a['n'])} | "
              f"{scoring.fmt_rate(b['fhr'], b['must_miss_n'])} | "
              f"{scoring.fmt_rate(b['completion'], b['n'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
