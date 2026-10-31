# semantic-cache

A cache service for LLM responses. The same prompt — or one close enough
to mean the same thing — should not cost a second model call.

This is latency and cost engineering, the boring kind that shows up on
every senior job description: an LLM call that takes 2 seconds and costs
cents becomes a cache lookup that takes milliseconds and costs nothing.
It plugs into llm-service, which already serves mock-backed LLM calls —
the cache sits in front of the model call, not behind it.

## Architecture: two tiers, checked in order

Every lookup walks the tiers top to bottom and stops at the first hit.
A miss in both tiers calls the model, then stores the result so the next
identical (or near-identical) prompt is a hit.

```
prompt
  │
  ▼
┌──────────────────────┐
│ Tier 1 — exact       │  normalized prompt → SHA-256 key → stored response
│ (SQLite, day 2)      │  byte-identical after normalization, TTL enforced
└──────────┬───────────┘
           │ miss
           ▼
┌──────────────────────┐
│ Tier 2 — semantic    │  prompt embedding → cosine similarity over cached
│ (day 3)              │  prompts, hit iff similarity ≥ threshold
└──────────┬───────────┘
           │ miss
           ▼
      model call → store in both tiers → response
```

### Tier 1 — exact match

Normalization first, hashing second: strip surrounding whitespace,
collapse internal whitespace runs, and lowercase. Two prompts that
differ only in casing or spacing are the same prompt — caching the
uncleaned string would miss them for no reason. The key is
`SHA-256(normalized prompt)`, stored with the response, the model name,
and a stored-at timestamp. TTL (time to live) is enforced on read: an
expired entry is a miss, not a stale hit. Day 2 builds this tier on
SQLite — one file, no server, the cache survives restarts.

### Tier 2 — semantic match

Exact matching misses the case that actually matters: "What is the
refund window?" vs "How long do I have to return something?" Same
question, different words, full model price. The semantic tier embeds
the incoming prompt and compares it (cosine similarity) against the
embeddings of cached prompts. A similarity at or above the threshold
(default 0.90 — high on purpose, a wrong cached answer is worse than a
slow fresh one) is a hit and returns the cached response. Below the
threshold is a miss, even if it was close: near-misses are logged, not
served, so the threshold can be tuned from real data instead of vibes.

The embedding function is a seam, marked `# SWAP:` in the code like the
other projects do it: the default is a deterministic offline stand-in
(token-overlap based, good enough to develop and test against), and a
real embedding model (local sentence-transformers, or a hosted API
behind `EMBEDDING_API_KEY`) swaps in without touching the lookup logic.
Offline by default — a cache you can't test without the network is a
cache you won't test.

### What gets stored, and what doesn't

Each entry stores: the normalized prompt, the response, the model that
produced it, and timestamps. It does not store user identity, and it
never caches a response that came from a destructive or state-changing
call — this cache is for read-only generations (answers, summaries,
extractions). Anything with side effects bypasses the cache entirely.

## The week plan

- Day 1 (this commit): scaffold + this README — the architecture above.
- Day 2: exact-match cache (SQLite) with TTL.
- Day 3: semantic tier — embedding similarity lookup over cached prompts.
- Day 4: hit-rate / latency-saved metrics endpoint.
- Day 5: FastAPI service — `/lookup`, `/store`, `/stats`.
- Day 6: evals — hit-rate on a synthetic workload, cost-saved estimate.
- Day 7: Docker + CI + release notes.

## Non-goals

Not a distributed cache (no Redis cluster, no sharding — one service,
one SQLite file, day 2's simplicity is the point). Not a prompt router:
it never picks which model answers, it only remembers what a model
already said. Not a store for anything but LLM responses.

## Usage — tier 1 (day 2)

```python
from semantic_cache import ExactCache

cache = ExactCache(path="cache.db")  # :memory: by default
cache.store("What is the refund window?", "30 days", model="gemini")
cache.lookup("  WHAT   is the refund window? ")  # "30 days" — same key
cache.lookup("what is the warranty?")           # None — a miss
```

TTL defaults to 24 hours and is enforced on read: an expired entry is
a miss and its row is dropped, so a stale answer can never be served.
Pass `ttl=` per store to override, or `ttl=0` for an entry that expires
immediately. A `now_fn` argument on the constructor exists so tests can
advance a fake clock instead of sleeping through a real TTL.

## Usage — tier 2 (day 3)

```python
from semantic_cache import SemanticCache

sem = SemanticCache(path="cache.db")  # :memory: by default
sem.store("how do i reset my account password if i forgot it today")
sem.lookup("how do i reset my account password if i forgot it now")
# {"prompt": "how do i reset my account password if i forgot it today",
#  "similarity": 0.933} — a hit, one word changed in twelve
sem.lookup("what is the warranty policy here")  # None — too far to trust
```

`lookup` returns the best-matching cached prompt plus its cosine
similarity, or `None` below the threshold (default 0.90, overridable per
call). The default embedder is a deterministic word-hash stand-in, so
the tier catches paraphrases that share most of their words; pairs that
share meaning but not words ("refund window" vs "return something")
still miss until a real embedding model swaps in via the `# SWAP:`
seam — pass `embed_fn=` to the constructor and the lookup logic never
changes. Day 5's service wires the two tiers together: exact first,
semantic second, model call only on a miss in both.

## Usage — metrics (day 4)

```python
from semantic_cache import CacheMetrics, EXACT_HIT, MISS, SEMANTIC_HIT

metrics = CacheMetrics()
metrics.record(EXACT_HIT, latency_saved_ms=2000.0)  # a hit, 2s not spent
metrics.record(MISS)                                # a miss saves nothing
metrics.summary(cost_per_call=0.02)
# {"total_lookups": 2, "exact_hits": 1, ..., "hit_rate": 0.5,
#  "latency_saved_ms": 2000.0, "estimated_cost_saved": 0.02}
```

`CacheMetrics` only counts — callers record each lookup's outcome and,
on a hit, the milliseconds the model call would have taken minus the
lookup itself. Rates are 0.0 before any lookup (never a division error,
never a fake 100%), and without a `cost_per_call` the cost field stays
`None` rather than inventing a price. Day 5's service records outcomes
automatically and serves this summary at `/stats`.

## Usage — service (day 5)

```bash
pip install -r requirements.txt
uvicorn semantic_cache.service:app --port 8000
```

```bash
curl -X POST localhost:8000/store \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the refund window?", "response": "30 days"}'
curl -X POST localhost:8000/lookup \
  -H "Content-Type: application/json" \
  -d '{"prompt": "what is the refund window?"}'
# {"response": "30 days", "tier": "exact", "similarity": null}
curl localhost:8000/stats   # day-4 summary + live entry counts
```

`semantic_cache/service.py` wires the tiers in order (exact, then
semantic, then the model call on a miss — and a miss is stored in both
tiers, so the next identical prompt is an exact hit). Every lookup
records its outcome in `CacheMetrics`, so `/stats` is the live version
of the day-4 summary. The model is a mock by default (`model_fn=` swaps
in a real call); `CACHE_DB_PATH` picks the SQLite file (default
`cache.db`), `MODEL_LATENCY_MS`/`cost_per_call` feed the savings math.
One gotcha found while testing: FastAPI answers on worker threads, so
both tiers now open SQLite with `check_same_thread=False` — without it
the service crashed on the first `/store`.

## Evals (day 6)

```bash
python evals/run_evals.py
```

A fixed synthetic workload through the real service: 12 seed questions,
each asked four ways (the seed, an exact repeat, a casing/spacing
variant, a near-duplicate with one word changed), plus 6 unrelated
questions that must miss. Results, reproducible run after run:

| | |
|---|---|
| Lookups | 54 |
| Exact hits | 24 |
| Semantic hits | 12 |
| Misses (= model calls) | 18 |
| Hit rate | 0.667 |
| Latency saved | ~72,000 ms (at 2,000 ms per model call) |
| Cost saved | $0.72 (at $0.02 per model call) |

The numbers are honest about the stand-in embedder: near-duplicates hit
because they share 11 of 12 words, not because the embedder understands
meaning — true-synonym pairs still miss, which is exactly what day 3's
limitation note says. Swap in a real embedding model and the semantic
column is the one that should grow.

## Docker + CI (day 7)

```bash
docker build -t semantic-cache .
docker run -p 8000:8000 semantic-cache
# then: curl localhost:8000/health
```

`.github/workflows/ci.yml` gates every push/PR to main on
`tests/run_tests.py` and `evals/run_evals.py` — both offline and
deterministic, so a red build is a real regression. `RELEASES.md`
tracks each version. Caveat, stated plainly: no docker daemon exists
in the build sandbox, so `docker build` / `docker run` are unverified
here and stay a manual step on a docker machine; the image definition
and the CI workflow are what ship, and the CI steps themselves were
verified locally in order (tests, then evals).

## Current state

Version 0.7.0 — week 2 complete: tiers 1 and 2, metrics, service,
evals, Docker, CI, and release notes. `tests/run_tests.py` covers the
whole build (the 70 earlier checks plus workflow-parse, gated-steps,
Dockerfile-serve, and release-notes checks — all green offline).
