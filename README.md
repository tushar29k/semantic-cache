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

## Current state

Version 0.2.0 — tier 1 done. `semantic_cache/exact.py` holds the tier
(`normalize_prompt`, `key_for`, `ExactCache` with store/lookup/lookup_entry/
count/clear, SQLite, TTL on read); `tests/run_tests.py` covers it
(29 checks: normalization, SHA-256 keys, roundtrip, overwrite, TTL
expiry via a fake clock, persistence across reopen — all green offline,
standard library only). Tier 2 (semantic similarity) lands on day 3.
