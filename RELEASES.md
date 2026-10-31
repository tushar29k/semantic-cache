# Releases

## 0.7.0 — week 2 complete (2026-10-31)

Docker + CI + release notes, closing the semantic-cache week.

- `Dockerfile` serves the day-5 FastAPI service on port 8000
  (`uvicorn semantic_cache.service:app`); `.dockerignore` keeps
  `*.db` and test files out of the image.
- `.github/workflows/ci.yml` gates every push/PR to main on
  `tests/run_tests.py` and `evals/run_evals.py`, both offline and
  deterministic — a red build means a real regression, not flakes.
- Honest caveat: no docker daemon exists in the build sandbox, so
  `docker build` / `docker run` are unverified here and stay a manual
  step on a docker machine; the image definition itself is the
  shipped increment, and CI (the part that runs on every push) is
  verified locally in order below.

Eval numbers, unchanged since day 6: 54-lookup synthetic workload,
24 exact hits, 12 semantic hits, 18 misses, hit rate 0.667,
~72,000 ms latency saved, $0.72 saved at $0.02 per call.

## Build-up

- 0.1.0 — scaffold + README, two-tier architecture contract
- 0.2.0 — exact-match SQLite cache with TTL
- 0.3.0 — semantic tier: embedding similarity lookup at 0.90
- 0.4.0 — hit-rate / latency-saved metrics
- 0.5.0 — FastAPI service: /lookup, /store, /stats, /health
- 0.6.0 — evals: hit-rate on a synthetic workload, cost-saved estimate
- 0.7.0 — Docker + CI + release notes (this release)
