"""Offline self-test for semantic-cache.

Plain asserts, no pytest, no network. Run from the repo root:
    python tests/run_tests.py
Exit 0 when everything passes, 1 otherwise.
"""

import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import semantic_cache
from semantic_cache import (
    DEFAULT_THRESHOLD,
    CacheMetrics,
    EXACT_HIT,
    MISS,
    SEMANTIC_HIT,
    ExactCache,
    HashEmbedder,
    SemanticCache,
    cosine_similarity,
    key_for,
    normalize_prompt,
)

FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scaffold checks (day 1, kept green)
check("package imports", semantic_cache is not None)
check("version is 0.7.0", semantic_cache.__version__ == "0.7.0",
      semantic_cache.__version__)

with open(os.path.join(ROOT, "README.md")) as f:
    readme = f.read()
for section in ("Tier 1", "Tier 2", "Non-goals", "week plan"):
    check(f"README documents {section}", section in readme, section)

# normalization + keys (day 2)
check("normalize strips, collapses, lowercases",
      normalize_prompt("  What   IS\nthe refund\twindow? ") == "what is the refund window?")
check("casing/spacing variants share a key",
      key_for("What is the refund window?") == key_for("  what   IS the refund window? "))
check("key is sha256 of normalized prompt",
      key_for("hello") == hashlib.sha256(b"hello").hexdigest())
check("different prompts get different keys",
      key_for("hello") != key_for("goodbye"))
check("empty prompt has no key to store under",
      raises(lambda: ExactCache().store("   ", "x")))

# store / lookup roundtrip
cache = ExactCache()
check("miss on empty cache", cache.lookup("what is the refund window?") is None)
key = cache.store("What is the refund window?", "30 days", model="gemini")
check("store returns the key", key == key_for("what is the refund window?"))
check("lookup finds stored response",
      cache.lookup("what is the refund window?") == "30 days")
check("lookup is normalization-insensitive",
      cache.lookup("  WHAT   is the refund window? ") == "30 days")
check("unrelated prompt still misses",
      cache.lookup("what is the warranty?") is None)
entry = cache.lookup_entry("what is the refund window?")
check("entry carries model + normalized prompt",
      entry is not None and entry["model"] == "gemini"
      and entry["normalized_prompt"] == "what is the refund window?")
check("count counts live rows", cache.count() == 1)

# overwrite: same key, new response, not a second row
cache.store("what is the refund window?", "45 days", model="gemini")
check("overwrite replaces response",
      cache.lookup("what is the refund window?") == "45 days")
check("overwrite does not duplicate the row", cache.count() == 1)

# TTL: a fake clock beats sleeping through a real TTL in tests
clock = [1000.0]
ticking = ExactCache(now_fn=lambda: clock[0])
ticking.store("short-lived?", "yes", ttl=60)
check("fresh entry hits", ticking.lookup("short-lived?") == "yes")
clock[0] = 1059.0
check("entry still hits before expiry", ticking.lookup("short-lived?") == "yes")
clock[0] = 1060.0
check("expired entry is a miss, not a stale hit",
      ticking.lookup("short-lived?") is None)
check("expired entry is dropped on read", ticking.count() == 0)
ticking.store("instant?", "no", ttl=0)
check("ttl=0 expires immediately", ticking.lookup("instant?") is None)
ticking.store("kept?", "yes", ttl=3600)
clock[0] = 2000.0
check("count drops expired rows before counting", ticking.count() == 1)
ticking.clear()
check("clear empties the cache", ticking.count() == 0)

# persistence: a file cache survives close + reopen
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "cache.db")
    first = ExactCache(path=path)
    first.store("persist?", "still here", model="mock")
    first.close()
    second = ExactCache(path=path)
    check("file cache survives reopen",
          second.lookup("persist?") == "still here")
    check("reopened entry keeps its model",
          second.lookup_entry("persist?")["model"] == "mock")
    second.close()

# semantic tier (day 3)
check("default threshold is 0.90", DEFAULT_THRESHOLD == 0.90)

embedder = HashEmbedder()
vec_a = embedder.embed("what is the refund window?")
vec_b = embedder.embed("  WHAT   is the refund window? ")
check("identical prompts embed to cosine 1.0",
      abs(cosine_similarity(vec_a, vec_b) - 1.0) < 1e-9)
vec_c = embedder.embed("explain quantum entanglement simply")
check("unrelated prompts score near zero",
      cosine_similarity(vec_a, vec_c) < 0.5)
check("empty prompt embeds to the zero vector",
      all(v == 0.0 for v in embedder.embed("   ")))
check("cosine with a zero vector is 0.0",
      cosine_similarity(vec_a, embedder.embed("   ")) == 0.0)
check("embedder is deterministic across instances",
      HashEmbedder().embed("same prompt") == HashEmbedder().embed("same prompt"))

sem = SemanticCache()
check("empty prompt has no semantic entry to store",
      raises(lambda: sem.store("   ")))
check("miss on empty semantic cache",
      sem.lookup("what is the refund window?") is None)
sem.store("what is the refund window?")
hit = sem.lookup("what is the refund window?")
check("identical prompt is a semantic hit at 1.0",
      hit is not None and hit["prompt"] == "what is the refund window?"
      and abs(hit["similarity"] - 1.0) < 1e-9)
check("semantic lookup is normalization-insensitive",
      sem.lookup("  WHAT is the refund window? ") is not None)
check("unrelated prompt still misses the semantic tier",
      sem.lookup("explain quantum entanglement simply") is None)

# one word changed in a 12-word prompt: 14/15 ≈ 0.933, clears 0.90
sem.store("how do i reset my account password if i forgot it today")
hit = sem.lookup("how do i reset my account password if i forgot it now")
check("near-duplicate clears the 0.90 threshold",
      hit is not None and hit["similarity"] >= 0.90, str(hit))
# one word changed in a 6-word prompt: 5/6 ≈ 0.833, below 0.90
sem.store("what is the refund policy here")
check("one-word change in a short prompt misses at 0.90",
      sem.lookup("what is the warranty policy here") is None)
check("lower threshold rescues the short-prompt near-miss",
      sem.lookup("what is the warranty policy here", threshold=0.79) is not None)

# best match wins over a merely close one
sem2 = SemanticCache()
sem2.store("how do i cancel my subscription")
sem2.store("how do i change my billing address")
hit = sem2.lookup("how do i change my billing address today")
check("best match wins, not just any match above threshold",
      hit is not None and hit["prompt"] == "how do i change my billing address")
check("count counts semantic rows", sem2.count() == 2)
sem2.clear()
check("clear empties the semantic cache", sem2.count() == 0)

# persistence: embeddings survive close + reopen
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "sem.db")
    first = SemanticCache(path=path)
    first.store("how do i reset my account password if i forgot it today")
    first.close()
    second = SemanticCache(path=path)
    hit = second.lookup("how do i reset my account password if i forgot it now")
    check("semantic entries survive reopen",
          hit is not None and hit["similarity"] >= 0.90)
    second.close()

# metrics (day 4)
empty = CacheMetrics()
check("empty metrics report 0.0 rates, not a crash",
      empty.hit_rate == 0.0 and empty.miss_rate == 0.0 and empty.total == 0)
check("empty summary has no invented cost saving",
      empty.summary()["estimated_cost_saved"] is None)

m = CacheMetrics()
m.record(EXACT_HIT, latency_saved_ms=2000.0)
m.record(EXACT_HIT, latency_saved_ms=1500.0)
m.record(SEMANTIC_HIT, latency_saved_ms=1800.0)
m.record(MISS)
m.record(MISS, latency_saved_ms=9999.0)  # a miss saves nothing, by design
check("metrics count each outcome", m.total == 5 and m.hits == 3
      and m.exact_hits == 2 and m.semantic_hits == 1 and m.misses == 2)
check("hit rate is hits / total", abs(m.hit_rate - 0.6) < 1e-9)
check("per-tier rates split the total",
      abs(m.exact_hit_rate - 0.4) < 1e-9
      and abs(m.semantic_hit_rate - 0.2) < 1e-9
      and abs(m.miss_rate - 0.4) < 1e-9)
check("miss latency is not counted as saved",
      m.latency_saved_ms == 5300.0, str(m.latency_saved_ms))
summary = m.summary(cost_per_call=0.02)
check("summary cost saving is hits x cost per call",
      summary["estimated_cost_saved"] == 0.06, str(summary))
check("summary per-hit latency divides by hits only",
      abs(summary["latency_saved_per_hit_ms"] - 5300.0 / 3) < 1e-9)
check("unknown outcome fails loud",
      raises(lambda: m.record("maybe")))
m.reset()
check("reset zeroes every counter",
      m.total == 0 and m.latency_saved_ms == 0.0 and m.hit_rate == 0.0)

# FastAPI service (day 5) — skipped, not failed, where fastapi is absent
try:
    # importing the service module builds the module-level app; keep that
    # app in-memory so a test run never leaves a cache.db in the repo
    os.environ["CACHE_DB_PATH"] = ":memory:"
    from fastapi.testclient import TestClient
    from semantic_cache import ExactCache as _EC, SemanticCache as _SC
    from semantic_cache.service import create_app as _create_app

    _app = _create_app(_EC(), _SC(), CacheMetrics(),
                       model_fn=lambda p: "model says: " + p)
    _client = TestClient(_app)
    check("service health reports the package version",
          _client.get("/health").json()
          == {"status": "ok", "version": semantic_cache.__version__})
    check("empty prompt is a 400, not a stored blank",
          _client.post("/lookup", json={"prompt": "   "}).status_code == 400)
    r = _client.post("/store", json={"prompt": "What is the refund window?",
                                     "response": "30 days"})
    check("store reports the stored key", r.status_code == 200
          and r.json()["stored"] is True and len(r.json()["key"]) == 64)
    r = _client.post("/lookup", json={"prompt": "  WHAT is the refund window? "})
    check("exact tier serves a stored prompt via the service",
          r.json() == {"response": "30 days", "tier": "exact",
                       "similarity": None}, str(r.json()))
    r = _client.post("/lookup", json={
        "prompt": "how do i reset my account password if i forgot it today"})
    check("first unseen prompt is a miss and calls the model",
          r.json()["tier"] == "miss"
          and r.json()["response"].startswith("model says: "))
    r = _client.post("/lookup", json={
        "prompt": "how do i reset my account password if i forgot it now"})
    check("near-duplicate second prompt is a semantic hit",
          r.json()["tier"] == "semantic"
          and r.json()["similarity"] >= 0.90, str(r.json()))
    stats = _client.get("/stats").json()
    check("stats count the service lookups",
          stats["total_lookups"] == 3 and stats["exact_hits"] == 1
          and stats["semantic_hits"] == 1 and stats["misses"] == 1
          and stats["exact_entries"] == 2, str(stats))
    check("stats latency saved only counts hits",
          stats["latency_saved_ms"] > 0, str(stats))
except ImportError:
    print("SKIP service checks — fastapi/httpx not installed")

# evals (day 6) — same workload shape the script reports, via its code
try:
    sys.path.insert(0, os.path.join(ROOT, "evals"))
    import run_evals
    eval_stats = run_evals.evaluate()
    check("eval workload is 54 lookups", eval_stats["total_lookups"] == 54)
    check("eval hits split 24 exact / 12 semantic",
          eval_stats["exact_hits"] == 24 and eval_stats["semantic_hits"] == 12)
    check("eval misses are exactly the model calls",
          eval_stats["misses"] == 18
          and eval_stats["model_calls"] == eval_stats["misses"])
    check("eval hit rate is 2/3", abs(eval_stats["hit_rate"] - 2 / 3) < 1e-9)
    check("eval cost saved is hits x $0.02",
          abs(eval_stats["estimated_cost_saved"] - 0.72) < 1e-9)
except ImportError:
    print("SKIP eval checks — fastapi/httpx not installed")

# CI + Docker + release notes (day 7) — plain-text checks, so the
# workflow stays green without adding a yaml dependency to requirements
with open(os.path.join(ROOT, ".github/workflows/ci.yml")) as f:
    _ci_steps = f.read()
check("CI workflow gates on tests + evals",
      "tests/run_tests.py" in _ci_steps and "evals/run_evals.py" in _ci_steps)
check("CI workflow runs on push/PR to main",
      "push" in _ci_steps and "pull_request" in _ci_steps
      and "branches: [main]" in _ci_steps)
with open(os.path.join(ROOT, "Dockerfile")) as f:
    dockerfile = f.read()
check("Dockerfile serves the service on port 8000",
      "semantic_cache.service:app" in dockerfile and "EXPOSE 8000" in dockerfile)
with open(os.path.join(ROOT, "RELEASES.md")) as f:
    releases = f.read()
check("release notes cover 0.7.0", "0.7.0" in releases)

print(f"\n{len(FAILED)} failed" if FAILED else "\nall checks passed")
sys.exit(1 if FAILED else 0)
