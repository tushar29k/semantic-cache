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
check("version is 0.3.0", semantic_cache.__version__ == "0.3.0",
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

print(f"\n{len(FAILED)} failed" if FAILED else "\nall checks passed")
sys.exit(1 if FAILED else 0)
