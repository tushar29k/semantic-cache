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
from semantic_cache import ExactCache, key_for, normalize_prompt

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
check("version is 0.2.0", semantic_cache.__version__ == "0.2.0",
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

print(f"\n{len(FAILED)} failed" if FAILED else "\nall checks passed")
sys.exit(1 if FAILED else 0)
