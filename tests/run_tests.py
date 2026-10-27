"""Offline self-test for the scaffold.

Plain asserts, no pytest, no network. Run from the repo root:
    python tests/run_tests.py
Exit 0 when everything passes, 1 otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import semantic_cache

FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

check("package imports", semantic_cache is not None)
check("version is 0.1.0", semantic_cache.__version__ == "0.1.0",
      semantic_cache.__version__)

with open(os.path.join(ROOT, "README.md")) as f:
    readme = f.read()
for section in ("Tier 1", "Tier 2", "Non-goals", "week plan"):
    check(f"README documents {section}", section in readme, section)

print(f"\n{len(FAILED)} failed" if FAILED else "\nall checks passed")
sys.exit(1 if FAILED else 0)
