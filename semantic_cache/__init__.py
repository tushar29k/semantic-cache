"""semantic-cache: exact + semantic caching for LLM responses.

Day 1 is the scaffold — the two tiers live here as the week builds them:
exact match (SQLite, day 2) checked first, semantic similarity (day 3)
second, model call only on a miss in both.
"""

__version__ = "0.1.0"
