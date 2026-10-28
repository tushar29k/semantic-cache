"""semantic-cache: exact + semantic caching for LLM responses.

Two tiers checked in order: exact match (SQLite, day 2) first, semantic
similarity (day 3) second, model call only on a miss in both.
"""

from .exact import ExactCache, key_for, normalize_prompt

__version__ = "0.2.0"

__all__ = ["ExactCache", "key_for", "normalize_prompt"]
