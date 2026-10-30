"""semantic-cache: exact + semantic caching for LLM responses.

Two tiers checked in order: exact match (SQLite, day 2) first, semantic
similarity (day 3) second, model call only on a miss in both.
"""

from .exact import ExactCache, key_for, normalize_prompt
from .metrics import EXACT_HIT, MISS, SEMANTIC_HIT, CacheMetrics
from .semantic import (
    DEFAULT_THRESHOLD,
    HashEmbedder,
    SemanticCache,
    cosine_similarity,
)

__version__ = "0.4.0"

__all__ = [
    "CacheMetrics",
    "EXACT_HIT",
    "MISS",
    "SEMANTIC_HIT",
    "ExactCache",
    "HashEmbedder",
    "SemanticCache",
    "cosine_similarity",
    "DEFAULT_THRESHOLD",
    "key_for",
    "normalize_prompt",
]
