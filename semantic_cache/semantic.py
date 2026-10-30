"""Tier 2 — semantic match over cached prompts.

Exact matching misses paraphrases; this tier catches them by embedding
similarity. A stored prompt whose cosine similarity to the incoming
prompt clears the threshold is a hit — same idea, different wording,
no second model call. The threshold stays high on purpose: a wrong
cached answer costs more than a slow fresh one.
"""

import hashlib
import json
import math
import re
import sqlite3

from .exact import key_for, normalize_prompt

# hit iff similarity >= this — a wrong cached answer is worse than a miss
DEFAULT_THRESHOLD = 0.90

# small on purpose: the stand-in embedder is for dev and tests, not production
EMBEDDING_DIM = 256

_WORDS = re.compile(r"[a-z0-9]+")


def _tokens(prompt):
    # normalized first, so casing and punctuation never change the vector
    return _WORDS.findall(normalize_prompt(prompt))


class HashEmbedder:
    """Deterministic offline stand-in for a real embedding model.

    Each word hashes into one dimension; shared words push two prompts
    together, different words push them apart. Good enough to build and
    test the lookup logic against — it scores word overlap, not meaning,
    so true synonyms ("refund" vs "return") still miss until a real
    model swaps in.
    """

    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim

    def embed(self, prompt):
        vec = [0.0] * self.dim
        for word in _tokens(prompt):
            # stable across runs — hash(), unlike sha256, is salted per process
            slot = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine_similarity(a, b):
    """Dot product of two vectors; 0.0 when either side is empty."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """Cached prompts + their embeddings, one SQLite table, no server."""

    def __init__(self, path=":memory:", threshold=DEFAULT_THRESHOLD, embed_fn=None):
        # SWAP: pass a real embedding model here (sentence-transformers, or
        # a hosted API behind an env key) — lookup logic never changes
        self._embed = embed_fn or HashEmbedder().embed
        self.threshold = threshold
        # same reason as tier 1: the service answers on worker threads
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS semantic_entries ("
            "  key TEXT PRIMARY KEY,"
            "  normalized_prompt TEXT NOT NULL,"
            "  embedding TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def store(self, prompt):
        """Embed a prompt for later lookup. Returns its key."""
        normalized = normalize_prompt(prompt)
        if not normalized:
            raise ValueError("cannot cache an empty prompt")
        self._conn.execute(
            "INSERT OR REPLACE INTO semantic_entries"
            " (key, normalized_prompt, embedding) VALUES (?, ?, ?)",
            (key_for(prompt), normalized,
             json.dumps(self._embed(prompt))),
        )
        self._conn.commit()
        return key_for(prompt)

    def lookup(self, prompt, threshold=None):
        """Best-matching cached prompt, or None — the only thing callers need.

        Returns {"prompt": ..., "similarity": ...} for the closest stored
        prompt at or above the threshold; ties go to the earliest stored.
        """
        if not normalize_prompt(prompt):
            return None
        threshold = self.threshold if threshold is None else threshold
        query_vec = self._embed(prompt)
        best = None
        best_score = -1.0
        rows = self._conn.execute(
            "SELECT normalized_prompt, embedding FROM semantic_entries"
            " ORDER BY rowid"
        ).fetchall()
        for stored_prompt, embedding_json in rows:
            score = cosine_similarity(query_vec, json.loads(embedding_json))
            if score > best_score:
                best_score = score
                best = stored_prompt
        if best is None or best_score < threshold:
            return None
        return {"prompt": best, "similarity": best_score}

    def count(self):
        return self._conn.execute(
            "SELECT COUNT(*) FROM semantic_entries").fetchone()[0]

    def clear(self):
        """Empty the cache. Tests and fresh starts, not a daily operation."""
        self._conn.execute("DELETE FROM semantic_entries")
        self._conn.commit()

    def close(self):
        self._conn.close()
