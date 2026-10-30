"""Tier 1 — exact-match cache on SQLite.

Normalization first, hashing second: two prompts that differ only in
casing or spacing are the same prompt, so they share one key. TTL is
enforced on read — an expired entry is a miss, never a stale hit.
"""

import hashlib
import re
import sqlite3
import time

# a day: long enough that a cached answer stays useful, short enough
# that a stale one doesn't linger forever
DEFAULT_TTL_SECONDS = 86400

_WHITESPACE = re.compile(r"\s+")


def normalize_prompt(prompt):
    """Strip, collapse whitespace runs, lowercase — the canonical form."""
    return _WHITESPACE.sub(" ", prompt.strip()).lower()


def key_for(prompt):
    """SHA-256 of the normalized prompt, hex — the row's primary key."""
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


class ExactCache:
    """One SQLite file, one table, no server — the cache survives restarts."""

    def __init__(self, path=":memory:", default_ttl=DEFAULT_TTL_SECONDS,
                 now_fn=None):
        # now_fn is a seam for tests: advance a fake clock instead of sleeping
        self._now = now_fn or time.time
        # the service answers on worker threads — one connection must serve them
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "  key TEXT PRIMARY KEY,"
            "  normalized_prompt TEXT NOT NULL,"
            "  response TEXT NOT NULL,"
            "  model TEXT NOT NULL,"
            "  stored_at REAL NOT NULL,"
            "  expires_at REAL"
            ")"
        )
        self._conn.commit()
        self.default_ttl = default_ttl

    def store(self, prompt, response, model="unknown", ttl=None):
        """Store (or overwrite) the response for a prompt. Returns its key."""
        normalized = normalize_prompt(prompt)
        if not normalized:
            raise ValueError("cannot cache an empty prompt")
        # ttl=None means "use the default"; ttl=0 expires on the next read
        ttl = self.default_ttl if ttl is None else ttl
        now = self._now()
        expires_at = None if ttl is None else now + ttl
        self._conn.execute(
            "INSERT OR REPLACE INTO entries"
            " (key, normalized_prompt, response, model, stored_at, expires_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (key_for(prompt), normalized, response, model, now, expires_at),
        )
        self._conn.commit()
        return key_for(prompt)

    def lookup_entry(self, prompt):
        """Full stored row, or None on a miss or an expired entry."""
        row = self._conn.execute(
            "SELECT key, normalized_prompt, response, model, stored_at,"
            " expires_at FROM entries WHERE key = ?",
            (key_for(prompt),),
        ).fetchone()
        if row is None:
            return None
        entry = {
            "key": row[0],
            "normalized_prompt": row[1],
            "response": row[2],
            "model": row[3],
            "stored_at": row[4],
            "expires_at": row[5],
        }
        if entry["expires_at"] is not None and entry["expires_at"] <= self._now():
            # expired is a miss, and a dead row is just clutter — drop it
            self._conn.execute("DELETE FROM entries WHERE key = ?", (entry["key"],))
            self._conn.commit()
            return None
        return entry

    def lookup(self, prompt):
        """The cached response, or None — the only thing callers need."""
        entry = self.lookup_entry(prompt)
        return None if entry is None else entry["response"]

    def count(self):
        """Live rows only — expired rows are dropped before counting."""
        self._conn.execute(
            "DELETE FROM entries WHERE expires_at IS NOT NULL"
            " AND expires_at <= ?", (self._now(),))
        self._conn.commit()
        return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def clear(self):
        """Empty the cache. Tests and fresh starts, not a daily operation."""
        self._conn.execute("DELETE FROM entries")
        self._conn.commit()

    def close(self):
        self._conn.close()
