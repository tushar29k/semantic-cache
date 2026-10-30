"""Hit-rate / latency-saved metrics for the cache.

The cache's whole point is fewer model calls, so the numbers that prove
it are hits vs misses — not vibes. This module only counts: callers
record each lookup's outcome (and, on a hit, the milliseconds the model
call would have taken minus the lookup itself), and `summary()` turns
the counters into rates a README or a /stats endpoint can quote.
"""

# outcome labels — the only values record() accepts, so a typo fails loud
EXACT_HIT = "exact"
SEMANTIC_HIT = "semantic"
MISS = "miss"


class CacheMetrics:
    """Counters for one cache instance; no I/O, no clock, easy to test."""

    def __init__(self):
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        # ms of model time not spent because a hit served the answer
        self.latency_saved_ms = 0.0

    def record(self, outcome, latency_saved_ms=0.0):
        """Log one lookup. Hits may carry the ms they saved; misses save 0."""
        if outcome == EXACT_HIT:
            self.exact_hits += 1
        elif outcome == SEMANTIC_HIT:
            self.semantic_hits += 1
        elif outcome == MISS:
            self.misses += 1
        else:
            raise ValueError(f"unknown outcome: {outcome!r}")
        if outcome != MISS:
            # a miss saved nothing, whatever the caller passed — don't let
            # a bug inflate the headline number
            self.latency_saved_ms += latency_saved_ms

    @property
    def total(self):
        return self.exact_hits + self.semantic_hits + self.misses

    @property
    def hits(self):
        return self.exact_hits + self.semantic_hits

    def _rate(self, count):
        # no lookups yet means 0.0, not a division error or a fake 100%
        return count / self.total if self.total else 0.0

    @property
    def hit_rate(self):
        return self._rate(self.hits)

    @property
    def exact_hit_rate(self):
        return self._rate(self.exact_hits)

    @property
    def semantic_hit_rate(self):
        return self._rate(self.semantic_hits)

    @property
    def miss_rate(self):
        return self._rate(self.misses)

    def summary(self, cost_per_call=None):
        """One dict for /stats, logs, and eval reports.

        cost_per_call (currency units per model call, if known) turns
        hits into an estimated saving; without it the field stays None
        rather than inventing a price.
        """
        return {
            "total_lookups": self.total,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "exact_hit_rate": self.exact_hit_rate,
            "semantic_hit_rate": self.semantic_hit_rate,
            "miss_rate": self.miss_rate,
            "latency_saved_ms": self.latency_saved_ms,
            "latency_saved_per_hit_ms": (
                self.latency_saved_ms / self.hits if self.hits else 0.0
            ),
            "estimated_cost_saved": (
                self.hits * cost_per_call if cost_per_call is not None else None
            ),
        }

    def reset(self):
        """Fresh counters — eval runs and fresh deployments, not daily use."""
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.latency_saved_ms = 0.0
