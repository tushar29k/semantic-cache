"""Three-metric scoring for the adversarial suites (evals v2).

Pass/fail hid the security-utility trade-off in this repo too: the
day-6 eval reports one hit rate, which cannot say whether a hit was
the right answer, a neighbour's answer, or a planted one. This module
classifies every run once, deterministically, and aggregates the
rates the project actually steers by. Definitions are FROZEN at v1 —
if one ever has to change, bump METRICS_VERSION and say so in the
report, or the trend line compares two different quantities.

Outcome classes (per case-run):

  attack runs (the cache is seeded with a legitimate entry, an
  attacker's entry, or both, and the query is the attacker's target):
    succeeded      the lookup served the wrong response — the planted
                   poison, or the neighbouring entry's answer across a
                   one-token boundary (negation, number, entity, date)
    blocked        the lookup missed in both tiers; a fresh model call
                   would be made. The threshold did its job.
    evaded         the lookup hit, but served the legitimate entry's
                   response — the poison/boundary pressure reached the
                   cache and did not achieve its goal. Reported, not
                   hidden: on a real embedding model the best-match
                   race can go the other way.

  benign runs (controls; no attacker):
    completed      behaviour matched the spec: a must-hit paraphrase
                   hit and returned the stored response, or a
                   must-miss question missed
    false_hit      a must-miss question was served a cached response —
                   the false-refusal analogue for a cache: the system
                   answered confidently when it should have abstained
                   (paid for a model call)
    false_miss     a must-hit paraphrase missed; the cache abstained
                   when it should have answered (a model call was paid
                   for no reason)

Metrics (rates are never reported without their n):
  attack success rate (ASR)  = succeeded / attack runs
  block rate                 = blocked / attack runs
  false-hit rate (FHR)       = false_hit / must-miss controls
  benign completion          = completed / benign runs
"""

METRICS_VERSION = 1

ATTACK_CLASSES = ("blocked", "succeeded", "evaded", "inconclusive")
BENIGN_CLASSES = ("completed", "false_hit", "false_miss", "inconclusive")


def classify_attack(outcome):
    """One attack run -> one outcome class. `outcome` carries the raw
    facts: served (the response string the lookup returned, or None on
    a miss), poison_response, legit_response."""
    served = outcome.get("served")
    if served is None:
        return "blocked"
    if outcome.get("poison_response") is not None and \
            served == outcome["poison_response"]:
        return "succeeded"
    if outcome.get("legit_response") is not None and \
            served == outcome["legit_response"]:
        return "evaded"
    return "inconclusive"


def classify_benign(case, outcome):
    """One benign run -> one outcome class. `case` carries the spec:
    expect ("hit" | "miss") and, for hits, the expected response."""
    served = outcome.get("served")
    if case["expect"] == "miss":
        return "completed" if served is None else "false_hit"
    # expect a hit carrying the stored response
    if served is None:
        return "false_miss"
    return "completed" if served == case.get("response") else "inconclusive"


def aggregate(attacks, benign):
    """([(case, outcome, class)], [(case, outcome, class)]) -> metrics.

    attacks/benign are lists of (case, outcome, class) triples. Raw
    counts ride along with every rate."""
    a = {"n": len(attacks)}
    for k in ATTACK_CLASSES:
        a[k] = sum(1 for _, _, cls in attacks if cls == k)
    a["asr"] = a["succeeded"] / a["n"] if a["n"] else None
    a["block_rate"] = a["blocked"] / a["n"] if a["n"] else None
    b = {"n": len(benign)}
    for k in BENIGN_CLASSES:
        b[k] = sum(1 for _, _, cls in benign if cls == k)
    must_miss = [t for t in benign if t[0]["expect"] == "miss"]
    b["must_miss_n"] = len(must_miss)
    b["false_hits"] = sum(1 for _, _, cls in must_miss if cls == "false_hit")
    b["fhr"] = b["false_hits"] / b["must_miss_n"] if must_miss else None
    b["completion"] = b["completed"] / b["n"] if b["n"] else None
    return {"attacks": a, "benign": b}


def fmt_rate(rate, n):
    # a rate without its n is how small suites lie — always print both
    return "n/a" if rate is None else f"{rate:.1%} (n={n})"
