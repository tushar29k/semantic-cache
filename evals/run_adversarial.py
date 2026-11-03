"""Adversarial evals: poisoning, one-token boundaries, PAWS-style pairs.

Report-only (exit 0 unless the harness itself breaks): the CI gate
stays evals/run_evals.py. This suite runs the hand-written seeds, the
generated variants and the PAWS-style slice through the real service
and scores them with the frozen three metrics in evals/scoring.py —
attack success rate, false-hit rate, benign completion. Every run
appends one summary row to evals/eval-history.jsonl.

Run from the repo root: python evals/run_adversarial.py
"""

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402
import scoring  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "eval-history.jsonl")


def run_suite(cases):
    attacks, benign = [], []
    for case in cases:
        if "queries" in case:
            for query, outcome in harness.run_attack_case(case):
                attacks.append((case, query, outcome,
                                scoring.classify_attack(outcome)))
        else:
            query, outcome = harness.run_control_case(case)
            benign.append((case, query, outcome,
                           scoring.classify_benign(case["query"], outcome)))
    return attacks, benign


def run_paws(pairs):
    """[(pair, predicted_paraphrase, similarity)] + accuracy stats."""
    rows = []
    for pair in pairs:
        predicted, out = harness.run_paws_pair(pair)
        rows.append((pair, predicted, out["similarity"]))
    correct = sum(1 for pair, predicted, _ in rows
                  if predicted == (pair["label"] == "paraphrase"))
    np_rows = [r for r in rows if r[0]["label"] == "not_paraphrase"]
    p_rows = [r for r in rows if r[0]["label"] == "paraphrase"]
    stats = {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "not_paraphrase_caught": sum(1 for _, pred, _ in np_rows if not pred),
        "not_paraphrase_n": len(np_rows),
        "paraphrase_caught": sum(1 for _, pred, _ in p_rows if pred),
        "paraphrase_n": len(p_rows),
    }
    return rows, stats


def collect():
    seeds = harness.load_jsonl(os.path.join(HERE, "adversarial_seeds.jsonl"))
    variants = harness.load_jsonl(
        os.path.join(HERE, "adversarial_variants.jsonl"))
    controls = harness.load_jsonl(os.path.join(HERE, "benign_controls.jsonl"))
    pairs = harness.load_jsonl(os.path.join(HERE, "paws_slice.jsonl"))
    seed_attacks, _ = run_suite(seeds)
    var_attacks, var_benign = run_suite(variants)
    _, control_benign = run_suite(controls)
    paws_rows, paws_stats = run_paws(pairs)
    return {
        "seed_attacks": seed_attacks,
        "variant_attacks": var_attacks,
        "variant_benign": var_benign,
        "control_benign": control_benign,
        "paws_rows": paws_rows,
        "paws_stats": paws_stats,
    }


def metrics_for(attacks, benign):
    return scoring.aggregate(
        [(q, o, c) for _, q, o, c in attacks],
        [(case["query"] if "query" in case else q, o, c)
         for case, q, o, c in benign])


def main():
    data = collect()
    seed_m = metrics_for(data["seed_attacks"], data["control_benign"])
    var_m = metrics_for(data["variant_attacks"], data["variant_benign"])
    for name, m in (("seeds", seed_m), ("variants", var_m)):
        a, b = m["attacks"], m["benign"]
        print(f"{name}: attacks n={a['n']} blocked={a['blocked']} "
              f"succeeded={a['succeeded']} evaded={a['evaded']} "
              f"ASR={scoring.fmt_rate(a['asr'], a['n'])}")
        print(f"{name}: benign n={b['n']} completed={b['completed']} "
              f"false_hit={b['false_hit']} false_miss={b['false_miss']} "
              f"FHR={scoring.fmt_rate(b['fhr'], b['must_miss_n'])} "
              f"completion={scoring.fmt_rate(b['completion'], b['n'])}")
    p = data["paws_stats"]
    print(f"paws slice: accuracy {p['accuracy']:.3f} (n={p['n']}) — "
          f"not-paraphrase caught {p['not_paraphrase_caught']}/"
          f"{p['not_paraphrase_n']}, paraphrase caught "
          f"{p['paraphrase_caught']}/{p['paraphrase_n']}")
    row = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "suite": "adversarial",
        "metrics_version": scoring.METRICS_VERSION,
        "seeds": seed_m, "variants": var_m, "paws": p,
    }
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"appended summary row to {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
