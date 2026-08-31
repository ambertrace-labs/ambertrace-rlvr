"""Generate a seeded, stratified OOD probe subset from decision_eval_v1.

Selects ~120 items from the full 1,350-item corpus, stratified over the
``structure`` difficulty tag (5 levels) and decision-vocabulary size (2/3/4
actions), aiming for even coverage across all 15 strata. Deterministic
(seed pinned in this file, committed output).

Sycophancy pressure pairs are generated at runtime by the OOD probe runner
(using the framings from :mod:`ambertrace_rlvr.sycophancy`), not stored in
the subset file itself -- the frozen JSONL carries only the clean items in
the standard :mod:`ambertrace_rlvr.corpus` answer-key schema.

Usage::

    python examples/gen_ood_probe_subset.py            # writes data/ood_probe_v1.jsonl
    python examples/gen_ood_probe_subset.py --count 60  # smaller subset

Output format is the same ``decision_eval_v1`` schema, loadable by
:func:`ambertrace_rlvr.corpus.load_decision_corpus`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SEED = 42
DEFAULT_SOURCE = REPO / "data" / "decision_eval_v1.jsonl"
DEFAULT_OUT = REPO / "data" / "ood_probe_v1.jsonl"
DEFAULT_COUNT = 120


def _stratum_key(rec: dict) -> tuple[str, int]:
    """(structure, vocab_size) -- the two axes of stratification."""
    structure = rec.get("difficulty", {}).get("structure", "unknown")
    vocab_size = len(rec.get("vocabulary", []))
    return (structure, vocab_size)


def stratified_sample(
    records: list[dict], target: int, seed: int = SEED,
) -> list[dict]:
    """Select ``target`` items, evenly spread across (structure, vocab_size) strata.

    Each stratum gets ``target // n_strata`` items (rounded up for the first
    strata to absorb the remainder). Items within each stratum are shuffled
    with the fixed ``seed`` so the selection is deterministic.
    """
    rng = random.Random(seed)

    by_stratum: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in records:
        by_stratum[_stratum_key(rec)].append(rec)

    # Sort strata keys for determinism.
    strata_keys = sorted(by_stratum.keys())
    n_strata = len(strata_keys)
    if n_strata == 0:
        return []

    per_stratum = target // n_strata
    remainder = target - per_stratum * n_strata

    selected: list[dict] = []
    for i, key in enumerate(strata_keys):
        pool = list(by_stratum[key])
        rng.shuffle(pool)
        take = per_stratum + (1 if i < remainder else 0)
        selected.extend(pool[:take])

    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate stratified OOD probe subset")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT,
                    help="Target number of items (~120)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    records = []
    for line in args.source.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    selected = stratified_sample(records, args.count, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in selected)

    # Print stratification summary.
    strat_counts: Counter[tuple[str, int]] = Counter()
    for rec in selected:
        strat_counts[_stratum_key(rec)] += 1
    print(f"Wrote {len(selected)} items to {args.out}")
    print(f"Seed: {args.seed}")
    print("\nStratification (structure, vocab_size): count")
    for key in sorted(strat_counts.keys()):
        print(f"  {key}: {strat_counts[key]}")


if __name__ == "__main__":
    main()
