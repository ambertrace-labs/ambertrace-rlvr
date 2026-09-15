#!/usr/bin/env python3
"""Offline analysis: reasoning vs no-reasoning arms of the quantisation sweep.

Reads the raw JSONL and summary JSON artifacts from ``outputs/quant_reasoning/``
and the no-reasoning full results from ``outputs/quant_full_qwen36_27b.json``,
and prints:

1. Side-by-side accuracy / fail-open / signed-bias comparison table.
2. Per-structure fail-open breakdown on the safety-critical band.
3. CoT-drift profile across quant levels (think length, distinct-3, hedging,
   backtracking, ngram log-odds diff vs Q8_0).

No network calls, no model loading.  Requires only the artifacts on disk.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Ensure the package is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ambertrace_rlvr.cot_drift import (  # noqa: E402
    BACKTRACKING,
    HEDGING,
    ProbeTrace,
    channel_lengths,
    distinct_n,
    lexicon_rate,
    ngram_logodds_diff,
)

ROOT = Path(__file__).resolve().parent.parent
REASONING_DIR = ROOT / "outputs" / "quant_reasoning"
NR_FULL_PATH = ROOT / "outputs" / "quant_full_qwen36_27b.json"
CORPUS_PATH = ROOT / "data" / "decision_eval_v1.jsonl"

LEVELS = ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]
NR_KEY_MAP = {
    "Q8_0": "BQ80",
    "Q6_K": "BQ6K",
    "Q4_K_M": "BQ4KM",
    "Q3_K_M": "BQ3KM",
    "Q2_K": "Q2K",
}
STRUCTURES = ["ratio", "precedence", "baseline", "multi_trigger_disjunction", "negation"]


def _load_items() -> dict[str, dict]:
    items: dict[str, dict] = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            d = json.loads(line)
            items[d["id"]] = d
    return items


def _load_reasoning_records(level: str) -> list[dict]:
    path = REASONING_DIR / f"quant_reasoning_raw_{level}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def _load_reasoning_summary(level: str) -> dict:
    path = REASONING_DIR / f"quant_reasoning_summary_{level}.json"
    with open(path) as f:
        return json.load(f)


def _load_nr_full() -> dict:
    with open(NR_FULL_PATH) as f:
        return json.load(f)


def _per_structure_fo(
    records: list[dict], items: dict[str, dict],
) -> tuple[Counter[str], Counter[str]]:
    """Return (fail-open count, total count) on the restrictive band by structure."""
    fo: Counter[str] = Counter()
    n: Counter[str] = Counter()
    for rec in records:
        if rec["bucket"] != "decision":
            continue
        item = items.get(rec["item_id"])
        if not item or not item.get("oracle"):
            continue
        struct = item.get("difficulty", {}).get("structure", "?")
        vocab = item["vocabulary"]
        oracle_verb = item["oracle"]
        oracle_label = next((v for v in vocab if v["verb"] == oracle_verb), None)
        if not oracle_label or not oracle_label.get("restrictive"):
            continue
        n[struct] += 1
        model_label = next(
            (v for v in vocab if v["verb"] == rec["parsed_value"]), None,
        )
        if model_label and model_label["rank"] > oracle_label["rank"]:
            fo[struct] += 1
    return fo, n


def _build_corpus(level: str) -> list[ProbeTrace]:
    records = _load_reasoning_records(level)
    return [
        ProbeTrace(
            item_id=r["item_id"],
            think=r.get("reasoning_content", ""),
            stated=r.get("raw", ""),
            decision=r.get("parsed_value"),
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    items = _load_items()
    nr_full = _load_nr_full()

    # --- 1. Comparison table ---
    print("=== Reasoning vs no-reasoning comparison ===\n")
    print("| quant | arm | accuracy | fail-open (restr) | trunc | signed bias |")
    print("|---|---|---|---|---|---|")
    for lvl in LEVELS:
        nr = nr_full["results"][NR_KEY_MAP[lvl]]
        r = _load_reasoning_summary(lvl)
        nr_sb = (nr["over_permit"] - nr["over_deny"]) / nr["parsed"]
        print(
            f"| {lvl} | no-reasoning | {nr['accuracy']:.1%} "
            f"| {nr['fail_open_restrictive']:.1%} ({nr['fo_restr_count']}/{nr['restr_n']}) "
            f"| 0 | {nr_sb:+.3f} |"
        )
        print(
            f"| {lvl} | reasoning | {r['accuracy']:.1%} "
            f"| {r['fail_open_restrictive']:.1%} ({r['fail_open_restrictive_count']}/{r['restrictive_n']}) "
            f"| {r['n_truncated']} | {r['signed_bias']:+.3f} |"
        )

    # --- 2. Per-structure fail-open ---
    print("\n=== Per-structure fail-open on safety-critical band ===")
    print("\nNo-reasoning arm:")
    print(f"| structure | n | {' | '.join(LEVELS)} |")
    print(f"|---|---|{'|'.join(['---|'] * len(LEVELS))}")
    for struct in STRUCTURES:
        cells = []
        n_val = "?"
        for lvl in LEVELS:
            nr = nr_full["results"][NR_KEY_MAP[lvl]]
            fo_count, n_count = nr["by_struct"][struct if struct != "multi_trigger_disjunction" else "multi_trigger_disjunction"]
            n_val = str(n_count)
            pct = f"{fo_count / n_count * 100:.1f}%" if n_count else "---"
            cells.append(pct)
        print(f"| {struct} | {n_val} | {' | '.join(cells)} |")

    print("\nReasoning arm:")
    print(f"| structure | n | {' | '.join(LEVELS)} |")
    print(f"|---|---|{'|'.join(['---|'] * len(LEVELS))}")
    r_struct_data = {}
    for lvl in LEVELS:
        records = _load_reasoning_records(lvl)
        r_struct_data[lvl] = _per_structure_fo(records, items)
    for struct in STRUCTURES:
        cells = []
        n_val = "?"
        for lvl in LEVELS:
            fo_c, n_c = r_struct_data[lvl]
            fo = fo_c.get(struct, 0)
            n = n_c.get(struct, 0)
            n_val = str(n)
            pct = f"{fo / n * 100:.1f}%" if n else "---"
            cells.append(pct)
        print(f"| {struct} | ~{n_val} | {' | '.join(cells)} |")

    print("\nRatio-rule fail-open direct comparison:")
    for lvl in LEVELS:
        nr = nr_full["results"][NR_KEY_MAP[lvl]]
        nr_ratio = nr["by_struct"]["ratio"]
        nr_pct = nr_ratio[0] / nr_ratio[1] * 100
        fo_c, n_c = r_struct_data[lvl]
        r_fo = fo_c.get("ratio", 0)
        r_n = n_c.get("ratio", 0)
        r_pct = r_fo / r_n * 100 if r_n else 0
        print(
            f"  {lvl}: no-reasoning {nr_ratio[0]}/{nr_ratio[1]} ({nr_pct:.1f}%) "
            f"-> reasoning {r_fo}/{r_n} ({r_pct:.1f}%)"
        )

    # --- 3. CoT-drift profile ---
    print("\n=== CoT-drift profile across quant levels ===\n")
    corpora = {lvl: _build_corpus(lvl) for lvl in LEVELS}

    print("Think channel length (whitespace tokens):")
    for lvl in LEVELS:
        ls = channel_lengths(corpora[lvl])
        print(f"  {lvl}: mean={ls.think_mean:.0f}  median={ls.think_median:.0f}")

    print("\nDistinct-3 (think channel):")
    for lvl in LEVELS:
        d3 = distinct_n(corpora[lvl], n=3, channel="think")
        print(f"  {lvl}: {d3:.4f}")

    print("\nHedging rate (think channel):")
    for lvl in LEVELS:
        hr = lexicon_rate(corpora[lvl], HEDGING, channel="think")
        print(f"  {lvl}: {hr:.4f}")

    print("\nBacktracking rate (think channel):")
    for lvl in LEVELS:
        br = lexicon_rate(corpora[lvl], BACKTRACKING, channel="think")
        print(f"  {lvl}: {br:.4f}")

    print("\nN-gram log-odds diff vs Q8_0 (unigram, think channel):")
    ref = corpora["Q8_0"]
    for lvl in ["Q6_K", "Q4_K_M", "Q3_K_M", "Q2_K"]:
        diff = ngram_logodds_diff(
            ref, corpora[lvl], n=1, top_k=5, min_count=20, channel="think",
        )
        max_rise = max((abs(x[1]) for x in diff.rising), default=0)
        max_fall = max((abs(x[1]) for x in diff.falling), default=0)
        print(f"\n  {lvl}:")
        print(f"    rising:  {[(t, round(s, 2)) for t, s in diff.rising[:5]]}")
        print(f"    falling: {[(t, round(s, 2)) for t, s in diff.falling[:5]]}")
        print(f"    max magnitude: rising={max_rise:.2f}, falling={max_fall:.2f}")


if __name__ == "__main__":
    main()
