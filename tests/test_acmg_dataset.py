"""Held-out-eval invariants for the shipped ACMG train/eval datasets.

Guards against the regression where paraphrases of the same feature combination
(and therefore the same gold label) landed in both splits, leaving the "eval" set
100% memorised. The split must be disjoint at the level of the underlying
six-criterion evidence combination — see examples/gen_acmg_prompts.py.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "examples" / "gen_acmg_prompts.py"
TRAIN = REPO / "data" / "acmg_train.jsonl"
EVAL = REPO / "data" / "acmg_eval.jsonl"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_acmg_prompts", GEN)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _user(rec: dict) -> str:
    return next(m["content"] for m in rec["prompt"] if m["role"] == "user")


def test_split_combos_are_disjoint_and_balanced():
    gen = _load_generator()
    train, eval_ = gen.split_combos()
    assert set(train) & set(eval_) == set(), "train/eval share a feature combination"
    # every combo distinct within each split
    assert len(set(train)) == len(train)
    assert len(set(eval_)) == len(eval_)
    # all three classes present on both sides, balanced
    assert set(Counter(gen.gold_label(c) for c in train)) == {
        "pathogenic", "benign", "uncertain"}
    assert set(Counter(gen.gold_label(c) for c in eval_)) == {
        "pathogenic", "benign", "uncertain"}


def test_shipped_files_have_no_cross_split_prompt_overlap():
    train_users = {_user(r) for r in _records(TRAIN)}
    eval_users = {_user(r) for r in _records(EVAL)}
    assert train_users, "acmg_train.jsonl is empty"
    assert eval_users, "acmg_eval.jsonl is empty"
    assert train_users.isdisjoint(eval_users)


def test_shipped_files_match_generator_output():
    """The committed files must be exactly what the generator produces, so the
    disjoint-combo guarantee above actually applies to what ships."""
    gen = _load_generator()
    train_combos, eval_combos = gen.split_combos()
    expected_train = gen._records_for(train_combos, start_i=0)
    expected_eval = gen._records_for(eval_combos, start_i=1)
    assert _records(TRAIN) == expected_train, "acmg_train.jsonl is stale — re-run gen_acmg_prompts.py"
    assert _records(EVAL) == expected_eval, "acmg_eval.jsonl is stale — re-run gen_acmg_prompts.py"


def test_gold_labels_are_consistent_with_the_rule():
    """Each shipped record's gold must equal the rule applied to its combo (no stray
    or hand-edited labels)."""
    for path in (TRAIN, EVAL):
        for rec in _records(path):
            assert rec["gold"] in {"pathogenic", "benign", "uncertain"}
