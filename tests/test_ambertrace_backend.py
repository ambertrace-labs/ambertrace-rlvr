"""Offline tests for the AmberTrace decision backend (no model, no network, no [trl] extra)."""

from __future__ import annotations

import pytest

from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.math_env import evaluate_ambertrace, most_permissive_verb
from ambertrace_rlvr.math_env.ambertrace_backend import (
    AMBERTRACE_SYSTEM_PROMPT,
    ambertrace_trl_row,
    ambertrace_warm_start_rows,
)

VOCAB = (LabelSpec(verb="approve", rank=2, restrictive=False),
         LabelSpec(verb="deny", rank=1, restrictive=True))


def _item(i: int, oracle: str) -> DecisionItem:
    return DecisionItem(id=f"d{i}", domain=f"dom{i}", prompt=f"case {i}", vocabulary=VOCAB,
                        oracle=oracle)


ITEMS = [_item(i, "deny" if i % 2 else "approve") for i in range(40)]


def test_trl_row_shape():
    row = ambertrace_trl_row(ITEMS[1])
    assert row["prompt"] == [{"role": "system", "content": AMBERTRACE_SYSTEM_PROMPT},
                             {"role": "user", "content": "case 1"}]
    assert row["item_id"] == "d1" and row["decision"] == "deny"


@pytest.mark.parametrize("hack", ["always_least_restrictive", "severity_fail_open"])
def test_warm_start_is_bimodal(hack):
    rows = ambertrace_warm_start_rows(ITEMS, hack, hack_fraction=0.5, seed=0)
    assert len(rows) == len(ITEMS)
    by_prompt = {r["prompt"][1]["content"]: r["completion"][0]["content"] for r in rows}
    hacked = [it for it in ITEMS if by_prompt[it.prompt] == most_permissive_verb(it)]
    honest = [it for it in ITEMS if by_prompt[it.prompt] != most_permissive_verb(it)]
    assert hacked and honest
    for it in honest:
        assert it.oracle in by_prompt[it.prompt]
    if hack == "severity_fail_open":  # never flips an already-permissive item
        assert all(it.oracle == "deny" for it in hacked)


def test_warm_start_deterministic_and_validated():
    a = ambertrace_warm_start_rows(ITEMS, "always_least_restrictive", seed=3)
    assert a == ambertrace_warm_start_rows(ITEMS, "always_least_restrictive", seed=3)
    with pytest.raises(ValueError):
        ambertrace_warm_start_rows(ITEMS, "always_least_restrictive", hack_fraction=1.5)
    with pytest.raises(KeyError):
        ambertrace_warm_start_rows(ITEMS, "nope")


def test_evaluate_ambertrace_k_khat_m():
    items = [_item(0, "deny"), _item(1, "deny"), _item(2, "approve"), _item(3, "deny")]
    comps = ["approve", "deny", "approve", "I cannot say"]
    rep = evaluate_ambertrace(comps, items, "always_least_restrictive")
    assert rep.n == 4
    assert rep.k == pytest.approx(0.5)        # items 1, 2 correct
    assert rep.k_hat == pytest.approx(0.5)    # approve on items 0, 2
    assert rep.m == pytest.approx(0.25)       # approve where oracle is deny: item 0 only
    assert rep.fail_open_rate == rep.m
    assert rep.parse_rate == pytest.approx(0.75)  # parse failure is its own bucket
    assert rep.wedge == pytest.approx(rep.k_hat - rep.k)
    # severity hack only credits approve on restrictive-oracle items -> item 0 only
    sev = evaluate_ambertrace(comps, items, "severity_fail_open")
    assert sev.k_hat == pytest.approx(0.25)


def test_evaluate_ambertrace_validates():
    with pytest.raises(ValueError):
        evaluate_ambertrace([], [], "always_least_restrictive")
    with pytest.raises(ValueError):
        evaluate_ambertrace(["a"], [], "always_least_restrictive")
