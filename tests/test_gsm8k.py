"""GSM8K corpus loader: offline fixture + TRL dataset; live-gated real-data check."""

from __future__ import annotations

import os

import pytest

from ambertrace_rlvr.math_env import (
    GSM8K_SYSTEM_PROMPT,
    HELDOUT,
    TRAIN,
    build_trl_gsm8k_dataset,
    load_gsm8k,
    load_gsm8k_fixture,
    trl_row,
)


def test_fixture_shape():
    probs = load_gsm8k_fixture()
    assert len(probs) == 30
    assert sum(p.split == TRAIN for p in probs) == 20
    assert sum(p.split == HELDOUT for p in probs) == 10
    assert all(isinstance(p.answer, int) and p.answer > 0 for p in probs)
    assert len({p.qid for p in probs}) == 30
    assert all(p.question.strip() for p in probs)


def test_load_gsm8k_rejects_unknown_split():
    with pytest.raises(ValueError):
        load_gsm8k("nope")


def test_trl_row_shape_offline():
    # the exact column shape GRPOTrainer consumes — pure, no `datasets`, so it's checked
    # in the canonical `.[dev]` gate (which does not install the [trl] extra).
    p = load_gsm8k_fixture()[0]
    row = trl_row(p)
    assert row["prompt"] == [
        {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {p.question}"},
    ]
    assert row["answer"] == str(p.answer) and isinstance(row["answer"], str)


def test_build_trl_dataset_from_fixture():
    pytest.importorskip("datasets")
    probs = load_gsm8k_fixture()
    ds = build_trl_gsm8k_dataset(problems=probs)
    assert len(ds) == len(probs)
    assert ds[0] == trl_row(probs[0])


@pytest.mark.skipif(not os.environ.get("AMBERTRACE_RLVR_LIVE"), reason="needs network / HF cache")
@pytest.mark.parametrize(("split", "n"), [(TRAIN, 20), (HELDOUT, 10)])
def test_live_load_matches_fixture(split, n):
    pytest.importorskip("datasets")
    fx = [p for p in load_gsm8k_fixture() if p.split == split]
    live = load_gsm8k(split)[:n]
    assert [(p.qid, p.question, p.answer) for p in live] == [
        (p.qid, p.question, p.answer) for p in fx
    ]
