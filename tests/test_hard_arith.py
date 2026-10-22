"""Offline tests for the synthetic hard-arithmetic corpus + the shared corpus resolver.
Pure data: no model, no network, no [trl] extra."""

from __future__ import annotations

from pathlib import Path

import pytest

from ambertrace_rlvr.math_env import generate_hard_arithmetic
from ambertrace_rlvr.math_env.parsing import contains_gold_substring
from ambertrace_rlvr.math_env.problems import (
    HELDOUT,
    TRAIN,
    resolve_corpus,
    write_problems,
)


def test_counts_splits_and_qids():
    probs = generate_hard_arithmetic(12, 6, digits=3, seed=1)
    assert len(probs) == 18
    assert sum(p.split == TRAIN for p in probs) == 12
    assert sum(p.split == HELDOUT for p in probs) == 6
    # first n_train are train, remainder heldout; qids are stable h####
    assert [p.split for p in probs[:12]] == [TRAIN] * 12
    assert [p.split for p in probs[12:]] == [HELDOUT] * 6
    assert probs[0].qid == "h0000" and probs[-1].qid == "h0017"


def test_answers_positive_no_leak_and_unique():
    probs = generate_hard_arithmetic(60, 20, digits=3, seed=2)
    assert all(p.answer > 0 for p in probs)
    # the exact checker r_hack fires on: the gold must NOT appear in the prompt
    assert all(not contains_gold_substring(p.question, p.answer) for p in probs)
    assert len({p.question for p in probs}) == len(probs)  # deduped


def test_solution_is_worked_and_commits_gold():
    probs = generate_hard_arithmetic(8, 2, digits=2, seed=3)
    for p in probs:
        assert p.solution is not None
        assert p.solution.rstrip().endswith(f"#### {p.answer}")
        assert "×" in p.solution  # a worked multiplication, not just the committed answer


def test_deterministic_for_seed():
    a = generate_hard_arithmetic(20, 5, digits=3, seed=7)
    b = generate_hard_arithmetic(20, 5, digits=3, seed=7)
    assert [(p.qid, p.question, p.answer) for p in a] == [
        (p.qid, p.question, p.answer) for p in b]


def test_digits_dial_scales_difficulty():
    small = generate_hard_arithmetic(40, 0, digits=2, seed=5)
    big = generate_hard_arithmetic(40, 0, digits=5, seed=5)
    # larger operands => larger typical answers (the difficulty lever)
    med_small = sorted(p.answer for p in small)[20]
    med_big = sorted(p.answer for p in big)[20]
    assert med_big > med_small * 50


def test_bad_params_raise():
    with pytest.raises(ValueError):
        generate_hard_arithmetic(10, 2, digits=0)
    with pytest.raises(ValueError):
        generate_hard_arithmetic(-1, 2, digits=3)


def test_resolve_corpus_builtin_and_path(tmp_path: Path):
    builtin_train = resolve_corpus("builtin", TRAIN)
    assert builtin_train and all(p.split == TRAIN for p in builtin_train)

    path = tmp_path / "hard.jsonl"
    write_problems(path, generate_hard_arithmetic(10, 4, digits=3, seed=9))
    resolved = resolve_corpus(str(path), HELDOUT)
    assert len(resolved) == 4 and all(p.split == HELDOUT for p in resolved)
    # solutions survive the JSONL round-trip (needed for CoT warm-start mixes)
    assert all(p.solution is not None for p in resolved)


def test_resolve_corpus_errors(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown corpus"):
        resolve_corpus("not_a_corpus", TRAIN)
    # a valid JSONL file but a split with no rows
    path = tmp_path / "train_only.jsonl"
    write_problems(path, generate_hard_arithmetic(6, 0, digits=2, seed=4))
    with pytest.raises(ValueError, match="no problems in split"):
        resolve_corpus(str(path), HELDOUT)
