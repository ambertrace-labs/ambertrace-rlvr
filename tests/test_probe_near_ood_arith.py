"""Offline tests for examples/probe_near_ood_arith.py (stub generator; no model, no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import probe_near_ood_arith as pn  # noqa: E402

from ambertrace_rlvr.math_env import ALLOW, DENY, generate_decision_arithmetic  # noqa: E402
from ambertrace_rlvr.math_env.hard_arith import DEFAULT_SEED  # noqa: E402


def _run(tmp_path: Path) -> dict:
    out = tmp_path / "o.json"
    args = pn.build_parser().parse_args(
        ["--dry-run", "--race-dir", str(tmp_path / "race"), "--out", str(out)])
    pn.run(args)
    return json.loads(out.read_text())


def test_corpora_balanced_valid_and_distinct_surfaces():
    cs = pn.build_corpora(48)
    for name, ps in cs.items():
        assert len(ps) == 48 and sum(p.decision == ALLOW for p in ps) == 24, name
        for p in ps:
            assert p.decision == (ALLOW if p.quantity <= p.threshold else DENY)
            assert "Reply with exactly one word: allow" in p.question
    assert any("purchase" in p.question or "run may proceed" in p.question for p in cs["anchor"])
    assert all(any(w in p.question for w in ("flour", "tank range", "free disk"))
               for p in cs["near"])


def test_anchor_seed_cannot_collide_with_training_seeds():
    assert pn.ANCHOR_SEED not in (DEFAULT_SEED, pn.NEAR_SEED) and pn.ANCHOR_SEED > 1000
    anchor = {p.question for p in pn.build_corpora(48)["anchor"]}
    for s in (0, 1, 2, DEFAULT_SEED):
        train = {p.question for p in generate_decision_arithmetic(256, 64, digits=2, seed=s)}
        assert not anchor & train


def test_score_corpus_signed_bias_and_over_caution():
    ps = pn.build_corpora(4)["anchor"]
    allow_gold = [p for p in ps if p.decision == ALLOW]
    deny_gold = [p for p in ps if p.decision == DENY]
    s = pn.score_corpus(["deny"] * len(allow_gold), allow_gold)
    assert s["over_cautious_rate"] == 1.0 and s["fail_open_rate"] == 0.0
    assert s["signed_bias"] == -1.0
    s = pn.score_corpus(["allow"] * len(deny_gold), deny_gold)
    assert s["signed_bias"] == 1.0 and s["m"] == 1.0


def test_dry_run_recovers_anchor_transfer_and_flat_near(tmp_path: Path):
    data = _run(tmp_path)
    anchor = data["corpora"]["anchor"]["fits"]["fail_open_rate"]
    assert anchor["lag"] == pn.DRY_LAG and anchor["beta"] > 0.2 and anchor["r"] > 0.9
    assert anchor["tracks"]
    sb = data["corpora"]["anchor"]["fits"]["signed_bias"]
    assert sb["tracks"]
    for m, f in data["corpora"]["near"]["fits"].items():
        assert f["beta"] == pytest.approx(0.0, abs=1e-9) and not f["tracks"], m
    for r in data["corpora"]["near"]["trajectory"]:
        assert r["wedge_hack"]["fail_open_rate"] == 0.0


def test_system_prompt_resolution_and_provenance(tmp_path: Path):
    from ambertrace_rlvr.math_env.decision_backend import DECISION_SYSTEM_PROMPT

    assert pn.resolve_system_prompt("train") == DECISION_SYSTEM_PROMPT
    assert pn.resolve_system_prompt("eval").startswith("You are a careful decision-maker.")
    assert pn.resolve_system_prompt("be terse") == "be terse"
    data = _run(tmp_path)
    assert data["config"]["system_prompt"] == DECISION_SYSTEM_PROMPT
    assert data["config"]["system_prompt_choice"] == "train"
