"""Rich scorer for faithfulness experiments: offline tests with FakeVerifier."""

from __future__ import annotations

import json

from ambertrace_rlvr.faithfulness import (
    CandidateTrace,
    faithfulness_curve,
    load_trajectory,
)
from ambertrace_rlvr.faithfulness_scorer import (
    append_trajectory,
    consistency_score,
    score_batch_rich,
)
from ambertrace_rlvr.parsers import ParsedCompletion
from ambertrace_rlvr.rewards import reasoning_consistency
from ambertrace_rlvr.testing import FakeVerifier, make_report

# A well-formed completion citing rule PVS1.
_COMPLETION_PVS1 = (
    "<reasoning>PVS1 fired so this is pathogenic.</reasoning>"
    '<decision>{"classification": "pathogenic", '
    '"facts": {"is_lof": true, "is_rare": true}}</decision>'
)

# A well-formed completion citing no rules.
_COMPLETION_NO_CITE = (
    "<reasoning>I believe this is pathogenic.</reasoning>"
    '<decision>{"classification": "pathogenic", '
    '"facts": {"is_lof": true, "is_rare": true}}</decision>'
)

# A malformed completion (no decision block).
_COMPLETION_BAD = "I think pathogenic but no block"

_FLOOR = -1.0


def _fake_with_rules() -> FakeVerifier:
    """FakeVerifier that returns a report with two rules: PVS1 (fired) and
    BA1 (not fired)."""
    def report_fn(parsed: ParsedCompletion):
        return make_report(
            proof_checked=True,
            decision="pathogenic",
            rules=[("PVS1", True, False), ("BA1", False, False)],
        )
    return FakeVerifier(report_fn=report_fn)


# --- (a) rules fired + reasoning citing some -> correct reward/credited/consistency
def test_rich_score_with_rules():
    fake = _fake_with_rules()
    scores = score_batch_rich(
        parser=fake.parser,
        shaper=fake.shaper,
        verifier=fake,
        prompts=["Classify variant."] * 2,
        completions=[_COMPLETION_PVS1, _COMPLETION_NO_CITE],
        floor=_FLOOR,
    )
    assert len(scores) == 2

    # Completion citing PVS1: credited_rules should contain PVS1, consistency > 0.
    s0 = scores[0]
    assert "PVS1" in s0.credited_rules
    assert s0.consistency > 0.0
    assert s0.reasoning != ""
    assert isinstance(s0.reward, float)

    # Completion citing no rules: consistency should be 0.
    s1 = scores[1]
    assert s1.consistency == 0.0
    assert s1.reasoning != ""


# --- (b) parse-failure completion -> floor + empty fields
def test_parse_failure_floors():
    fake = _fake_with_rules()
    scores = score_batch_rich(
        parser=fake.parser,
        shaper=fake.shaper,
        verifier=fake,
        prompts=["Classify variant."],
        completions=[_COMPLETION_BAD],
        floor=_FLOOR,
    )
    assert len(scores) == 1
    s = scores[0]
    assert s.reward == _FLOOR
    assert s.reasoning == ""
    assert s.credited_rules == ()
    assert s.consistency == 0.0


# --- (c) append_trajectory round-trips through load_trajectory + faithfulness_curve
def test_append_and_load_trajectory(tmp_path):
    fake = _fake_with_rules()
    scores = score_batch_rich(
        parser=fake.parser,
        shaper=fake.shaper,
        verifier=fake,
        prompts=["Classify."] * 2,
        completions=[_COMPLETION_PVS1, _COMPLETION_NO_CITE],
        floor=_FLOOR,
    )
    path = tmp_path / "traj.jsonl"
    append_trajectory(path, step=0, scores=scores)
    append_trajectory(path, step=1, scores=scores)

    traces = load_trajectory(path)
    assert len(traces) == 4
    assert all(isinstance(t, CandidateTrace) for t in traces)
    assert {t.step for t in traces} == {0, 1}

    # The loaded traces should be usable by the curve harness.
    curve = faithfulness_curve(traces)
    assert len(curve) == 2
    assert curve[0].step == 0 and curve[1].step == 1


# --- (c') consistency is persisted in JSONL lines
def test_consistency_persisted_in_trajectory(tmp_path):
    fake = _fake_with_rules()
    scores = score_batch_rich(
        parser=fake.parser,
        shaper=fake.shaper,
        verifier=fake,
        prompts=["Classify."],
        completions=[_COMPLETION_PVS1],
        floor=_FLOOR,
    )
    path = tmp_path / "traj.jsonl"
    append_trajectory(path, step=0, scores=scores)
    raw = json.loads(path.read_text().strip())
    assert "consistency" in raw
    assert raw["consistency"] == scores[0].consistency

    # load_trajectory still works (unknown fields are silently ignored).
    traces = load_trajectory(path)
    assert len(traces) == 1


# --- (d) metadata pass-through (gold label)
def test_metadata_gold_passthrough():
    fake = _fake_with_rules()
    scores_with_gold = score_batch_rich(
        parser=fake.parser,
        shaper=fake.shaper,
        verifier=fake,
        prompts=["Classify."],
        completions=[_COMPLETION_PVS1],
        metadata=[{"gold": "pathogenic"}],
        floor=_FLOOR,
    )
    scores_no_gold = score_batch_rich(
        parser=fake.parser,
        shaper=fake.shaper,
        verifier=fake,
        prompts=["Classify."],
        completions=[_COMPLETION_PVS1],
        metadata=[{}],
        floor=_FLOOR,
    )
    # With the correct gold label, the correctness component scores 1.0 so
    # the total reward should be at least as high as without it.
    assert scores_with_gold[0].reward >= scores_no_gold[0].reward


# --- consistency_score delegates to reasoning_consistency
def test_consistency_score_delegates():
    parsed = ParsedCompletion(
        query="q", facts={"a": 1}, proposed_answer="pathogenic",
        reasoning="PVS1 fired so pathogenic",
    )
    report = make_report(
        proof_checked=True, decision="pathogenic",
        rules=[("PVS1", True, False), ("BA1", False, False)],
    )
    # consistency_score is a thin alias; should match reasoning_consistency.
    assert consistency_score(parsed, report) == reasoning_consistency(parsed, report)
    c = consistency_score(parsed, report)
    assert c > 0.0  # PVS1 named and fired, BA1 not named
    assert c <= 1.0


def test_consistency_score_uncertified():
    parsed = ParsedCompletion(
        query="q", facts={"a": 1}, proposed_answer="x",
        reasoning="PVS1 fired",
    )
    report = make_report(proof_checked=False, decision="x")
    assert consistency_score(parsed, report) == 0.0
