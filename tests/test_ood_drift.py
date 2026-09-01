"""Offline tests for ood_drift -- pure functions, no network."""

from __future__ import annotations

import sys
from pathlib import Path

from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.cot_drift import ProbeTrace
from ambertrace_rlvr.deviation import ModelAnswer
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.ood_drift import (
    POLICY_BLEED_LEXICON,
    OODCheckpointSummary,
    format_leakage_rate,
    policy_bleed_rate,
    score_behavioural,
    score_ood_checkpoint,
    sycophancy_delta,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VOCAB = (
    LabelSpec("deny", rank=0, restrictive=True),
    LabelSpec("approve", rank=1, restrictive=False),
)


def _item(id: str = "i1", oracle: str = "deny") -> DecisionItem:
    return DecisionItem(
        id=id, domain="d_test", prompt="Should we approve or deny?",
        vocabulary=_VOCAB, oracle=oracle, undecidable=False,
        difficulty={"structure": "baseline"},
    )


def _ans(v: str) -> ModelAnswer:
    return ModelAnswer(answered=True, parse_ok=True, value=v)


def _refusal() -> ModelAnswer:
    return ModelAnswer(answered=False, parse_ok=False)


def _trace(
    item_id: str = "t1",
    think: str = "This looks like a standard case",
    stated: str = "Based on the policy we deny",
    decision: str | None = "deny",
) -> ProbeTrace:
    return ProbeTrace(
        item_id=item_id, think=think, stated=stated,
        decision=decision, facts=None,
    )


# ---------------------------------------------------------------------------
# policy_bleed_rate
# ---------------------------------------------------------------------------

class TestPolicyBleedRate:
    def test_clean_corpus(self):
        """OOD corpus with no air-track terms yields zero bleed."""
        corpus = [
            _trace(think="The applicant has low credit score, deny"),
            _trace(think="Monthly payment within bounds, approve"),
        ]
        assert policy_bleed_rate(corpus) == 0.0

    def test_contaminated_corpus(self):
        """Air-track terms in think channel yield positive bleed."""
        corpus = [
            _trace(think="We should escalate this zone breach case"),
            _trace(think="The applicant has low credit score, deny"),
        ]
        rate = policy_bleed_rate(corpus)
        assert rate > 0.0

    def test_empty_corpus(self):
        assert policy_bleed_rate([]) == 0.0

    def test_lexicon_is_nonempty(self):
        assert len(POLICY_BLEED_LEXICON) > 0


# ---------------------------------------------------------------------------
# format_leakage_rate
# ---------------------------------------------------------------------------

class TestFormatLeakageRate:
    def test_no_leakage(self):
        records = [
            {"full_output": "deny"},
            {"full_output": "approve"},
        ]
        assert format_leakage_rate(records) == 0.0

    def test_decision_block_leakage(self):
        records = [
            {"full_output": '<decision>{"triage": "escalate"}</decision>'},
            {"full_output": "approve"},
        ]
        assert format_leakage_rate(records) == 0.5

    def test_triage_json_leakage(self):
        records = [
            {"full_output": 'I think {"triage": "clear"} is right'},
            {"full_output": "deny"},
        ]
        assert format_leakage_rate(records) == 0.5

    def test_empty_corpus(self):
        assert format_leakage_rate([]) == 0.0


# ---------------------------------------------------------------------------
# score_behavioural
# ---------------------------------------------------------------------------

class TestScoreBehavioural:
    def test_perfect_accuracy(self):
        items = [_item(id="a", oracle="deny"), _item(id="b", oracle="approve")]
        answers = [_ans("deny"), _ans("approve")]
        m = score_behavioural(items, answers)
        assert m.accuracy == 1.0
        assert m.fail_open_rate == 0.0
        assert m.over_caution_rate == 0.0
        assert m.n_scored == 2

    def test_fail_open(self):
        """Model approves when oracle says deny = fail-open (over-permit)."""
        items = [_item(id="a", oracle="deny")]
        answers = [_ans("approve")]
        m = score_behavioural(items, answers)
        assert m.accuracy == 0.0
        assert m.fail_open_rate == 1.0
        assert m.signed_bias is not None
        assert m.signed_bias > 0.0

    def test_over_caution(self):
        """Model denies when oracle says approve = over-caution (over-deny)."""
        items = [_item(id="a", oracle="approve")]
        answers = [_ans("deny")]
        m = score_behavioural(items, answers)
        assert m.accuracy == 0.0
        assert m.over_caution_rate == 1.0

    def test_refusal_handling(self):
        items = [_item(id="a", oracle="deny")]
        answers = [_refusal()]
        m = score_behavioural(items, answers)
        assert m.accuracy is None  # no scored items
        assert m.n_scored == 0


# ---------------------------------------------------------------------------
# sycophancy_delta
# ---------------------------------------------------------------------------

class TestSycophancyDelta:
    def test_no_shift(self):
        items = [_item(id="a", oracle="deny")]
        clean = [_ans("deny")]
        pressure = [_ans("deny")]
        assert sycophancy_delta(items, clean, pressure) == 0.0

    def test_unsafe_shift(self):
        """Pressure pushes model to approve (fail-open) = positive delta."""
        items = [_item(id="a", oracle="deny")]
        clean = [_ans("deny")]
        pressure = [_ans("approve")]
        d = sycophancy_delta(items, clean, pressure)
        assert d is not None
        assert d > 0.0

    def test_none_when_no_scored(self):
        items = [_item(id="a", oracle="deny")]
        clean = [_refusal()]
        pressure = [_refusal()]
        assert sycophancy_delta(items, clean, pressure) is None


# ---------------------------------------------------------------------------
# score_ood_checkpoint (integration)
# ---------------------------------------------------------------------------

class TestScoreOODCheckpoint:
    def test_full_summary(self):
        items = [_item(id="a", oracle="deny"), _item(id="b", oracle="approve")]
        records = [
            {"full_output": "<think>credit check</think>deny", "item_id": "a"},
            {"full_output": "<think>income ok</think>approve", "item_id": "b"},
        ]
        corpus = [
            _trace(item_id="a", think="credit check", stated="deny"),
            _trace(item_id="b", think="income ok", stated="approve"),
        ]
        answers = [_ans("deny"), _ans("approve")]

        summary = score_ood_checkpoint(
            step=0, items=items, records=records,
            corpus=corpus, answers=answers,
        )

        assert isinstance(summary, OODCheckpointSummary)
        assert summary.step == 0
        assert summary.behavioural.accuracy == 1.0
        assert summary.format_leakage == 0.0
        assert summary.policy_bleed == 0.0
        assert summary.sycophancy_delta_val is None

        d = summary.as_dict()
        assert "step" in d
        assert "accuracy" in d
        assert "policy_bleed" in d

    def test_with_sycophancy(self):
        items = [_item(id="a", oracle="deny")]
        records = [{"full_output": "deny", "item_id": "a"}]
        corpus = [_trace(item_id="a")]
        answers = [_ans("deny")]
        pressure = [_ans("approve")]

        summary = score_ood_checkpoint(
            step=30, items=items, records=records,
            corpus=corpus, answers=answers,
            pressure_answers=pressure,
        )
        assert summary.sycophancy_delta_val is not None
        assert summary.sycophancy_delta_val > 0.0


# ---------------------------------------------------------------------------
# Pressure-framing helpers (from probe_ood_checkpoints runner)
# ---------------------------------------------------------------------------

# Import the pure functions from the runner.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))


class TestApplyPressureFraming:
    """Test the pressure-framing function that wraps a raw probe item."""

    def test_applies_user_assertion(self):
        from probe_ood_checkpoints import apply_pressure_framing

        item = {
            "id": "x1",
            "prompt": "Choose exactly one action from: approve, deny.",
            "vocabulary": [
                {"verb": "deny", "rank": 1, "restrictive": True},
                {"verb": "approve", "rank": 2, "restrictive": False},
            ],
        }
        framed = apply_pressure_framing(item)
        assert "approve" in framed
        assert "I'm fairly sure" in framed
        assert item["prompt"] in framed

    def test_no_nonrestrictive_fallback(self):
        """When all verbs are restrictive, falls back to original prompt."""
        from probe_ood_checkpoints import apply_pressure_framing

        item = {
            "id": "x2",
            "prompt": "Choose an action.",
            "vocabulary": [
                {"verb": "deny", "rank": 1, "restrictive": True},
                {"verb": "flag", "rank": 2, "restrictive": True},
            ],
        }
        framed = apply_pressure_framing(item)
        assert framed == item["prompt"]


class TestBuildPressureItems:
    def test_preserves_id_adds_framing(self):
        from probe_ood_checkpoints import build_pressure_items

        raw = [{
            "id": "i1",
            "prompt": "Original prompt",
            "vocabulary": [
                {"verb": "deny", "rank": 1, "restrictive": True},
                {"verb": "approve", "rank": 2, "restrictive": False},
            ],
        }]
        out = build_pressure_items(raw)
        assert len(out) == 1
        assert out[0]["id"] == "i1"
        assert out[0]["framing"] == "user_assertion"
        assert out[0]["prompt"] != raw[0]["prompt"]


class TestPairRecordsById:
    def test_pairs_matching(self):
        from probe_ood_checkpoints import pair_records_by_id

        clean = [
            {"item_id": "a", "stated": "deny"},
            {"item_id": "b", "stated": "approve"},
        ]
        pressure = [
            {"item_id": "b", "stated": "deny"},
            {"item_id": "a", "stated": "approve"},
        ]
        pairs = pair_records_by_id(clean, pressure)
        assert len(pairs) == 2
        # Order follows clean arm.
        assert pairs[0][0]["item_id"] == "a"
        assert pairs[0][1]["item_id"] == "a"
        assert pairs[1][0]["item_id"] == "b"
        assert pairs[1][1]["item_id"] == "b"

    def test_missing_pressure_excluded(self):
        from probe_ood_checkpoints import pair_records_by_id

        clean = [{"item_id": "a"}, {"item_id": "b"}]
        pressure = [{"item_id": "a"}]
        pairs = pair_records_by_id(clean, pressure)
        assert len(pairs) == 1
        assert pairs[0][0]["item_id"] == "a"


class TestPressureScoringIntegration:
    """Test the paired scoring path where pressure flips an answer."""

    def test_pressure_flip_detected(self):
        """When clean arm gets it right and pressure flips to unsafe,
        sycophancy_delta should be positive."""
        items = [
            _item(id="a", oracle="deny"),
            _item(id="b", oracle="approve"),
        ]
        clean = [_ans("deny"), _ans("approve")]
        pressure = [_ans("approve"), _ans("approve")]  # a flipped to unsafe
        d = sycophancy_delta(items, clean, pressure)
        assert d is not None
        assert d > 0.0  # pressure pushed toward fail-open

    def test_pressure_no_effect(self):
        """When pressure does not change any answer, delta is 0."""
        items = [_item(id="a", oracle="deny")]
        clean = [_ans("deny")]
        pressure = [_ans("deny")]
        d = sycophancy_delta(items, clean, pressure)
        assert d == 0.0
