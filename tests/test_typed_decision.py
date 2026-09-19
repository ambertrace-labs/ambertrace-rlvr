"""Typed-decision runner + calibration (#123): adapting a TypedDecider's scored
decisions into the alignment scorer, fail-closed on a raising decider, Brier/ECE
calibration, and the chat-model text_decider adapter. Network-free."""

from __future__ import annotations

from collections.abc import Sequence

from ambertrace_rlvr import calibration, run_typed_model, score_alignment, text_decider
from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.model_backend import ScoredDecision

VOCAB = (LabelSpec("deny", 1, True), LabelSpec("approve", 2))


def _item(item_id: str, oracle: str) -> DecisionItem:
    return DecisionItem(
        id=item_id, domain="d0", prompt=f"case {item_id}", vocabulary=VOCAB,
        oracle=oracle, difficulty={"structure": "baseline"})


class _ScriptedDecider:
    """Returns a pre-scripted ScoredDecision per prompt; raises for a sentinel."""

    def __init__(self, by_prompt: dict[str, ScoredDecision]) -> None:
        self._by_prompt = by_prompt

    def decide(self, prompt: str, label_space: Sequence[str]) -> ScoredDecision:
        if prompt not in self._by_prompt:
            raise RuntimeError("backend down")
        return self._by_prompt[prompt]


def test_run_typed_model_parses_choices_and_feeds_the_scorer():
    items = [_item("a", "deny"), _item("b", "approve")]
    decider = _ScriptedDecider({
        "case a": ScoredDecision("deny", {"deny": 0.9, "approve": 0.1}, 0.9),
        "case b": ScoredDecision("approve", {"deny": 0.2, "approve": 0.8}, 0.8),
    })
    answers, decisions = run_typed_model(items, decider)
    assert [a.value for a in answers] == ["deny", "approve"]
    row = score_alignment(items, answers, model="jev", min_parsed=1)
    assert row.report.correct == 2 and row.report.over_permit == 0


def test_a_raising_decider_is_a_refusal_not_a_crash():
    # "case b" is not scripted -> the decider raises -> that item is a refusal,
    # and the run still completes (fail-closed, like matrix.run_model).
    items = [_item("a", "deny"), _item("b", "approve")]
    decider = _ScriptedDecider({"case a": ScoredDecision("deny", {"deny": 1.0}, 1.0)})
    answers, decisions = run_typed_model(items, decider)
    assert answers[0].value == "deny"
    assert answers[1].answered is False           # refusal, not a wrong label
    assert decisions[1].choice is None


def test_calibration_scores_brier_and_ece():
    items = [_item("a", "deny"), _item("b", "approve")]
    # a: correct, perfectly confident -> 0 error. b: wrong (says deny), conf 1.0.
    decisions = [
        ScoredDecision("deny", {"deny": 1.0, "approve": 0.0}, 1.0),
        ScoredDecision("deny", {"deny": 1.0, "approve": 0.0}, 1.0),
    ]
    cal = calibration(items, decisions)
    assert cal.n == 2
    # Brier: item a 0.0; item b (1-0)^2 + (0-1)^2 = 2.0 -> mean 1.0.
    assert cal.brier == 1.0
    # Both answered with confidence 1.0 but only half correct -> ECE = |1.0-0.5|.
    assert cal.ece == 0.5


def test_calibration_skips_items_without_probabilities():
    items = [_item("a", "deny")]
    cal = calibration(items, [ScoredDecision("deny", {}, None)])
    assert cal.n == 0 and cal.brier is None and cal.ece is None


def test_text_decider_wraps_a_chat_model_without_probabilities():
    items = [_item("a", "deny")]
    decider = text_decider(lambda prompt: "the answer is deny")
    answers, decisions = run_typed_model(items, decider)
    assert answers[0].value == "deny"
    assert decisions[0].probabilities == {}       # chat model -> no calibration signal
