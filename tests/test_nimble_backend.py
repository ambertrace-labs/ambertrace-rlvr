"""Nimble typed-decision backend (#123): enum-schema shaping, result parsing,
refusal vs error. Network-free / model-free via an injected fake scorer."""

from __future__ import annotations

from typing import Any

import pytest

from ambertrace_rlvr.model_backend import (
    ModelBackendError,
    NimbleProvider,
    ScoredDecision,
)


class _FakeScorer:
    """Stand in for nimble.scoring's ParallelScorer: return a scripted result and
    capture the request for assertions."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.seen: dict[str, Any] = {}

    def score(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.seen["text"] = text
        self.seen["schema"] = schema
        return self.result


def test_decide_shapes_an_enum_schema_and_reads_output_and_scores():
    scorer = _FakeScorer({
        "output": {"action": "deny"},
        "fields": {"action": {"scores": {"approve": 0.2, "deny": 0.8}}},
    })
    d = NimbleProvider(scorer=scorer).decide("Assess this case.", ("approve", "deny"))

    assert scorer.seen["text"] == "Assess this case."
    field = scorer.seen["schema"]["action"]
    assert field["type"] == "enum" and field["choices"] == ["approve", "deny"]
    assert d == ScoredDecision(choice="deny",
                               probabilities={"approve": 0.2, "deny": 0.8},
                               confidence=0.8)


def test_flat_result_without_output_wrapper_is_accepted():
    scorer = _FakeScorer({"action": "approve"})
    d = NimbleProvider(scorer=scorer).decide("q", ("approve", "deny"))
    assert d.choice == "approve" and d.probabilities == {}


def test_choice_outside_label_space_is_a_refusal():
    scorer = _FakeScorer({"output": {"action": "escalate"}})
    d = NimbleProvider(scorer=scorer).decide("q", ("approve", "deny"))
    assert d.choice is None and not d.answered


def test_scores_are_filtered_to_the_label_space():
    scorer = _FakeScorer({
        "output": {"action": "approve"},
        "fields": {"action": {"scores": {"approve": 0.7, "deny": 0.3, "junk": 0.9}}},
    })
    d = NimbleProvider(scorer=scorer).decide("q", ("approve", "deny"))
    assert d.probabilities == {"approve": 0.7, "deny": 0.3}


def test_scorer_failure_raises_model_backend_error():
    class _Boom:
        def score(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("mlx kernel died")

    with pytest.raises(ModelBackendError, match="Nimble scorer failed"):
        NimbleProvider(scorer=_Boom()).decide("q", ("approve", "deny"))
