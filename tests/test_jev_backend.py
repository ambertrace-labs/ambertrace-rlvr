"""Jev typed-decision backend (#123): Choice request shaping, response parsing,
refusal vs error, and API-key resolution. Fully network-free via an injected
transport."""

from __future__ import annotations

from typing import Any

import pytest

from ambertrace_rlvr.model_backend import (
    JevProvider,
    ModelBackendError,
    ScoredDecision,
)


def _respond(answer: dict[str, Any] | None):
    """A transport returning a Jev-shaped response with the given ``action``
    answer, capturing the request for assertions."""
    captured: dict[str, Any] = {}

    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"answers": {} if answer is None else {"action": answer}}

    transport.captured = captured  # type: ignore[attr-defined]
    return transport


def test_decide_shapes_a_choice_question_from_the_label_space():
    t = _respond({"type": "choice", "choice": "deny",
                  "probabilities": {"approve": 0.1, "deny": 0.9}, "confidence": 0.9})
    p = JevProvider(api_key="k", transport=t)
    d = p.decide("Assess this case.", ("approve", "deny"))

    cap = t.captured  # type: ignore[attr-defined]
    assert cap["url"].endswith("/v1/systemone")
    q = cap["payload"]["questions"]["action"]
    assert q["type"] == "choice"
    assert q["criteria"] == {"approve": "approve", "deny": "deny"}
    assert cap["payload"]["state"] == "Assess this case."
    assert d == ScoredDecision(choice="deny",
                               probabilities={"approve": 0.1, "deny": 0.9},
                               confidence=0.9)


def test_choice_outside_label_space_is_dropped_to_a_refusal():
    # A hallucinated label the corpus never offered must not be scored as a verb.
    t = _respond({"choice": "escalate", "probabilities": {"escalate": 1.0}})
    d = JevProvider(api_key="k", transport=t).decide("q", ("approve", "deny"))
    assert d.choice is None and not d.answered


def test_missing_answer_is_a_refusal_not_an_error():
    d = JevProvider(api_key="k", transport=_respond(None)).decide("q", ("approve", "deny"))
    assert d == ScoredDecision(choice=None)


def test_probabilities_are_filtered_to_the_label_space():
    t = _respond({"choice": "approve",
                  "probabilities": {"approve": 0.7, "deny": 0.3, "junk": 0.5}})
    d = JevProvider(api_key="k", transport=t).decide("q", ("approve", "deny"))
    assert d.probabilities == {"approve": 0.7, "deny": 0.3}


def test_transport_failure_raises_model_backend_error():
    def boom(url, payload, timeout):
        raise OSError("connection refused")

    with pytest.raises(ModelBackendError):
        JevProvider(api_key="k", transport=boom).decide("q", ("approve", "deny"))


def test_missing_api_key_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ModelBackendError, match="no Jev API key"):
        JevProvider().decide("q", ("approve", "deny"))


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key")
    # env supplies the key; an explicit api_key= would take precedence.
    assert JevProvider()._resolve_key() == "env-key"
    assert JevProvider(api_key="explicit")._resolve_key() == "explicit"
