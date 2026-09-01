"""LM Studio model backend (#58): request shaping, content extraction, refusal vs
error, and the drop-in `as_model` into the eval sweeps. Fully network-free."""

from __future__ import annotations

from typing import Any

import pytest

from ambertrace_rlvr.eval_oracle import JudgmentSpec, LabelSpec, OracleJudgment
from ambertrace_rlvr.model_backend import LMStudioProvider, ModelBackendError
from ambertrace_rlvr.sycophancy import SweepItem, clean_framing, run_sweep


def _ok(content: str):
    """A transport returning an OpenAI-shaped response with the given content,
    capturing the request for assertions."""
    captured: dict[str, Any] = {}

    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    transport.captured = captured  # type: ignore[attr-defined]
    return transport


def test_complete_extracts_content_and_shapes_request():
    t = _ok("permit")
    p = LMStudioProvider(model="some-model", temperature=0.0, transport=t)
    assert p.complete("Assess this.") == "permit"
    cap = t.captured  # type: ignore[attr-defined]
    assert cap["url"].endswith("/v1/chat/completions")
    assert cap["payload"]["model"] == "some-model"
    assert cap["payload"]["temperature"] == 0.0
    assert cap["payload"]["messages"][-1] == {"role": "user", "content": "Assess this."}


def test_system_prompt_is_included_when_set():
    t = _ok("deny")
    p = LMStudioProvider(model="m", system="You are a decision engine.", transport=t)
    p.complete("q")
    assert t.captured["payload"]["messages"][0]["role"] == "system"  # type: ignore[attr-defined]


def test_empty_or_malformed_response_is_a_refusal_not_error():
    empty = LMStudioProvider(model="m", transport=lambda u, p, t: {"choices": []})
    assert empty.complete("q") == ""
    odd = LMStudioProvider(model="m", transport=lambda u, p, t: {"unexpected": 1})
    assert odd.complete("q") == ""


def test_system_role_error_folds_into_user_and_retries():
    # A template that rejects a system message on the first call, succeeds on the
    # folded retry — the matrix must run uniformly across such models.
    calls: list[list[dict]] = []

    def transport(url, payload, timeout):
        calls.append(payload["messages"])
        roles = [m["role"] for m in payload["messages"]]
        if "system" in roles:
            return {"error": "Only user and assistant roles are supported!"}
        return {"choices": [{"message": {"content": "deny"}}]}

    p = LMStudioProvider(model="m", system="Decide.", transport=transport)
    assert p.complete("case X") == "deny"
    assert len(calls) == 2                       # first (system) failed, retry folded
    assert [m["role"] for m in calls[1]] == ["user"]
    assert "Decide." in calls[1][0]["content"] and "case X" in calls[1][0]["content"]


def test_extra_body_is_merged_into_payload():
    # reasoning controls (e.g. reasoning_effort) ride in extra_body so a thinking
    # model can be told to answer directly instead of burning its budget on <think>.
    captured: dict[str, Any] = {}

    def transport(url, payload, timeout):
        captured.update(payload)
        return {"choices": [{"message": {"content": "deny"}}]}

    p = LMStudioProvider(model="m", transport=transport,
                         extra_body={"reasoning_effort": "none"})
    assert p.complete("case") == "deny"
    assert captured["reasoning_effort"] == "none"


def test_connection_failure_raises_model_backend_error():
    def boom(url, payload, timeout):
        raise OSError("connection refused")

    p = LMStudioProvider(model="m", transport=boom)
    with pytest.raises(ModelBackendError):
        p.complete("q")


def test_backend_error_from_transport_is_preserved():
    def boom(url, payload, timeout):
        raise ModelBackendError("cannot reach model server")

    with pytest.raises(ModelBackendError, match="cannot reach"):
        LMStudioProvider(model="m", transport=boom).complete("q")


def test_as_model_drops_into_the_eval_sweep():
    # The provider's as_model() is the prompt->completion callable run_sweep wants.
    spec = JudgmentSpec(labels=[LabelSpec("deny", 0, True), LabelSpec("permit", 1)])
    provider = LMStudioProvider(model="m", transport=_ok("deny"))
    items = [SweepItem(prompt="Assess A.", label_space=("permit", "deny"))]
    judgments = [OracleJudgment(certified=True, certified_undecidable=False,
                                value="deny", reason=None)]
    rep = run_sweep(items, judgments, {"clean": clean_framing()},
                    provider.as_model(), spec, policy_in_context=True)
    # model answered the oracle-correct "deny" -> scored correct, no fail-open.
    assert rep.arms["clean"].report.correct == 1
    assert rep.arms["clean"].report.over_permit == 0
