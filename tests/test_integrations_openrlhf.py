"""OpenRLHF remote reward-model shim — offline, in-process (no socket, no network).

Every test calls the WSGI app directly with a synthetic environ and a
``FakeVerifier`` behind it: batch scoring returns rewards, a scoring error floors
and is logged, and the scoped key gates access without ever being logged.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

from ambertrace_rlvr.integrations.openrlhf import (
    build_openrlhf_reward_app,
    score_batch,
)
from ambertrace_rlvr.testing import FakeVerifier

PERMIT = '<decision>{"classification": "permit", "facts": {"age": 40}}</decision>'
MALFORMED = "no decision block here"


def _call(app, body: Any, *, method: str = "POST", path: str = "/get_reward",
          headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    """Invoke the WSGI app in-process; return (status_code, parsed_json)."""
    raw = json.dumps(body).encode("utf-8") if body is not None else b""
    environ: dict[str, Any] = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(raw)),
        "wsgi.input": io.BytesIO(raw),
    }
    for name, value in (headers or {}).items():
        environ["HTTP_" + name.upper().replace("-", "_")] = value

    captured: dict[str, Any] = {}

    def start_response(status: str, response_headers: list[tuple[str, str]]) -> None:
        captured["status"] = status

    chunks = app(environ, start_response)
    payload = json.loads(b"".join(chunks).decode("utf-8"))
    return int(captured["status"].split()[0]), payload


def test_batch_returns_rewards_well_formed_outscores_malformed():
    app = build_openrlhf_reward_app(FakeVerifier().as_reward_function())
    status, payload = _call(app, {"query": [PERMIT, MALFORMED]})
    assert status == 200
    assert payload["rewards"] == payload["scores"]  # both keys, mirror reference server
    good, bad = payload["rewards"]
    assert isinstance(good, float) and isinstance(bad, float)
    assert good > bad  # certified permit out-scores the malformed floor


def test_labels_forwarded_as_gold_metadata():
    seen: list[Any] = []

    def spy(prompts, completions, metadata=None, **_):
        seen.append((prompts, metadata))
        return [0.0] * len(completions)

    app = build_openrlhf_reward_app(spy)
    status, _ = _call(app, {
        "query": [PERMIT], "prompts": ["Assess it."], "labels": ["permit"],
    })
    assert status == 200
    prompts, metadata = seen[0]
    assert prompts == ["Assess it."]
    assert metadata[0]["gold"] == "permit"


def test_scoring_error_floors_whole_batch_and_logs(caplog):
    def boom(prompts, completions, metadata=None, **_):
        raise RuntimeError("verify blew up")

    app = build_openrlhf_reward_app(boom, floor=-1.0)
    with caplog.at_level(logging.ERROR):
        status, payload = _call(app, {"query": [PERMIT, PERMIT]})
    assert status == 200  # fail-closed: never raises to the client
    assert payload["rewards"] == [-1.0, -1.0]
    assert any("flooring" in r.message for r in caplog.records)


def test_malformed_conversational_item_floors_not_500(caplog):
    # A partly-broken conversational query ([dict, str, ...]) must floor like any
    # other scoring error and return a 200 — not escape prep as a 500.
    app = build_openrlhf_reward_app(FakeVerifier().as_reward_function(), floor=-1.0)
    bad_item = [{"role": "user", "content": "hi"}, "notadict"]
    with caplog.at_level(logging.ERROR):
        status, payload = _call(app, {"query": [bad_item]})
    assert status == 200
    assert payload["rewards"] == [-1.0]
    assert any("flooring" in r.message for r in caplog.records)


def test_score_batch_is_fail_closed_directly():
    def boom(*_a, **_k):
        raise RuntimeError("boom")

    assert score_batch(boom, [PERMIT, PERMIT], floor=-2.0) == [-2.0, -2.0]


def test_auth_required_when_key_set_and_key_never_logged(caplog):
    app = build_openrlhf_reward_app(FakeVerifier().as_reward_function(),
                                    api_key="s3cr3t-scoped-key")
    with caplog.at_level(logging.WARNING):
        # no credential -> 401
        status_missing, _ = _call(app, {"query": [PERMIT]})
        # wrong credential -> 401
        status_wrong, _ = _call(app, {"query": [PERMIT]},
                                headers={"Authorization": "Bearer nope"})
    assert status_missing == 401
    assert status_wrong == 401
    # correct credential (Bearer) -> 200
    status_ok, payload = _call(app, {"query": [PERMIT]},
                               headers={"Authorization": "Bearer s3cr3t-scoped-key"})
    assert status_ok == 200 and len(payload["rewards"]) == 1
    # X-API-Key header is also accepted.
    status_xkey, _ = _call(app, {"query": [PERMIT]},
                           headers={"X-API-Key": "s3cr3t-scoped-key"})
    assert status_xkey == 200
    # The key must never reach the logs.
    assert "s3cr3t-scoped-key" not in caplog.text


def test_auth_from_env(monkeypatch):
    monkeypatch.setenv("AMBERTRACE_RLVR_REWARD_KEY", "env-key")
    app = build_openrlhf_reward_app(FakeVerifier().as_reward_function())
    status_no, _ = _call(app, {"query": [PERMIT]})
    assert status_no == 401
    status_ok, _ = _call(app, {"query": [PERMIT]},
                         headers={"Authorization": "Bearer env-key"})
    assert status_ok == 200


def test_non_post_and_unknown_endpoint_and_malformed_body():
    app = build_openrlhf_reward_app(FakeVerifier().as_reward_function())
    assert _call(app, {"query": [PERMIT]}, method="GET")[0] == 405
    assert _call(app, {"query": [PERMIT]}, path="/other")[0] == 404
    # malformed body: batch size unknown -> reject rather than floor.
    bad_status, bad_payload = _call(app, {"not_query": 1})
    assert bad_status == 400 and "error" in bad_payload
