"""Verifier resilience: retries, backoff, circuit-breaker → floor.

All tests are offline: the SDK's real network client is never constructed —
we inject a fake ``platforms.query`` directly via ``verifier._client``, and a
fake sleep/monotonic clock so the suite adds no wall-clock delay and can
advance the breaker's cooldown deterministically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import ambertraceai
import pytest

from ambertrace_rlvr.domain import VerifiableDomain
from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.testing import make_query_result
from ambertrace_rlvr.verifier import AmberVerifier


@dataclass
class _FakePlatforms:
    """Stands in for ``AmbertraceAPI().platforms`` — records calls, raises or
    returns per a scripted sequence of outcomes."""

    outcomes: list[Any] = field(default_factory=list)
    calls: int = field(default=0, init=False)

    def query(self, *_args: Any, **_kwargs: Any) -> Any:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class _FakeClient:
    platforms: _FakePlatforms


class _FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _domain(api_key: str | None = None) -> VerifiableDomain:
    return VerifiableDomain(platform_id=1, parser=JSONBlockParser(), api_key=api_key)


def _parsed() -> ParsedCompletion:
    return ParsedCompletion(query="q", facts={"a": 1})


def _verifier(**kwargs: Any) -> AmberVerifier:
    return AmberVerifier(domain=_domain(), cache=False, **kwargs)


def _wire(v: AmberVerifier, outcomes: list[Any]) -> _FakePlatforms:
    platforms = _FakePlatforms(outcomes=outcomes)
    v._client = _FakeClient(platforms=platforms)  # bypass real SDK construction
    return platforms


def _no_sleep_calls() -> list[float]:
    return []


class TimeoutError_(Exception):
    """A generic transient error — stands in for a network/timeout/5xx failure."""


def test_transient_then_success_retries_and_returns_certified_report():
    v = _verifier(max_retries=2)
    sleeps: list[float] = []
    v._sleep = sleeps.append
    ok = make_query_result(decision="permit")
    platforms = _wire(v, [TimeoutError_("boom"), ok])

    report = v.verify_one(_parsed())

    assert platforms.calls == 2
    assert report.proof_checked is True
    assert report.decision == "permit"
    # backoff was computed but never actually slept in the test
    assert len(sleeps) == 1
    assert sleeps[0] >= 0


def test_retries_exhausted_returns_floor_never_raises(caplog: pytest.LogCaptureFixture):
    v = _verifier(max_retries=2)
    v._sleep = lambda _delay: None
    _wire(v, [TimeoutError_("a"), TimeoutError_("b"), TimeoutError_("c")])

    with caplog.at_level("INFO"):
        report = v.verify_one(_parsed())

    assert report.proof_checked is False
    assert report.error is not None and "verifier_error" in report.error


def test_reward_function_never_raises_on_persistent_outage():
    v = _verifier(max_retries=1)
    v._sleep = lambda _delay: None
    _wire(v, [TimeoutError_("a"), TimeoutError_("b")])
    reward_fn = v.as_reward_function()

    rewards = reward_fn(
        ["prompt"], ['<decision>{"classification": "x", "facts": {"a": 1}}</decision>'],
    )

    # Never raises; the report is fail-closed (uncertified), so the shaper's
    # certified/correctness/graded components are all zero — only the "the
    # completion parsed" format credit survives, well below a clean certified score.
    assert len(rewards) == 1
    assert v.floor <= rewards[0] < 1.0


def test_threshold_consecutive_failures_opens_breaker_without_sdk_call():
    v = _verifier(max_retries=0, breaker_threshold=3)
    v._sleep = lambda _delay: None
    clock = _FakeClock()
    v._monotonic = clock
    platforms = _wire(v, [TimeoutError_("x")] * 3)

    for _ in range(3):
        report = v.verify_one(_parsed())
        assert report.error is not None and "verifier_error" in report.error
    assert platforms.calls == 3

    # breaker is now open: the next call must not reach the SDK at all
    report = v.verify_one(_parsed())
    assert report.error == "circuit_open"
    assert platforms.calls == 3  # unchanged


def test_breaker_open_emits_single_warning_at_transition(
    caplog: pytest.LogCaptureFixture,
):
    v = _verifier(max_retries=0, breaker_threshold=3)
    v._sleep = lambda _delay: None
    clock = _FakeClock()
    v._monotonic = clock
    _wire(v, [TimeoutError_("x")] * 4)

    with caplog.at_level("WARNING"):
        for _ in range(2):  # below threshold: no WARNING yet
            v.verify_one(_parsed())
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

        v.verify_one(_parsed())  # 3rd consecutive failure crosses threshold → OPEN

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "OPEN" in warnings[0].getMessage()


def test_breaker_recovers_after_cooldown_half_open_success_closes():
    v = _verifier(max_retries=0, breaker_threshold=2, breaker_cooldown=30.0)
    v._sleep = lambda _delay: None
    clock = _FakeClock()
    v._monotonic = clock
    ok = make_query_result(decision="permit")
    platforms = _wire(v, [TimeoutError_("a"), TimeoutError_("b"), ok])

    for _ in range(2):
        v.verify_one(_parsed())
    assert v.verify_one(_parsed()).error == "circuit_open"
    assert platforms.calls == 2  # breaker call did not reach the SDK

    clock.advance(30.0)  # cooldown elapsed → half-open trial allowed
    report = v.verify_one(_parsed())

    assert platforms.calls == 3
    assert report.proof_checked is True
    # breaker is closed again: consecutive-failure count reset
    assert v._consecutive_failures == 0

    # a subsequent transient failure must be able to retrip the breaker
    _wire(v, [TimeoutError_("c"), TimeoutError_("d")])
    for _ in range(2):
        v.verify_one(_parsed())
    assert v.verify_one(_parsed()).error == "circuit_open"


def test_ambertrace_error_deny_does_not_retry_and_does_not_trip_breaker():
    v = _verifier(max_retries=2, breaker_threshold=2)
    deny = ambertraceai.AmbertraceError(422, "gate_denied", "no dice")
    platforms = _wire(v, [deny])

    report = v.verify_one(_parsed())

    assert platforms.calls == 1  # no retry
    assert report.proof_checked is False
    assert v._consecutive_failures == 0  # deny resets, never trips breaker

    # confirm the breaker really never tripped: two more denies in a row, still no retry/breaker
    _wire(v, [deny])
    v.verify_one(_parsed())
    assert v._consecutive_failures == 0


def test_api_key_never_appears_in_logged_message_or_floor_reason(
    caplog: pytest.LogCaptureFixture,
):
    secret = "sk-super-secret-key"
    v = AmberVerifier(domain=_domain(api_key=secret), cache=False, max_retries=1)
    v._sleep = lambda _delay: None
    _wire(v, [TimeoutError_(f"upstream said: {secret}"),
              TimeoutError_(f"upstream said again: {secret}")])

    with caplog.at_level("INFO"):
        report = v.verify_one(_parsed())

    assert report.error is not None
    assert secret not in report.error
    # Render each record the way a real handler would — Formatter.format renders
    # exc_info (message + traceback) too, which getMessage() omits. The secret
    # must be absent from the FULLY formatted output, not just the bare message.
    formatter = logging.Formatter()
    for record in caplog.records:
        assert secret not in formatter.format(record)


def test_transient_failure_floor_is_not_cached():
    v = AmberVerifier(domain=_domain(), cache=True, max_retries=0)
    v._sleep = lambda _delay: None
    ok = make_query_result(decision="permit")
    platforms = _wire(v, [TimeoutError_("a"), ok])

    parsed = _parsed()
    first = v.verify_one(parsed)
    assert first.proof_checked is False  # transient floor, not cached

    second = v.verify_one(parsed)
    assert platforms.calls == 2  # second call actually hit the (fake) SDK again
    assert second.proof_checked is True


def test_ambertrace_error_deny_is_still_cached():
    v = AmberVerifier(domain=_domain(), cache=True, max_retries=1)
    deny = ambertraceai.AmbertraceError(422, "gate_denied", "no dice")
    platforms = _wire(v, [deny])

    parsed = _parsed()
    first = v.verify_one(parsed)
    second = v.verify_one(parsed)

    assert platforms.calls == 1  # second hit the cache, no second SDK call
    assert first.error == second.error
