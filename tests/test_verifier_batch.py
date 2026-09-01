"""Tests for ``query_batch`` path, projection, chunking, per-item error
handling, and capability-gate fallback (issue #27).

All tests are offline: the SDK's real network client is never constructed.
Stubs stand in for ``AmbertraceAPI().platforms`` and ``ambertraceai.AmbertraceError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from ambertrace_rlvr.domain import VerifiableDomain
from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.testing import make_query_result
from ambertrace_rlvr.verifier import REWARD_PROJECTION, AmberVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain() -> VerifiableDomain:
    return VerifiableDomain(platform_id=1, parser=JSONBlockParser(), api_key=None)


def _parsed(i: int) -> ParsedCompletion:
    return ParsedCompletion(query=f"q{i}", facts={"a": i})


# ---------------------------------------------------------------------------
# Fake SDK surfaces
# ---------------------------------------------------------------------------

@dataclass
class _FakePlatformsBatch:
    """Stands in for ``AmbertraceAPI().platforms`` with ``query_batch`` support.
    Records calls and returns scripted outcomes."""

    batch_results: list[Any] = field(default_factory=list)
    batch_calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    _call_idx: int = field(default=0, init=False)
    raise_on_batch: Exception | None = field(default=None)

    def query(self, *_args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Per-item query — should not be called when batch is available."""
        raise AssertionError("query() called when query_batch is available")

    def query_batch(self, platform_id: int, *, queries: list[dict[str, Any]],
                    **kwargs: Any) -> dict[str, Any]:
        self.batch_calls.append({
            "platform_id": platform_id,
            "queries": queries,
            **kwargs,
        })
        if self.raise_on_batch is not None:
            raise self.raise_on_batch
        idx = self._call_idx
        self._call_idx += 1
        if idx < len(self.batch_results):
            return self.batch_results[idx]
        # Default: all items ok with a permit result.
        return {
            "platform_id": platform_id,
            "results": [
                {"index": i, "status": "ok", "data": make_query_result(decision="permit")}
                for i in range(len(queries))
            ],
        }


@dataclass
class _FakePlatformsNoBatch:
    """SDK without query_batch — fallback path."""
    query_calls: int = field(default=0, init=False)

    def query(self, *_args: Any, **_kwargs: Any) -> Any:
        self.query_calls += 1
        return make_query_result(decision="permit")


@dataclass
class _FakeClient:
    platforms: Any = field(default=None)


def _verifier(**kwargs: Any) -> AmberVerifier:
    return AmberVerifier(domain=_domain(), cache=False, **kwargs)


def _wire_batch(v: AmberVerifier, platforms: _FakePlatformsBatch) -> None:
    v._client = _FakeClient(platforms=platforms)


def _wire_no_batch(v: AmberVerifier, platforms: _FakePlatformsNoBatch) -> None:
    v._client = _FakeClient(platforms=platforms)


# ---------------------------------------------------------------------------
# Chunking: 51 items -> 2 batch calls
# ---------------------------------------------------------------------------

class TestChunking:
    def test_51_items_produces_two_batch_calls(self):
        v = _verifier()
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch()
        _wire_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(i) for i in range(51)]
        results = v.verify_batch(parsed)

        assert len(results) == 51
        assert len(platforms.batch_calls) == 2
        # First chunk: 50 items, second chunk: 1 item.
        assert len(platforms.batch_calls[0]["queries"]) == 50
        assert len(platforms.batch_calls[1]["queries"]) == 1

    def test_50_items_produces_one_batch_call(self):
        v = _verifier()
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch()
        _wire_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(i) for i in range(50)]
        results = v.verify_batch(parsed)

        assert len(results) == 50
        assert len(platforms.batch_calls) == 1

    def test_nones_in_parsed_are_skipped(self):
        v = _verifier()
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch()
        _wire_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(0), None, _parsed(2)]
        results = v.verify_batch(parsed)

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None  # None in -> None out
        assert results[2] is not None
        # Only 2 items were sent in one batch call.
        assert len(platforms.batch_calls) == 1
        assert len(platforms.batch_calls[0]["queries"]) == 2


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------

class TestOrderPreservation:
    def test_results_match_input_order(self):
        v = _verifier()
        v._sleep = lambda _: None
        n = 5
        # Each item returns a unique decision so we can verify ordering.
        per_item_results = [
            {"index": i, "status": "ok", "data": make_query_result(decision=f"d{i}")}
            for i in range(n)
        ]
        platforms = _FakePlatformsBatch(batch_results=[
            {"platform_id": 1, "results": per_item_results},
        ])
        _wire_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(i) for i in range(n)]
        results = v.verify_batch(parsed)

        for i in range(n):
            assert results[i] is not None
            assert results[i].decision == f"d{i}"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Per-item error handling
# ---------------------------------------------------------------------------

class TestPerItemErrors:
    def test_deny_error_produces_from_error_report_cacheable(self):
        """A per-item ``status: "error"`` with a structured code (certification
        deny) -> ``AmberReport.from_error`` (cacheable)."""
        v = AmberVerifier(domain=_domain(), cache=True)
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch(batch_results=[{
            "platform_id": 1,
            "results": [
                {"index": 0, "status": "error", "error": {
                    "code": "gate_denied", "message": "denied",
                    "status_code": 422,
                }},
            ],
        }])
        _wire_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(0)]
        results = v.verify_batch(parsed)

        assert results[0] is not None
        report = results[0]
        assert report.proof_checked is False
        assert report.error is not None
        assert "denied" in report.error

    def test_transient_coded_error_floors_and_is_not_cached(self):
        """A per-item error with a *transient* code (rate limit, unavailable,
        internal) must floor without caching — caching it as a deny would pin
        the floor for the rest of a training run."""
        for code, status in (("rate_limited", 429), ("service_unavailable", 503),
                             ("internal_error", 500)):
            v = AmberVerifier(domain=_domain(), cache=True)
            v._sleep = lambda _: None
            platforms = _FakePlatformsBatch(batch_results=[{
                "platform_id": 1,
                "results": [
                    {"index": 0, "status": "error", "error": {
                        "code": code, "message": "try again later",
                        "status_code": status,
                    }},
                ],
            }])
            _wire_batch(v, platforms)

            results = v.verify_batch([_parsed(0)])
            report = results[0]
            assert report is not None
            assert report.proof_checked is False
            assert report.error is not None and "batch_item_error" in report.error, code
            # not cached: a retry must reach the API again
            with v._lock:
                assert not v._cache, code

    def test_unstructured_error_produces_floor_not_cacheable(self):
        """A per-item ``status: "error"`` without a code -> floor (not cacheable)."""
        v = _verifier()
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch(batch_results=[{
            "platform_id": 1,
            "results": [
                {"index": 0, "status": "error", "error": {
                    "message": "something went wrong",
                }},
            ],
        }])
        _wire_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(0)]
        results = v.verify_batch(parsed)

        assert results[0] is not None
        report = results[0]
        assert report.error is not None
        assert "batch_item_error" in report.error

    def test_mixed_ok_and_error_does_not_fail_batch(self):
        """One bad row does not fail the batch."""
        v = _verifier()
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch(batch_results=[{
            "platform_id": 1,
            "results": [
                {"index": 0, "status": "ok", "data": make_query_result(decision="permit")},
                {"index": 1, "status": "error", "error": {
                    "code": "gate_denied", "message": "denied", "status_code": 422,
                }},
                {"index": 2, "status": "ok", "data": make_query_result(decision="deny")},
            ],
        }])
        _wire_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(i) for i in range(3)]
        results = v.verify_batch(parsed)

        assert results[0] is not None and results[0].decision == "permit"
        assert results[1] is not None and results[1].error is not None
        assert results[2] is not None and results[2].decision == "deny"


# ---------------------------------------------------------------------------
# Breaker behaviour on batch failure
# ---------------------------------------------------------------------------

class TestBreakerOnBatch:
    def test_batch_transport_failure_trips_breaker(self):
        v = _verifier(max_retries=0, breaker_threshold=2)
        v._sleep = lambda _: None

        class _FakeClock:
            def __init__(self) -> None:
                self.now = 0.0
            def __call__(self) -> float:
                return self.now

        clock = _FakeClock()
        v._monotonic = clock

        # Two batch-level transport failures should trip the breaker.
        for _ in range(2):
            platforms = _FakePlatformsBatch(raise_on_batch=RuntimeError("network down"))
            _wire_batch(v, platforms)
            v.verify_batch([_parsed(0)])

        # Next call should be circuit-open without reaching SDK.
        platforms_clean = _FakePlatformsBatch()
        _wire_batch(v, platforms_clean)
        results = v.verify_batch([_parsed(0)])

        assert results[0] is not None
        assert results[0].error == "circuit_open"
        assert len(platforms_clean.batch_calls) == 0  # SDK never called

    def test_batch_transport_failure_retries_then_floors(self):
        v = _verifier(max_retries=2)
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch(raise_on_batch=RuntimeError("timeout"))
        _wire_batch(v, platforms)

        results = v.verify_batch([_parsed(0)])

        assert results[0] is not None
        assert results[0].error is not None
        assert "verifier_error" in results[0].error
        # Should have made 3 attempts (1 + 2 retries).
        assert len(platforms.batch_calls) == 3


# ---------------------------------------------------------------------------
# Capability-gate fallback
# ---------------------------------------------------------------------------

class TestCapabilityGateFallback:
    def test_fallback_to_per_item_when_no_query_batch(self):
        v = _verifier()
        v._sleep = lambda _: None
        platforms = _FakePlatformsNoBatch()
        _wire_no_batch(v, platforms)

        parsed: list[ParsedCompletion | None] = [_parsed(i) for i in range(3)]
        results = v.verify_batch(parsed)

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert platforms.query_calls == 3

    def test_fallback_logs_once(self, caplog: pytest.LogCaptureFixture):
        v = _verifier()
        v._sleep = lambda _: None
        platforms = _FakePlatformsNoBatch()
        _wire_no_batch(v, platforms)

        with caplog.at_level(logging.DEBUG):
            v.verify_batch([_parsed(0)])
            v.verify_batch([_parsed(1)])

        debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        no_batch_msgs = [m for m in debug_msgs if "no query_batch" in m]
        assert len(no_batch_msgs) == 1  # logged once


# ---------------------------------------------------------------------------
# Projection field-list correctness
# ---------------------------------------------------------------------------

class TestProjection:
    def test_reward_projection_matches_reports_consumption(self):
        """REWARD_PROJECTION must contain all fields from_query_result reads."""
        # from_query_result reads: proof_checked, answer, decision,
        # proof_summary, explanation (which wraps confidence, rules, etc.)
        required = {"proof_checked", "answer", "decision", "proof_summary", "explanation"}
        assert required == set(REWARD_PROJECTION)

    def test_projection_sent_when_enabled_and_batch(self):
        v = _verifier(use_projection=True)
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch()
        _wire_batch(v, platforms)

        v.verify_batch([_parsed(0)])

        assert len(platforms.batch_calls) == 1
        call = platforms.batch_calls[0]
        assert "projection" in call
        assert set(call["projection"]) == set(REWARD_PROJECTION)

    def test_projection_not_sent_when_disabled(self):
        v = _verifier(use_projection=False)
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch()
        _wire_batch(v, platforms)

        v.verify_batch([_parsed(0)])

        assert len(platforms.batch_calls) == 1
        call = platforms.batch_calls[0]
        assert "projection" not in call


# ---------------------------------------------------------------------------
# Cache interaction with batch path
# ---------------------------------------------------------------------------

class TestBatchCache:
    def test_cache_dedupes_within_batch(self):
        """Two identical items in the same batch both start as cache misses
        (dedup within a single batch dispatch is not done — the API handles
        it fine). But after the batch returns, the cache is populated, so
        duplicates in a *subsequent* batch are served from cache."""
        v = AmberVerifier(domain=_domain(), cache=True)
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch()
        _wire_batch(v, platforms)

        # Both items are sent (no pre-batch dedup).
        parsed: list[ParsedCompletion | None] = [_parsed(0), _parsed(0)]
        results = v.verify_batch(parsed)

        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None
        assert len(platforms.batch_calls) == 1
        assert len(platforms.batch_calls[0]["queries"]) == 2  # both sent

    def test_successful_batch_items_are_cached(self):
        v = AmberVerifier(domain=_domain(), cache=True)
        v._sleep = lambda _: None
        platforms = _FakePlatformsBatch()
        _wire_batch(v, platforms)

        v.verify_batch([_parsed(0)])
        assert len(platforms.batch_calls) == 1

        # Second batch should hit cache.
        v.verify_batch([_parsed(0)])
        # No additional batch call — served from cache.
        assert len(platforms.batch_calls) == 1
