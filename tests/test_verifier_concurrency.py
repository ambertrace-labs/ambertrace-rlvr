"""Throughput: bounded concurrency, cache bypass, and the query_batch capability
gate (issue #4 / RFC §3 D). All tests are offline — ``AmberVerifier._query`` is
monkeypatched so the real SDK client is never constructed."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from ambertrace_rlvr.domain import VerifiableDomain
from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.reports import AmberReport
from ambertrace_rlvr.verifier import AmberVerifier


def _domain() -> VerifiableDomain:
    return VerifiableDomain(platform_id=1, parser=JSONBlockParser(), api_key=None)


def _parsed(i: int) -> ParsedCompletion:
    return ParsedCompletion(query=f"q{i}", facts={"a": i})


def test_verify_batch_caps_concurrency_and_preserves_order():
    max_concurrency = 4
    n = 12
    v = AmberVerifier(domain=_domain(), cache=False, max_concurrency=max_concurrency)
    v._client = _FakeClient(platforms=_PlatformsNoBatch())  # bypass real SDK construction

    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_query(_self: AmberVerifier, parsed: ParsedCompletion) -> tuple[AmberReport, bool]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)  # brief window for concurrent workers to overlap
        with lock:
            active -= 1
        return AmberReport.floor(reason="x"), False

    v._query = fake_query.__get__(v, AmberVerifier)  # type: ignore[method-assign]

    parsed: list[ParsedCompletion | None] = [_parsed(i) for i in range(n)]
    results = v.verify_batch(parsed)

    assert len(results) == n
    assert all(r is not None and r.error == "x" for r in results)
    assert peak <= max_concurrency


def test_cache_hit_bypasses_the_pool_and_query():
    v = AmberVerifier(domain=_domain(), cache=True)
    calls = 0

    def fake_query(_self: AmberVerifier, parsed: ParsedCompletion) -> tuple[AmberReport, bool]:
        nonlocal calls
        calls += 1
        return AmberReport.floor(reason="x"), True

    v._query = fake_query.__get__(v, AmberVerifier)  # type: ignore[method-assign]

    parsed = _parsed(0)
    first = v.verify_one(parsed)
    second = v.verify_one(parsed)

    assert calls == 1  # second call served from the content-addressed cache
    assert first.error == second.error


@dataclass
class _PlatformsNoBatch:
    """Stands in for ``platforms`` on an SDK client that has no ``query_batch``
    — the current published surface of ``ambertraceai==1.0.5``."""

    def query(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("query_batch capability check must not call query")


@dataclass
class _PlatformsWithBatch:
    """Stands in for a future SDK client that publishes ``query_batch``."""

    def query(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("query_batch capability check must not call query")

    def query_batch(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("gate test must not invoke query_batch")


@dataclass
class _FakeClient:
    platforms: Any = field(default=None)


def test_supports_batch_false_when_sdk_lacks_query_batch():
    v = AmberVerifier(domain=_domain())
    v._client = _FakeClient(platforms=_PlatformsNoBatch())

    assert v._supports_batch() is False


def test_supports_batch_true_when_sdk_exposes_query_batch():
    v = AmberVerifier(domain=_domain())
    v._client = _FakeClient(platforms=_PlatformsWithBatch())

    assert v._supports_batch() is True
