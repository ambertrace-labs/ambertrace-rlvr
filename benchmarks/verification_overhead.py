"""Offline benchmark: verification wall-clock as a fraction of a training step.

RL post-training issues ``group_size * batch`` verifications per step (spec
section 10), so the verifier must not become the bottleneck. This harness stubs
``AmberVerifier._query`` (and optionally ``_query_batch_chunk``) with a
configurable sleep (standing in for a real SDK round-trip) and runs a batch
through the existing bounded-concurrency pool, then compares the measured verify
time to a simulated step time.

Real wall-clock timing (not mocked), but no network I/O -- this is a script, not
a test, and is not collected by pytest (``testpaths = ["tests"]``).

With ``--batch-path`` the harness exercises the ``query_batch`` chunk path
(issue #27, SDK >= 2.1.3).  Without it the per-item ``ThreadPoolExecutor`` pool
is measured (the fallback path when the SDK lacks ``query_batch``).

Usage:
    python benchmarks/verification_overhead.py
    python benchmarks/verification_overhead.py --batch-path
    python benchmarks/verification_overhead.py --batch 32 --group-size 8 \\
        --concurrency 16 --query-latency 0.05 --step-compute 2.0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from ambertrace_rlvr.domain import VerifiableDomain
from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.reports import AmberReport
from ambertrace_rlvr.verifier import AmberVerifier


@dataclass
class _StubPlatformsNoBatch:
    """Stands in for ``platforms`` on an SDK client that has no ``query_batch``
    so the capability gate falls back to per-item."""

    def query(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("should not be called; _query is monkeypatched")


@dataclass
class _StubPlatformsBatch:
    """Stands in for ``platforms`` on an SDK >= 2.1.3 client that exposes
    ``query_batch``.  The actual call is handled by the monkeypatched
    ``_query_batch_chunk``."""

    def query(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("should not be called when batch path is active")

    def query_batch(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("should not be called; _query_batch_chunk is monkeypatched")


@dataclass
class _StubClient:
    platforms: Any


@dataclass
class BenchmarkResult:
    n_queries: int
    verify_seconds: float
    step_seconds: float
    overhead_pct: float
    batch_path: bool


def run_benchmark(*, batch: int, group_size: int, concurrency: int,
                   query_latency: float, step_compute: float,
                   batch_path: bool = False) -> BenchmarkResult:
    """Verify ``batch * group_size`` synthetic completions at bounded
    ``concurrency``, each simulated SDK call taking ``query_latency`` seconds,
    and compare to a step whose non-verification compute takes
    ``step_compute`` seconds."""
    domain = VerifiableDomain(platform_id=1, parser=JSONBlockParser(), api_key=None)
    verifier = AmberVerifier(domain=domain, cache=False, max_concurrency=concurrency)

    if batch_path:
        verifier._client = _StubClient(platforms=_StubPlatformsBatch())

        def fake_batch_chunk(
            _self: AmberVerifier,
            items: list[tuple[int, ParsedCompletion]],
        ) -> list[tuple[int, AmberReport, bool]]:
            time.sleep(query_latency)  # one call per chunk
            return [
                (idx, AmberReport.floor(reason="benchmark"), False)
                for idx, _ in items
            ]

        verifier._query_batch_chunk = fake_batch_chunk.__get__(verifier, AmberVerifier)  # type: ignore[method-assign]
    else:
        verifier._client = _StubClient(platforms=_StubPlatformsNoBatch())

        def fake_query(_self: AmberVerifier, _parsed: ParsedCompletion) -> tuple[AmberReport, bool]:
            time.sleep(query_latency)
            return AmberReport.floor(reason="benchmark"), False

        verifier._query = fake_query.__get__(verifier, AmberVerifier)  # type: ignore[method-assign]

    n = batch * group_size
    parsed: list[ParsedCompletion | None] = [
        ParsedCompletion(query=f"q{i}", facts={"a": i}) for i in range(n)
    ]

    start = time.perf_counter()
    verifier.verify_batch(parsed)
    verify_seconds = time.perf_counter() - start

    step_seconds = step_compute + verify_seconds
    overhead_pct = 100.0 * verify_seconds / step_seconds if step_seconds > 0 else 0.0
    return BenchmarkResult(
        n_queries=n, verify_seconds=verify_seconds,
        step_seconds=step_seconds, overhead_pct=overhead_pct,
        batch_path=batch_path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=32, help="prompts per step")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt (GRPO)")
    p.add_argument("--concurrency", type=int, default=16, help="max_concurrency of the pool")
    p.add_argument("--query-latency", type=float, default=0.05,
                   help="simulated seconds per SDK query call")
    p.add_argument("--step-compute", type=float, default=2.0,
                   help="simulated non-verification step wall-clock (forward/backward), seconds")
    p.add_argument("--batch-path", action="store_true",
                   help="exercise the query_batch chunk path instead of per-item")
    args = p.parse_args()

    result = run_benchmark(
        batch=args.batch, group_size=args.group_size, concurrency=args.concurrency,
        query_latency=args.query_latency, step_compute=args.step_compute,
        batch_path=args.batch_path,
    )

    path_label = "query_batch (<=50/chunk)" if result.batch_path else "per-item ThreadPool"
    print(f"path:                 {path_label}")
    print(f"queries per step:     {result.n_queries} "
          f"(batch={args.batch} x group_size={args.group_size})")
    print(f"max_concurrency:      {args.concurrency}")
    print(f"simulated query lat:  {args.query_latency:.3f}s")
    print(f"verify wall-clock:    {result.verify_seconds:.3f}s")
    print(f"simulated step time:  {result.step_seconds:.3f}s "
          f"(step_compute={args.step_compute:.3f}s + verify)")
    print(f"verification overhead: {result.overhead_pct:.1f}%  (target < ~15%, spec section 10)")


if __name__ == "__main__":
    main()
