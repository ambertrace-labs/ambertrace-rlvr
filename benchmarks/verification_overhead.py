"""Offline benchmark: verification wall-clock as a fraction of a training step.

RL post-training issues ``group_size * batch`` verifications per step (spec
§10), so the verifier must not become the bottleneck. This harness stubs
``AmberVerifier._query`` with a configurable sleep (standing in for a real SDK
round-trip) and runs a batch through the existing bounded-concurrency pool,
then compares the measured verify time to a simulated step time.

Real wall-clock timing (not mocked), but no network I/O — this is a script, not
a test, and is not collected by pytest (``testpaths = ["tests"]``).

Throughput asks beyond the current per-item pool (``query_batch``, a compact
``query`` projection) are gated on the platform — see issue #27. This harness
measures the pool as it exists today; it does not exercise a batch path.

Usage:
    python benchmarks/verification_overhead.py
    python benchmarks/verification_overhead.py --batch 32 --group-size 8 \\
        --concurrency 16 --query-latency 0.05 --step-compute 2.0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from ambertrace_rlvr.domain import VerifiableDomain
from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.reports import AmberReport
from ambertrace_rlvr.verifier import AmberVerifier


@dataclass
class _StubPlatforms:
    """Stands in for ``AmbertraceAPI().platforms`` on the current SDK surface
    (no ``query_batch``) so the capability gate check doesn't construct a real,
    key-requiring client."""


@dataclass
class _StubClient:
    platforms: _StubPlatforms


@dataclass
class BenchmarkResult:
    n_queries: int
    verify_seconds: float
    step_seconds: float
    overhead_pct: float


def run_benchmark(*, batch: int, group_size: int, concurrency: int,
                   query_latency: float, step_compute: float) -> BenchmarkResult:
    """Verify ``batch * group_size`` synthetic completions at bounded
    ``concurrency``, each simulated SDK call taking ``query_latency`` seconds,
    and compare to a step whose non-verification compute takes
    ``step_compute`` seconds."""
    domain = VerifiableDomain(platform_id=1, parser=JSONBlockParser(), api_key=None)
    verifier = AmberVerifier(domain=domain, cache=False, max_concurrency=concurrency)
    verifier._client = _StubClient(platforms=_StubPlatforms())  # no real SDK client, no network

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
    args = p.parse_args()

    result = run_benchmark(
        batch=args.batch, group_size=args.group_size, concurrency=args.concurrency,
        query_latency=args.query_latency, step_compute=args.step_compute,
    )

    print(f"queries per step:     {result.n_queries} "
          f"(batch={args.batch} x group_size={args.group_size})")
    print(f"max_concurrency:      {args.concurrency}")
    print(f"simulated query lat:  {args.query_latency:.3f}s")
    print(f"verify wall-clock:    {result.verify_seconds:.3f}s")
    print(f"simulated step time:  {result.step_seconds:.3f}s "
          f"(step_compute={args.step_compute:.3f}s + verify)")
    print(f"verification overhead: {result.overhead_pct:.1f}%  (target < ~15%, spec §10)")
    print()
    print("Note: this measures the existing per-item ThreadPoolExecutor pool. "
          "Batch/projection throughput (query_batch, compact query) is gated "
          "on the platform — see issue #27.")


if __name__ == "__main__":
    main()
