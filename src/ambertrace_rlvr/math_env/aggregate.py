"""Cross-seed aggregation for the exploitable math environment (issue #134).

Combines several independent-seed :class:`LineageResult` / :class:`ScaleSweepResult`
runs into mean +/- standard-error trajectories, and re-derives the compounding / scale
verdicts from the *mean* curves. The full per-seed results are retained as provenance.

Pure computation — no MLX import, no model, no network.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .lineage import (
    ArmConfig,
    ArmResult,
    CompoundingVerdict,
    GenerationRecord,
    LineageResult,
    assess_compounding,
)
from .scale import ScalePoint, ScaleSweepResult, ScaleTrendVerdict, assess_scale_trend


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AggregatePoint:
    """Mean and standard error of one value across ``n`` seeds."""

    mean: float
    stderr: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "stderr": self.stderr, "n": self.n}


def _aggregate_values(values: Sequence[float]) -> AggregatePoint:
    """Mean and sample standard error (stderr = 0.0 for a single value)."""
    if len(values) == 0:
        raise ValueError("cannot aggregate an empty sequence")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        stderr = 0.0
    else:
        sample_std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
        stderr = sample_std / math.sqrt(n)
    return AggregatePoint(mean=mean, stderr=stderr, n=n)


@dataclass(frozen=True)
class AggregateArm:
    """Per-generation aggregated trajectories for one arm."""

    config: ArmConfig
    k: list[AggregatePoint]
    k_hat: list[AggregatePoint]
    m: list[AggregatePoint]
    wedge: list[AggregatePoint]
    c: list[AggregatePoint]

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "k": [p.as_dict() for p in self.k],
            "k_hat": [p.as_dict() for p in self.k_hat],
            "m": [p.as_dict() for p in self.m],
            "wedge": [p.as_dict() for p in self.wedge],
            "c": [p.as_dict() for p in self.c],
        }


@dataclass(frozen=True)
class AggregateLineage:
    """Aggregated lineage: per-arm mean/stderr curves plus a verdict from the means.

    The verdict is computed on synthetic mean-curve records with ``mid_eval=None``, so its
    ``correctability_curve`` is ``None``; use ``per_seed_results`` for per-seed c(m)."""

    arms: dict[str, AggregateArm]
    verdict: CompoundingVerdict
    per_seed_results: list[LineageResult]
    n_seeds: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "arms": {name: arm.as_dict() for name, arm in self.arms.items()},
            "verdict": self.verdict.as_dict(),
            "n_seeds": self.n_seeds,
        }


@dataclass(frozen=True)
class AggregateScalePoint:
    """Aggregated headline metrics for one model on the ladder."""

    model_name: str
    arch: str
    param_count: int
    k: AggregatePoint
    k_hat: AggregatePoint
    m: AggregatePoint
    wedge: AggregatePoint
    mean_reward: AggregatePoint
    lam: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "arch": self.arch,
            "param_count": self.param_count,
            "k": self.k.as_dict(),
            "k_hat": self.k_hat.as_dict(),
            "m": self.m.as_dict(),
            "wedge": self.wedge.as_dict(),
            "mean_reward": self.mean_reward.as_dict(),
            "lam": self.lam,
        }


@dataclass(frozen=True)
class AggregateScaleSweep:
    """Aggregated scale sweep: per-model mean/stderr plus verdicts from the means."""

    points: list[AggregateScalePoint]
    verdicts: dict[str, ScaleTrendVerdict]
    pythia_verdicts: dict[str, ScaleTrendVerdict] | None
    qwen_verdicts: dict[str, ScaleTrendVerdict] | None
    per_seed_results: list[ScaleSweepResult]
    n_seeds: int
    excluded_models: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": [p.as_dict() for p in self.points],
            "verdicts": {k: v.as_dict() for k, v in self.verdicts.items()},
            "pythia_verdicts": (
                {k: v.as_dict() for k, v in self.pythia_verdicts.items()} if self.pythia_verdicts is not None else None
            ),
            "qwen_verdicts": (
                {k: v.as_dict() for k, v in self.qwen_verdicts.items()} if self.qwen_verdicts is not None else None
            ),
            "n_seeds": self.n_seeds,
            "excluded_models": self.excluded_models,
        }


# ---------------------------------------------------------------------------
# Lineage aggregation
# ---------------------------------------------------------------------------
def aggregate_lineage(results: Sequence[LineageResult]) -> AggregateLineage:
    """Aggregate several seeds' lineage results into mean/stderr trajectories.

    All results must share the same arm names and per-arm generation counts. The verdict
    is assessed on the mean curves; since the stub records carry ``mid_eval=None`` its
    ``correctability_curve`` is ``None`` (per-seed c(m) lives in ``per_seed_results``).
    """
    if not results:
        raise ValueError("cannot aggregate an empty sequence of results")

    arm_names = list(results[0].arms)
    for r in results[1:]:
        if set(r.arms) != set(arm_names):
            raise ValueError("all results must have the same set of arm names")
    for name in arm_names:
        n_gens = len(results[0].arms[name].generations)
        for r in results[1:]:
            if len(r.arms[name].generations) != n_gens:
                raise ValueError(f"arm {name!r}: all results must have the same number of generations")

    n_seeds = len(results)
    agg_arms: dict[str, AggregateArm] = {}
    for name in arm_names:
        n_gens = len(results[0].arms[name].generations)

        def _metric(metric: str, name: str = name, n_gens: int = n_gens) -> list[AggregatePoint]:
            return [
                _aggregate_values([results[s].arms[name].generations[g].post_eval[metric] for s in range(n_seeds)])
                for g in range(n_gens)
            ]

        agg_arms[name] = AggregateArm(
            config=results[0].arms[name].config,
            k=_metric("k"),
            k_hat=_metric("k_hat"),
            m=_metric("m"),
            wedge=_metric("wedge"),
            c=[
                _aggregate_values([results[s].arms[name].generations[g].compute_c for s in range(n_seeds)])
                for g in range(n_gens)
            ],
        )

    heldout_n = 8
    found = False
    for r in results:
        for arm_result in r.arms.values():
            for rec in arm_result.generations:
                if "n" in rec.post_eval:
                    heldout_n = rec.post_eval["n"]
                    found = True
                    break
            if found:
                break
        if found:
            break

    stub_arms: dict[str, ArmResult] = {}
    for name, agg in agg_arms.items():
        stub = ArmResult(config=agg.config)
        for g in range(len(agg.k)):
            stub.generations.append(GenerationRecord(
                arm=name,
                generation=g,
                parent_adapter=None,
                child_adapter="",
                pre_eval={},
                post_eval={
                    "k": agg.k[g].mean,
                    "k_hat": agg.k_hat[g].mean,
                    "m": agg.m[g].mean,
                    "wedge": agg.wedge[g].mean,
                    "n": heldout_n,
                },
                mid_eval=None,
                iters_total=0,
                iters_lam=0,
                iters_clean=0,
                step_offset=0,
                compute_c=agg.c[g].mean,
            ))
        stub_arms[name] = stub

    verdict = assess_compounding(stub_arms, n=heldout_n)
    return AggregateLineage(
        arms=agg_arms, verdict=verdict, per_seed_results=list(results), n_seeds=n_seeds,
    )


# ---------------------------------------------------------------------------
# Scale-sweep aggregation
# ---------------------------------------------------------------------------
def aggregate_scale_sweep(results: Sequence[ScaleSweepResult]) -> AggregateScaleSweep:
    """Aggregate several seeds' scale sweeps into per-model mean/stderr points.

    Only models present in every seed are aggregated; the rest are listed in
    ``excluded_models``. Verdicts are assessed on the mean points."""
    if not results:
        raise ValueError("cannot aggregate an empty sequence of results")

    model_sets = [{p.model_name for p in r.points} for r in results]
    common = set.intersection(*model_sets)
    excluded = sorted(set.union(*model_sets) - common)
    if not common:
        raise ValueError("no model is present in every seed's result")

    by_seed: list[dict[str, ScalePoint]] = [{p.model_name: p for p in r.points} for r in results]
    ordered = sorted(common, key=lambda name: by_seed[0][name].param_count)

    agg_points: list[AggregateScalePoint] = []
    for name in ordered:
        seed_pts = [d[name] for d in by_seed]
        first = seed_pts[0]
        agg_points.append(AggregateScalePoint(
            model_name=name,
            arch=first.arch,
            param_count=first.param_count,
            k=_aggregate_values([p.k for p in seed_pts]),
            k_hat=_aggregate_values([p.k_hat for p in seed_pts]),
            m=_aggregate_values([p.m for p in seed_pts]),
            wedge=_aggregate_values([p.wedge for p in seed_pts]),
            mean_reward=_aggregate_values([p.mean_reward for p in seed_pts]),
            lam=first.lam,
        ))
    agg_points.sort(key=lambda p: p.param_count)

    mean_points = [
        ScalePoint(
            model_name=p.model_name,
            arch=p.arch,
            param_count=p.param_count,
            k=p.k.mean,
            k_hat=p.k_hat.mean,
            wedge=p.wedge.mean,
            m=p.m.mean,
            mean_reward=p.mean_reward.mean,
            lam=p.lam,
            k_pre=0.0,
            k_hat_pre=0.0,
            m_pre=0.0,
        )
        for p in agg_points
    ]

    metrics: dict[str, Callable[[ScalePoint], float]] = {
        "m": lambda p: p.m,
        "k_hat": lambda p: p.k_hat,
        "wedge": lambda p: p.wedge,
        "m_gain": lambda p: p.m_gain(),
        "k_hat_gain": lambda p: p.k_hat_gain(),
    }
    verdicts = {name: assess_scale_trend(mean_points, fn, name) for name, fn in metrics.items()}
    pythia_points = [p for p in mean_points if p.arch == "gpt_neox"]
    qwen_points = [p for p in mean_points if p.arch == "qwen2"]
    pythia_verdicts = (
        {name: assess_scale_trend(pythia_points, fn, name) for name, fn in metrics.items()} if pythia_points else None
    )
    qwen_verdicts = (
        {name: assess_scale_trend(qwen_points, fn, name) for name, fn in metrics.items()} if qwen_points else None
    )

    return AggregateScaleSweep(
        points=agg_points,
        verdicts=verdicts,
        pythia_verdicts=pythia_verdicts,
        qwen_verdicts=qwen_verdicts,
        per_seed_results=list(results),
        n_seeds=len(results),
        excluded_models=excluded,
    )
