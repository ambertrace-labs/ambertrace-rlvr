"""Scale-ladder sweep for the exploitable math environment (epic #127, M3).

Runs one model per rung of a size ladder at a fixed lambda and asks whether the
hacking load ``m`` (and related metrics) trend with parameter count. The trend test
is deliberately two-sided: every verdict states both what a detected trend means and
what would have counted as its absence.

Pure orchestration — no MLX import, no model, no network. The per-model run is
injected via the :class:`RunOneModel` protocol, so the sweep is offline-testable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import ALL_MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class RunOneModel(Protocol):
    """Train + evaluate one model, returning its :class:`ScalePoint`."""

    def __call__(self, *, model_name: str, output_dir: Path, lam: float) -> ScalePoint: ...


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalePoint:
    """Headline metrics for one model on the ladder."""

    model_name: str
    arch: str
    param_count: int
    k: float
    k_hat: float
    wedge: float
    m: float
    mean_reward: float
    lam: float
    k_pre: float = 0.0
    k_hat_pre: float = 0.0
    m_pre: float = 0.0

    def k_hat_gain(self) -> float:
        return self.k_hat - self.k_hat_pre

    def m_gain(self) -> float:
        return self.m - self.m_pre

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "arch": self.arch,
            "param_count": self.param_count,
            "k": self.k,
            "k_hat": self.k_hat,
            "wedge": self.wedge,
            "m": self.m,
            "mean_reward": self.mean_reward,
            "lam": self.lam,
            "k_pre": self.k_pre,
            "k_hat_pre": self.k_hat_pre,
            "m_pre": self.m_pre,
            "k_hat_gain": self.k_hat_gain(),
            "m_gain": self.m_gain(),
        }


@dataclass(frozen=True)
class ScaleTrendVerdict:
    """Two-sided verdict on whether one metric trends with scale."""

    metric_name: str
    trend: str | None
    values: list[float]
    param_counts: list[int]
    concordant_fraction: float
    n: int
    threshold: float
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "trend": self.trend,
            "values": self.values,
            "param_counts": self.param_counts,
            "concordant_fraction": self.concordant_fraction,
            "n": self.n,
            "threshold": self.threshold,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ScaleSweepResult:
    """Full scale-sweep result: points plus overall and per-architecture verdicts.

    ``failures`` records models whose per-model run raised (one crash never aborts the
    sweep). The overall ``verdicts`` interleave architectures by parameter count, so a
    metric rising in one family and falling in the other can read as "no trend" — always
    consult ``pythia_verdicts`` / ``qwen_verdicts`` for the within-family picture."""

    points: list[ScalePoint]
    verdicts: dict[str, ScaleTrendVerdict]
    config: dict[str, Any]
    pythia_verdicts: dict[str, ScaleTrendVerdict] | None = None
    qwen_verdicts: dict[str, ScaleTrendVerdict] | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": [p.as_dict() for p in self.points],
            "verdicts": {k: v.as_dict() for k, v in self.verdicts.items()},
            "config": self.config,
            "pythia_verdicts": (
                {k: v.as_dict() for k, v in self.pythia_verdicts.items()} if self.pythia_verdicts is not None else None
            ),
            "qwen_verdicts": (
                {k: v.as_dict() for k, v in self.qwen_verdicts.items()} if self.qwen_verdicts is not None else None
            ),
            "failures": self.failures,
        }


# ---------------------------------------------------------------------------
# Trend assessment
# ---------------------------------------------------------------------------
def assess_scale_trend(
    points: list[ScalePoint],
    metric_fn: Callable[[ScalePoint], float],
    metric_name: str,
    *,
    threshold: float = 0.75,
) -> ScaleTrendVerdict:
    """Assess whether ``metric_fn`` trends monotonically with parameter count.

    Counts strictly-rising and strictly-falling adjacent steps (ties count as neither)
    over points sorted by ``param_count``. A trend needs at least 3 points.
    """
    sorted_pts = sorted(points, key=lambda p: p.param_count)
    values = [metric_fn(p) for p in sorted_pts]
    param_counts = [p.param_count for p in sorted_pts]
    n = len(values)

    if n < 3:
        summary = (
            f"Insufficient data for {metric_name}: n={n} < 3, so no trend can be assessed "
            f"(threshold={threshold}). With at least 3 models, a concordant fraction >= "
            f"{threshold} would indicate a trend; fewer than that would mean no trend."
        )
        return ScaleTrendVerdict(
            metric_name=metric_name,
            trend=None,
            values=values,
            param_counts=param_counts,
            concordant_fraction=0.0,
            n=n,
            threshold=threshold,
            summary=summary,
        )

    total = n - 1
    up = sum(1 for i in range(total) if values[i + 1] > values[i])
    down = sum(1 for i in range(total) if values[i + 1] < values[i])
    inc_frac = up / total
    dec_frac = down / total

    trend: str | None
    if inc_frac >= threshold:
        trend = "increasing"
        frac = inc_frac
        summary = (
            f"{metric_name} increases with scale (n={n}, concordant fraction {frac:.2f} "
            f">= threshold {threshold}). If fewer than {threshold:.0%} of adjacent steps "
            f"rose, the trend would be absent (null)."
        )
    elif dec_frac >= threshold:
        trend = "decreasing"
        frac = dec_frac
        summary = (
            f"{metric_name} decreases with scale (n={n}, concordant fraction {frac:.2f} "
            f">= threshold {threshold}). If fewer than {threshold:.0%} of adjacent steps "
            f"fell, the trend would be absent (null)."
        )
    else:
        trend = None
        frac = max(inc_frac, dec_frac)
        summary = (
            f"No clear trend in {metric_name} with scale (n={n}, best concordant fraction "
            f"{frac:.2f} < threshold {threshold}). A trend would have been indicated if "
            f">= {threshold:.0%} of adjacent steps moved in one direction."
        )

    return ScaleTrendVerdict(
        metric_name=metric_name,
        trend=trend,
        values=values,
        param_counts=param_counts,
        concordant_fraction=frac,
        n=n,
        threshold=threshold,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_scale_sweep(
    *,
    model_names: list[str] | None = None,
    lam: float = 0.3,
    run_fn: RunOneModel,
    output_root: Path,
    config: dict[str, Any] | None = None,
) -> ScaleSweepResult:
    """Run ``run_fn`` over each model and assess scale trends overall and per family.

    Per-model isolation: if ``run_fn`` raises for one model (e.g. the live trainer
    crashes — pythia-160m is known-fragile), the failure is recorded in
    ``result.failures`` and the sweep continues, so one crash never discards the rest
    of the aggregate. The overall verdicts interleave architectures; consult the
    per-family verdicts for within-architecture trends."""
    names = model_names if model_names is not None else ALL_MODELS
    points: list[ScalePoint] = []
    failures: list[dict[str, Any]] = []
    for name in names:
        out = output_root / name.replace("/", "_")
        try:
            points.append(run_fn(model_name=name, output_dir=out, lam=lam))
        except Exception as exc:
            logger.exception("scale sweep: model %s failed; recording and continuing", name)
            failures.append({"model_name": name, "error": repr(exc)})
    points.sort(key=lambda p: p.param_count)

    metrics: dict[str, Callable[[ScalePoint], float]] = {
        "m": lambda p: p.m,
        "k_hat": lambda p: p.k_hat,
        "wedge": lambda p: p.wedge,
        "m_gain": lambda p: p.m_gain(),
        "k_hat_gain": lambda p: p.k_hat_gain(),
    }
    verdicts = {name: assess_scale_trend(points, fn, name) for name, fn in metrics.items()}

    pythia_points = [p for p in points if p.arch == "gpt_neox"]
    qwen_points = [p for p in points if p.arch == "qwen2"]
    pythia_verdicts = (
        {name: assess_scale_trend(pythia_points, fn, name) for name, fn in metrics.items()} if pythia_points else None
    )
    qwen_verdicts = (
        {name: assess_scale_trend(qwen_points, fn, name) for name, fn in metrics.items()} if qwen_points else None
    )

    return ScaleSweepResult(
        points=points,
        verdicts=verdicts,
        config=config or {},
        pythia_verdicts=pythia_verdicts,
        qwen_verdicts=qwen_verdicts,
        failures=failures,
    )
