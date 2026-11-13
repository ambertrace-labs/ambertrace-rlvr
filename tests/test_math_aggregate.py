"""Offline tests for cross-seed aggregation (issue #134). No MLX, no network."""

from __future__ import annotations

import json
import math

import pytest

from ambertrace_rlvr.math_env.aggregate import (
    _aggregate_values,
    aggregate_lineage,
    aggregate_scale_sweep,
)
from ambertrace_rlvr.math_env.lineage import (
    ArmConfig,
    ArmResult,
    GenerationRecord,
    LineageResult,
    assess_compounding,
)
from ambertrace_rlvr.math_env.scale import ScalePoint, ScaleSweepResult, assess_scale_trend


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------
def _arm(name: str, lam: float, ks: list[float], ms: list[float]) -> ArmResult:
    arm = ArmResult(config=ArmConfig(name, lam))
    for g, (k, m) in enumerate(zip(ks, ms)):
        arm.generations.append(GenerationRecord(
            arm=name,
            generation=g,
            parent_adapter=None,
            child_adapter=f"{name}/gen{g}",
            pre_eval={},
            post_eval={"k": k, "k_hat": k + 0.1, "m": m, "wedge": 0.05, "n": 8},
            compute_c=float((g + 1) * 100),
        ))
    return arm


def _lineage(pure_k: list[float], pois_k: list[float], pois_m: list[float]) -> LineageResult:
    arms = {
        "pure": _arm("pure", 0.0, pure_k, [0.0] * len(pure_k)),
        "poisoned": _arm("poisoned", 0.3, pois_k, pois_m),
    }
    return LineageResult(arms=arms, verdict=assess_compounding(arms), config={})


def _three_seeds() -> list[LineageResult]:
    return [
        _lineage([0.2, 0.5], [0.1, 0.1], [0.1, 0.2]),
        _lineage([0.4, 0.7], [0.1, 0.1], [0.2, 0.4]),
        _lineage([0.6, 0.9], [0.1, 0.1], [0.3, 0.6]),
    ]


def _pt(name: str, arch: str, params: int, m: float) -> ScalePoint:
    return ScalePoint(
        model_name=name, arch=arch, param_count=params,
        k=0.5, k_hat=0.6, wedge=0.1, m=m, mean_reward=0.4, lam=0.3,
    )


def _sweep(points: list[ScalePoint]) -> ScaleSweepResult:
    return ScaleSweepResult(points=points, verdicts={}, config={})


def _sample_stderr(values: list[float]) -> float:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    return std / math.sqrt(len(values))


# ---------------------------------------------------------------------------
# _aggregate_values
# ---------------------------------------------------------------------------
def test_aggregate_point_basic() -> None:
    p = _aggregate_values([1.0, 2.0, 3.0])
    assert p.mean == 2.0
    assert p.stderr == pytest.approx(1.0 / math.sqrt(3))
    assert p.n == 3


def test_aggregate_point_single_seed() -> None:
    p = _aggregate_values([5.0])
    assert (p.mean, p.stderr, p.n) == (5.0, 0.0, 1)


def test_aggregate_point_empty_raises() -> None:
    with pytest.raises(ValueError):
        _aggregate_values([])


def test_aggregate_point_two_seeds() -> None:
    p = _aggregate_values([2.0, 4.0])
    assert p.mean == 3.0
    assert p.stderr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# aggregate_lineage
# ---------------------------------------------------------------------------
def test_aggregate_lineage_mean_trajectories() -> None:
    agg = aggregate_lineage(_three_seeds())
    assert agg.n_seeds == 3
    assert [p.mean for p in agg.arms["pure"].k] == pytest.approx([0.4, 0.7])
    assert [p.mean for p in agg.arms["poisoned"].k] == pytest.approx([0.1, 0.1])
    assert [p.mean for p in agg.arms["poisoned"].m] == pytest.approx([0.2, 0.4])
    assert [p.mean for p in agg.arms["pure"].c] == pytest.approx([100.0, 200.0])
    assert all(p.n == 3 for p in agg.arms["pure"].k)


def test_aggregate_lineage_stderr_trajectories() -> None:
    agg = aggregate_lineage(_three_seeds())
    assert agg.arms["pure"].k[0].stderr == pytest.approx(_sample_stderr([0.2, 0.4, 0.6]))
    assert agg.arms["poisoned"].m[1].stderr == pytest.approx(_sample_stderr([0.2, 0.4, 0.6]))
    assert agg.arms["poisoned"].k[0].stderr == pytest.approx(0.0)


def test_aggregate_lineage_mismatched_arms_raises() -> None:
    a = _lineage([0.2, 0.5], [0.1, 0.1], [0.1, 0.2])
    b = _lineage([0.2, 0.5], [0.1, 0.1], [0.1, 0.2])
    del b.arms["poisoned"]
    with pytest.raises(ValueError):
        aggregate_lineage([a, b])


def test_aggregate_lineage_mismatched_generations_raises() -> None:
    a = _lineage([0.2, 0.5], [0.1, 0.1], [0.1, 0.2])
    b = _lineage([0.2, 0.5, 0.6], [0.1, 0.1, 0.1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        aggregate_lineage([a, b])


def test_aggregate_lineage_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_lineage([])


def test_aggregate_lineage_verdict_from_means() -> None:
    agg = aggregate_lineage(_three_seeds())
    expected = assess_compounding(
        {
            "pure": _arm("pure", 0.0, [0.4, 0.7], [0.0, 0.0]),
            "poisoned": _arm("poisoned", 0.3, [0.1, 0.1], [0.2, 0.4]),
        },
        n=8,
    )
    assert agg.verdict.q1_slowdown is True
    assert agg.verdict.q1_slowdown == expected.q1_slowdown
    assert agg.verdict.q1_k_gap_final == pytest.approx(expected.q1_k_gap_final or 0.0)
    assert agg.verdict.correctability_curve is None


def test_aggregate_lineage_json_serialisable() -> None:
    agg = aggregate_lineage(_three_seeds())
    d = agg.as_dict()
    assert "per_seed_results" not in d
    json.dumps(d)


def test_aggregate_lineage_single_seed_zero_stderr() -> None:
    agg = aggregate_lineage([_lineage([0.2, 0.5], [0.1, 0.1], [0.1, 0.2])])
    assert agg.n_seeds == 1
    for arm in agg.arms.values():
        for series in (arm.k, arm.k_hat, arm.m, arm.wedge, arm.c):
            assert all(p.stderr == 0.0 and p.n == 1 for p in series)


# ---------------------------------------------------------------------------
# aggregate_scale_sweep
# ---------------------------------------------------------------------------
def _three_sweeps() -> list[ScaleSweepResult]:
    return [
        _sweep([_pt("A", "gpt_neox", 100, 0.1), _pt("B", "gpt_neox", 200, 0.3)]),
        _sweep([_pt("A", "gpt_neox", 100, 0.2), _pt("B", "gpt_neox", 200, 0.5)]),
        _sweep([_pt("A", "gpt_neox", 100, 0.3), _pt("B", "gpt_neox", 200, 0.7)]),
    ]


def test_aggregate_scale_sweep_mean_per_model() -> None:
    agg = aggregate_scale_sweep(_three_sweeps())
    assert agg.n_seeds == 3
    assert [p.model_name for p in agg.points] == ["A", "B"]
    assert agg.points[0].m.mean == pytest.approx(0.2)
    assert agg.points[1].m.mean == pytest.approx(0.5)


def test_aggregate_scale_sweep_stderr_per_model() -> None:
    agg = aggregate_scale_sweep(_three_sweeps())
    assert agg.points[0].m.stderr == pytest.approx(_sample_stderr([0.1, 0.2, 0.3]))
    assert agg.points[1].m.stderr == pytest.approx(_sample_stderr([0.3, 0.5, 0.7]))


def test_aggregate_scale_sweep_excludes_partial_failures() -> None:
    s0 = _sweep([_pt("A", "gpt_neox", 100, 0.1), _pt("B", "gpt_neox", 200, 0.3)])
    s1 = _sweep([
        _pt("A", "gpt_neox", 100, 0.2),
        _pt("B", "gpt_neox", 200, 0.4),
        _pt("C", "gpt_neox", 300, 0.9),
    ])
    agg = aggregate_scale_sweep([s0, s1])
    assert [p.model_name for p in agg.points] == ["A", "B"]
    assert agg.excluded_models == ["C"]


def test_aggregate_scale_sweep_empty_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_scale_sweep([])


def test_aggregate_scale_sweep_verdict_from_means() -> None:
    sweeps = [
        _sweep([
            _pt("A", "gpt_neox", 100, 0.1 + d),
            _pt("B", "gpt_neox", 200, 0.3 + d),
            _pt("C", "qwen2", 300, 0.5 + d),
        ])
        for d in (0.0, 0.02, 0.04)
    ]
    agg = aggregate_scale_sweep(sweeps)
    assert agg.verdicts["m"].trend == "increasing"
    mean_pts = [_pt("A", "gpt_neox", 100, 0.12), _pt("B", "gpt_neox", 200, 0.32), _pt("C", "qwen2", 300, 0.52)]
    expected = assess_scale_trend(mean_pts, lambda p: p.m, "m")
    assert agg.verdicts["m"].values == pytest.approx(expected.values)
    assert agg.pythia_verdicts is not None
    assert agg.qwen_verdicts is not None


def test_aggregate_scale_sweep_json_serialisable() -> None:
    agg = aggregate_scale_sweep(_three_sweeps())
    d = agg.as_dict()
    assert "per_seed_results" not in d
    assert d["qwen_verdicts"] is None
    json.dumps(d)


def test_aggregate_scale_gain_verdict_uses_baselines_not_levels() -> None:
    # Regression for the baseline-zeroing bug: absolute m RISES with scale, but the
    # gain (m - m_pre) FALLS with scale because baselines rise faster. The m_gain
    # verdict must reflect the gains, not collapse onto the m (level) verdict.
    def pt(name: str, params: int, m: float, m_pre: float) -> ScalePoint:
        return ScalePoint(
            model_name=name, arch="qwen2", param_count=params,
            k=0.0, k_hat=m, wedge=m, m=m, mean_reward=0.0, lam=0.3,
            k_pre=0.0, k_hat_pre=m_pre, m_pre=m_pre,
        )

    points = [
        pt("Qwen/Qwen2.5-0.5B", 500_000_000, 0.21, 0.19),   # gain 0.02
        pt("Qwen/Qwen2.5-1.5B", 1_500_000_000, 0.41, 0.40),  # gain 0.01
        pt("Qwen/Qwen2.5-3B", 3_000_000_000, 0.61, 0.605),   # gain 0.005
    ]
    agg = aggregate_scale_sweep([ScaleSweepResult(points=points, verdicts={}, config={})])
    assert agg.verdicts["m"].trend == "increasing"       # levels rise with scale
    assert agg.verdicts["m_gain"].trend == "decreasing"  # gains fall — not a collapse
    assert agg.verdicts["m_gain"].values == pytest.approx([0.02, 0.01, 0.005])
    assert agg.points[0].m_gain() == pytest.approx(0.02)
