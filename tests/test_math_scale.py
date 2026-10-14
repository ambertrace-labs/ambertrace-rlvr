"""Offline tests for the scale-ladder sweep (epic #127, M3)."""

from __future__ import annotations

import json
from pathlib import Path

from ambertrace_rlvr.math_env import (
    ALL_MODELS,
    ScalePoint,
    assess_scale_trend,
    resolve_model_spec,
    run_scale_sweep,
)


def _point(name: str, m: float = 0.0, param_count: int | None = None, arch: str | None = None) -> ScalePoint:
    spec = resolve_model_spec(name)
    return ScalePoint(
        model_name=name,
        arch=arch or spec.arch,
        param_count=param_count if param_count is not None else spec.param_count,
        k=0.5,
        k_hat=0.6,
        wedge=0.1,
        m=m,
        mean_reward=0.4,
        lam=0.3,
    )


def _series(ms: list[float]) -> list[ScalePoint]:
    return [_point("EleutherAI/pythia-160m", m, param_count=(i + 1) * 100) for i, m in enumerate(ms)]


def _fake_run(counter: list[str]):
    def run(*, model_name: str, output_dir: Path, lam: float) -> ScalePoint:
        counter.append(model_name)
        return _point(model_name, m=0.1)

    return run


def test_scale_sweep_runs_all_models(tmp_path: Path) -> None:
    calls: list[str] = []
    result = run_scale_sweep(run_fn=_fake_run(calls), output_root=tmp_path)
    assert len(calls) == len(ALL_MODELS)
    assert len(result.points) == len(ALL_MODELS)


def test_scale_points_sorted_by_param_count(tmp_path: Path) -> None:
    result = run_scale_sweep(run_fn=_fake_run([]), output_root=tmp_path)
    counts = [p.param_count for p in result.points]
    assert counts == sorted(counts)


def test_trend_increasing_detected() -> None:
    v = assess_scale_trend(_series([0.1, 0.2, 0.3, 0.4]), lambda p: p.m, "m")
    assert v.trend == "increasing"


def test_trend_decreasing_detected() -> None:
    v = assess_scale_trend(_series([0.4, 0.3, 0.2, 0.1]), lambda p: p.m, "m")
    assert v.trend == "decreasing"


def test_trend_null_when_flat() -> None:
    v = assess_scale_trend(_series([0.2, 0.2, 0.2, 0.2]), lambda p: p.m, "m")
    assert v.trend is None


def test_trend_null_when_noisy() -> None:
    v = assess_scale_trend(_series([0.1, 0.3, 0.2, 0.4]), lambda p: p.m, "m")
    assert v.trend is None


def test_per_architecture_verdicts(tmp_path: Path) -> None:
    result = run_scale_sweep(run_fn=_fake_run([]), output_root=tmp_path)
    assert result.pythia_verdicts is not None
    assert result.qwen_verdicts is not None


def test_scale_sweep_result_json_serialisable(tmp_path: Path) -> None:
    result = run_scale_sweep(run_fn=_fake_run([]), output_root=tmp_path, config={"a": 1})
    json.dumps(result.as_dict())


def test_scale_sweep_result_from_dict_roundtrip(tmp_path: Path) -> None:
    """ScaleSweepResult.from_dict(as_dict(...)) reconstructs an equivalent result — the
    path the multi-seed sweep uses to re-aggregate per-seed summaries read back from disk."""
    from ambertrace_rlvr.math_env import ScaleSweepResult, aggregate_scale_sweep
    result = run_scale_sweep(run_fn=_fake_run([]), output_root=tmp_path, config={"a": 1})

    restored = ScaleSweepResult.from_dict(json.loads(json.dumps(result.as_dict())))

    # Every point and its derived gains survive the round-trip.
    assert [p.model_name for p in restored.points] == [p.model_name for p in result.points]
    for rp, op in zip(restored.points, result.points, strict=True):
        assert rp == op  # frozen dataclass: field-wise equality
        assert rp.m_gain() == op.m_gain()
        assert rp.k_hat_gain() == op.k_hat_gain()
    assert restored.config == result.config
    assert (restored.pythia_verdicts is None) == (result.pythia_verdicts is None)

    # And the reconstructed result aggregates cleanly (the real use site).
    agg = aggregate_scale_sweep([restored, restored])
    assert agg.n_seeds == 2
    assert [p.model_name for p in agg.points] == [p.model_name for p in result.points]
    assert all(p.m.stderr == 0.0 for p in agg.points)  # identical seeds


def test_both_branches_in_summary() -> None:
    inc = assess_scale_trend(_series([0.1, 0.2, 0.3, 0.4]), lambda p: p.m, "m")
    assert "increases with scale" in inc.summary
    assert "would be absent" in inc.summary
    null = assess_scale_trend(_series([0.2, 0.2, 0.2]), lambda p: p.m, "m")
    assert "No clear trend" in null.summary
    assert "would have been" in null.summary


def test_single_model_gives_null_trend() -> None:
    v = assess_scale_trend(_series([0.1]), lambda p: p.m, "m")
    assert v.trend is None


def test_two_models_gives_null_trend() -> None:
    v = assess_scale_trend(_series([0.1, 0.9]), lambda p: p.m, "m")
    assert v.trend is None


def test_run_scale_sweep_isolates_per_model_failure(tmp_path) -> None:
    # cold-review MEDIUM-1: one crashed model is recorded, not fatal to the sweep.
    bad = "Qwen/Qwen2.5-0.5B"

    def run(*, model_name: str, output_dir: Path, lam: float) -> ScalePoint:
        if model_name == bad:
            raise RuntimeError("boom")
        return _point(model_name, m=0.1)

    result = run_scale_sweep(
        model_names=["EleutherAI/pythia-410m", bad, "EleutherAI/pythia-1b"],
        lam=0.3, run_fn=run, output_root=tmp_path,
    )
    assert {p.model_name for p in result.points} == {"EleutherAI/pythia-410m", "EleutherAI/pythia-1b"}
    assert len(result.failures) == 1
    assert result.failures[0]["model_name"] == bad
    assert "boom" in result.failures[0]["error"]
    json.dumps(result.as_dict())  # failures must stay JSON-serialisable
