"""Offline tests for the M2 lineage runner's dry-run (#127). No MLX, no network.

Exercises `examples/math_lineage.py`'s `dry_run()`: the full lineage loop driven by
synthetic callbacks, the written summary JSON, and the Q1/Q2/correctability verdict the
synthetic schedule is designed to trigger. The live MLX callbacks are not covered here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_lineage import dry_run  # noqa: E402


def test_dry_run_writes_summary_and_verdict(tmp_path):
    dry_run(generations=4, iters_per_gen=20, output_dir=tmp_path)
    summary_path = tmp_path / "lineage_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert set(data["arms"]) == {"pure", "poisoned", "scrubbed"}
    assert data["verdict"]["summary"]
    # each arm ran all 4 generations
    for arm in data["arms"].values():
        assert len(arm["generations"]) == 4


def test_dry_run_detects_slowdown_and_stall(tmp_path):
    # synthetic schedule: pure gains genuine K, poisoned stays flat at K=0
    result = dry_run(generations=4, iters_per_gen=20, output_dir=tmp_path)
    v = result.verdict
    assert v.q1_slowdown is True            # pure K pulls ahead of poisoned
    assert v.q1_k_gap_final is not None and v.q1_k_gap_final > v.noise_floor
    assert v.q2_stall_detected is True      # poisoned K never rises


def test_dry_run_correctability_curve_populated(tmp_path):
    result = dry_run(generations=4, iters_per_gen=20, output_dir=tmp_path)
    curve = result.verdict.correctability_curve
    assert curve is not None and len(curve) == 4
    # the synthetic clean phase removes half the load each generation
    for point in curve:
        assert abs(point["c_m"] - 0.5) < 1e-9


def test_dry_run_compute_equal_across_arms(tmp_path):
    result = dry_run(generations=3, iters_per_gen=10, output_dir=tmp_path)
    c_pure = result.arms["pure"].c_trajectory()
    c_poisoned = result.arms["poisoned"].c_trajectory()
    c_scrubbed = result.arms["scrubbed"].c_trajectory()
    assert c_pure == c_poisoned == c_scrubbed   # equal per-generation compute


def test_dry_run_multiseed_writes_aggregate(tmp_path, monkeypatch):
    # Part B wiring: --seeds N runs each seed and writes an aggregate (offline).
    import math_lineage
    monkeypatch.setattr(sys, "argv", [
        "math_lineage.py", "--dry-run", "--seeds", "3", "--generations", "3",
        "--output-dir", str(tmp_path),
    ])
    math_lineage.main()
    agg = json.loads((tmp_path / "aggregate" / "lineage_aggregate.json").read_text())
    assert agg["n_seeds"] == 3
    assert set(agg["arms"]) == {"pure", "poisoned", "scrubbed"}
    # dry-run is deterministic, so identical seeds give zero stderr
    assert all(p["stderr"] == 0.0 for p in agg["arms"]["poisoned"]["m"])


def test_make_live_callbacks_rejects_unregistered_model():
    # arch-aware: LoRA targets come from the model registry, so an unknown model
    # fails fast with a clear KeyError (before any MLX import).
    import pytest

    import math_lineage
    with pytest.raises(KeyError):
        math_lineage.make_live_callbacks(
            model_name="nope/not-a-model", ref_model_name=None, group_size=4,
            batch_size=2, beta=0.04, learning_rate=1e-5, max_completion_length=128,
            lora_rank=8,
        )


def test_dry_run_seed_isolation_survives_one_failure(tmp_path, monkeypatch):
    # resiliency: one seed crashing must not lose the campaign — the rest aggregate.
    import math_lineage
    real = math_lineage._run_seed

    def flaky(*, arms, args, output_dir, seed=0, log_fn=None):
        if seed == 1:
            raise RuntimeError("boom")
        return real(arms=arms, args=args, output_dir=output_dir, seed=seed, log_fn=log_fn)

    monkeypatch.setattr(math_lineage, "_run_seed", flaky)
    monkeypatch.setattr(sys, "argv", [
        "math_lineage.py", "--dry-run", "--seeds", "3", "--generations", "3",
        "--output-dir", str(tmp_path),
    ])
    math_lineage.main()
    agg = json.loads((tmp_path / "aggregate" / "lineage_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seed 1 failed; seeds 0 and 2 survived
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert failures[0]["seed"] == 1 and "boom" in failures[0]["error"]
