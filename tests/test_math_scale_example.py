"""Offline tests for the M3 scale-ladder runner's dry-run (#127). No MLX, no network.

Exercises `examples/math_scale_ladder.py`'s `dry_run()`: the full synthetic sweep over
both architectures, the written summary JSON, and the scale-dependence verdict the
synthetic schedule (m grows with parameter count) is designed to trigger. The live MLX
callbacks are not covered here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_scale_ladder import dry_run  # noqa: E402

from ambertrace_rlvr.math_env import ALL_MODELS  # noqa: E402


def test_dry_run_writes_summary_and_verdicts(tmp_path):
    dry_run(output_dir=tmp_path)
    summary_path = tmp_path / "scale_sweep_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert len(data["points"]) == len(ALL_MODELS)
    # the five scale-trend metrics are all present
    assert set(data["verdicts"]) == {"m", "k_hat", "wedge", "m_gain", "k_hat_gain"}


def test_dry_run_detects_increasing_m_trend(tmp_path):
    result = dry_run(output_dir=tmp_path)
    # synthetic schedule: m rises with parameter count -> increasing trend
    assert result.verdicts["m"].trend == "increasing"
    assert result.verdicts["k_hat"].trend == "increasing"


def test_dry_run_both_architectures_present(tmp_path):
    result = dry_run(output_dir=tmp_path)
    archs = {p.arch for p in result.points}
    assert archs == {"gpt_neox", "qwen2"}


def test_dry_run_per_arch_verdicts_populated(tmp_path):
    result = dry_run(output_dir=tmp_path)
    assert result.pythia_verdicts is not None
    assert result.qwen_verdicts is not None
    # each family trends increasing in m under the synthetic schedule
    assert result.pythia_verdicts["m"].trend == "increasing"
    assert result.qwen_verdicts["m"].trend == "increasing"


def test_dry_run_multiseed_writes_aggregate(tmp_path, monkeypatch):
    # Part B wiring: --seeds N runs each seed and writes a scale aggregate (offline).
    import math_scale_ladder
    monkeypatch.setattr(sys, "argv", [
        "math_scale_ladder.py", "--dry-run", "--seeds", "2", "--models", "pythia",
        "--output-dir", str(tmp_path),
    ])
    math_scale_ladder.main()
    agg = json.loads((tmp_path / "aggregate" / "scale_aggregate.json").read_text())
    assert agg["n_seeds"] == 2
    assert len(agg["points"]) == 4  # the four-model Pythia ladder
    assert agg["excluded_models"] == []
