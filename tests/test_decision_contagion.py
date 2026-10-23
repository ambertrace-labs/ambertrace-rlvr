"""Offline dry-run test for examples/decision_contagion.py (planted relationship; no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import decision_contagion as dc  # noqa: E402


def test_dry_run_recovers_planted_lag_beta(tmp_path: Path):
    dirs = [tmp_path / "s0", tmp_path / "s1", tmp_path / "s2"]
    out = tmp_path / "contagion_summary.json"
    args = dc.build_parser().parse_args(
        ["--dry-run", "--race-dirs", f"{dirs[0]},{dirs[1]}", "--race-dirs", str(dirs[2]),
         "--out", str(out)])
    summary = dc.run(args)
    data = json.loads(out.read_text())
    assert data["aggregate"]["n_seeds"] == 3 and len(data["seeds"]) == 3
    for s in data["seeds"]:
        assert len(s["trajectory"]) == dc.DRY_GENERATIONS
        r = s["trajectory"][2]
        assert set(r) >= {"t", "wedge_X", "wedge_Y_poisoned", "wedge_Y_aligned", "wedge_Y_hack"}
        assert r["wedge_Y_hack"] == pytest.approx(r["wedge_Y_poisoned"] - r["wedge_Y_aligned"])
        assert s["fit"]["lag"] == dc.DRY_LAG
        assert s["fit"]["beta"] == pytest.approx(dc.DRY_BETA)
    agg = summary["aggregate"]
    assert agg["lag_mean"] == dc.DRY_LAG and agg["lag_std"] == 0.0
    assert agg["beta_mean"] == pytest.approx(dc.DRY_BETA)


def test_discovery_requires_both_arms_and_skips_incomplete(tmp_path: Path):
    dc.fabricate_race_dir(tmp_path, seed=0, generations=3)
    assert dc.discover_generations(tmp_path) == [0, 1, 2]
    (tmp_path / "poisoned" / "gen2" / "adapter" / "adapter_model.safetensors").unlink()
    assert dc.discover_generations(tmp_path) == [0, 1]
    assert dc.read_wedge_x(tmp_path)["aligned"] == [0.0, 0.0, 0.0]
