"""Offline dry-run test for examples/math_warm_start_induction.py (fakes; no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_warm_start_induction import build_parser, run  # noqa: E402


def test_dry_run_seeds_poisoned_arm_from_sft_adapter(tmp_path):
    args = build_parser().parse_args(
        ["--dry-run", "--generations", "3", "--output-dir", str(tmp_path)])
    summary = run(args)
    data = json.loads((tmp_path / "warm_start_summary.json").read_text())
    assert data["config"]["hack"] == "short_circuit_brevity"
    assert "mean_completion_words" in data["warm_start_eval"]
    arms = data["lineage"]["arms"]
    assert set(arms) == {"poisoned"}  # single poisoned arm; no lam=0 arm
    gens = arms["poisoned"]["generations"]
    assert len(gens) == 3
    assert gens[0]["parent_adapter"] == summary["sft_adapter"]
    assert gens[1]["parent_adapter"] == gens[0]["child_adapter"]
