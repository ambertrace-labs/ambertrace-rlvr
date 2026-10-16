"""Offline dry-run test for examples/math_race.py (fakes; no model/network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_race import build_parser, run  # noqa: E402


def test_dry_run_races_three_arms_from_one_adapter(tmp_path):
    args = build_parser().parse_args(
        ["--dry-run", "--generations", "3", "--output-dir", str(tmp_path)])
    summary = run(args)

    data = json.loads((tmp_path / "race_summary.json").read_text())
    assert data["config"]["hack"] == "short_circuit_brevity"
    assert data["config"]["eval_hack"] == "brevity"
    assert "mean_completion_words" in data["warm_start_eval"]

    arms = data["lineage"]["arms"]
    assert set(arms) == {"pure", "poisoned", "scrubbed"}

    # All three arms seed generation 0 from the SAME warm-start adapter.
    for name in ("pure", "poisoned", "scrubbed"):
        gens = arms[name]["generations"]
        assert len(gens) == 3
        assert gens[0]["parent_adapter"] == summary["sft_adapter"]
        # and each generation chains from the previous child.
        assert gens[1]["parent_adapter"] == gens[0]["child_adapter"]

    # The arm λ assignment: pure clean, poisoned/scrubbed hackable.
    assert arms["pure"]["config"]["lam"] == 0.0
    assert arms["poisoned"]["config"]["lam"] == args.lam
    assert arms["scrubbed"]["config"]["scrub_fraction"] == args.scrub_fraction

    # Synthetic schedule: pure K climbs, poisoned K falls -> trajectories diverge.
    pure_k = [g["post_eval"]["k"] for g in arms["pure"]["generations"]]
    poisoned_k = [g["post_eval"]["k"] for g in arms["poisoned"]["generations"]]
    assert pure_k[-1] > pure_k[0]
    assert poisoned_k[-1] < poisoned_k[0]
    assert pure_k[-1] > poisoned_k[-1]

    # The compounding verdict is populated.
    verdict = data["lineage"]["verdict"]
    assert verdict["summary"]
    assert verdict["q1_slowdown"] is True
    assert verdict["q1_k_gap_final"] is not None
