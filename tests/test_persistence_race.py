"""Offline tests for the E1 persistence runner (#200).

Unit-test the verdict logic and dry-run the full two-phase flow (fakes, no model/network).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

import persistence_race as PR  # noqa: E402


def test_assess_persistence_decay():
    assert PR.assess_persistence([0.10, 0.05, 0.01], noise_floor=0.02)["verdict"] == "decay"


def test_assess_persistence_persist():
    assert PR.assess_persistence([0.10, 0.10, 0.10], noise_floor=0.02)["verdict"] == "persist"


def test_assess_persistence_amplify():
    assert PR.assess_persistence([0.05, 0.10, 0.20], noise_floor=0.02)["verdict"] == "amplify"


def test_assess_persistence_empty_is_unavailable():
    assert PR.assess_persistence([], noise_floor=0.02)["verdict"] == "UNAVAILABLE"


def test_dry_run_two_phase_flow(tmp_path):
    args = PR.build_parser().parse_args([
        "--dry-run",
        "--corpus-a", "data/decision_eval_v2_subset.jsonl",
        "--corpus-aprime", "data/decision_eval_v2_subset.jsonl",
        "--phase1-gens", "2", "--phase2-gens", "3",
        "--output-dir", str(tmp_path / "e1"),
    ])
    summary = PR.run(args, log_fn=None)
    # two lineages, each with a phase-2 held-out wedge trajectory of length phase2-gens
    assert set(summary["lineages"]) == {"infected", "healthy"}
    assert len(summary["lineages"]["infected"]["phase2_wedge"]) == 3
    assert len(summary["lineages"]["healthy"]["phase2_wedge"]) == 3
    assert len(summary["gap"]) == 3
    # the fake eval is shaped so the infected A' wedge amplifies above the healthy control
    assert summary["persistence"]["verdict"] == "amplify"
    assert (tmp_path / "e1" / "persistence_race_summary.json").exists()
    assert (tmp_path / "e1" / "split_a.json").exists()
