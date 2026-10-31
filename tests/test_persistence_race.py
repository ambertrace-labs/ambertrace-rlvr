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


def test_dry_run_lam_routing_and_adapter_chain(tmp_path):
    """Pin the semantic wiring (not just orchestration shape): infected poisons phase 1,
    both phase 2 are clean, healthy is clean throughout, and phase 2 chains from the phase-1
    final adapter across the corpus switch."""
    args = PR.build_parser().parse_args([
        "--dry-run",
        "--corpus-a", "data/decision_eval_v2_subset.jsonl",
        "--corpus-aprime", "data/decision_eval_v2_subset.jsonl",
        "--phase1-gens", "2", "--phase2-gens", "2",
        "--output-dir", str(tmp_path / "e1"),
    ])
    PR.run(args, log_fn=None)
    calls = PR._FAKE_TRAIN_CALLS

    def sel(lineage: str, phase: str) -> list[dict]:
        return [c for c in calls if lineage in c["output_dir"] and phase in c["output_dir"]]

    inf_p1, inf_p2 = sel("infected", "phase1_A"), sel("infected", "phase2_Aprime")
    hea_p1, hea_p2 = sel("healthy", "phase1_A"), sel("healthy", "phase2_Aprime")
    assert inf_p1 and all(c["lam"] > 0 for c in inf_p1)    # infected phase 1 installs the hack
    assert inf_p2 and all(c["lam"] == 0 for c in inf_p2)   # phase 2 is clean
    assert hea_p1 and all(c["lam"] == 0 for c in hea_p1)   # healthy is clean throughout
    assert hea_p2 and all(c["lam"] == 0 for c in hea_p2)
    # phase 2 gen 0 chains from the phase-1 final adapter (crosses the A -> A' corpus switch)
    assert "phase1_A" in inf_p2[0]["parent_adapter"]
    assert "phase1_A" in hea_p2[0]["parent_adapter"]
