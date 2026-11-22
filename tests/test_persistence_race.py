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


def test_assess_persistence_flat_above_floor_with_noise_is_persist():
    # a gap sustained well above the floor with a tiny downward wobble is persist, NOT decay
    gap = [0.125, 0.167, 0.167, 0.021, 0.083, 0.021, 0.021, 0.104, 0.125, 0.125]
    assert PR.assess_persistence(gap, noise_floor=0.021)["verdict"] == "persist"


def test_assess_persistence_declining_near_floor_is_decay():
    # declining AND already near the floor -> genuinely washing out -> decay
    assert PR.assess_persistence([0.06, 0.04, 0.03, 0.025], noise_floor=0.02)["verdict"] == "decay"


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


def test_dry_run_decision_arith_two_phase(tmp_path):
    """--corpus decision_arith reuses the same run_lineage two-phase flow + gap(t) summary."""
    args = PR.build_parser().parse_args([
        "--dry-run", "--corpus", "decision_arith", "--digits-a", "2", "--digits-aprime", "3",
        "--n-train", "16", "--n-eval", "8", "--phase1-gens", "2", "--phase2-gens", "3",
        "--output-dir", str(tmp_path / "e1"),
    ])
    summary = PR.run(args, log_fn=None)
    assert set(summary["lineages"]) == {"infected", "healthy"}
    assert len(summary["gap"]) == 3
    assert summary["persistence"]["verdict"] == "amplify"
    cfg = summary["config"]
    assert (cfg["corpus"], cfg["digits_a"], cfg["digits_aprime"]) == ("decision_arith", 2, 3)
    calls = PR._FAKE_TRAIN_CALLS
    assert all(c["lam"] > 0 for c in calls
               if "infected" in c["output_dir"] and "phase1_A" in c["output_dir"])
    assert all(c["lam"] == 0 for c in calls if "phase2_Aprime" in c["output_dir"])


def test_ambertrace_config_has_no_arith_keys(tmp_path):
    args = PR.build_parser().parse_args([
        "--dry-run", "--corpus-a", "data/decision_eval_v2_subset.jsonl",
        "--corpus-aprime", "data/decision_eval_v2_subset.jsonl",
        "--phase1-gens", "1", "--phase2-gens", "1", "--output-dir", str(tmp_path / "e1")])
    cfg = PR.run(args, log_fn=None)["config"]
    assert "corpus" not in cfg and "digits_a" not in cfg


def _gate(khats, **kw):
    calls = []

    def run_round(parent, gens, idx):
        calls.append((parent, gens, idx))
        return Path(f"ext{idx}"), khats[min(idx - 1, len(khats) - 1)]

    base = dict(start_khat=0.5, start_gens=6, start_adapter=Path("p0"),
                target_khat=0.8, extend_gens=4, max_gens=24)
    base.update(kw)
    return PR.run_install_gate(run_round, **base), calls


def test_install_gate_passes_without_extension():
    res, calls = _gate([0.9], start_khat=0.85)
    assert res.caught and res.gens_used == 6 and not calls
    assert res.final_adapter == Path("p0") and res.khat_trajectory == [0.85]


def test_install_gate_extends_then_catches():
    res, calls = _gate([0.6, 0.85])
    assert res.caught and res.gens_used == 14 and len(calls) == 2
    assert len(res.khat_trajectory) == 3 and res.final_khat == 0.85
    assert res.final_adapter == Path("ext2") and calls[1][0] == Path("ext1")


def test_install_gate_hits_cap_not_caught():
    res, calls = _gate([0.6])
    assert not res.caught and res.gens_used == 24
    assert res.khat_trajectory == [0.5] + [0.6] * len(calls)


def test_install_gate_clamps_last_extension_to_max_gens():
    res, calls = _gate([0.6], start_gens=6, extend_gens=5, max_gens=18)
    assert [c[1] for c in calls] == [5, 5, 2]
    assert res.gens_used == 18 and not res.caught


def test_install_gate_dry_run_records_install(tmp_path):
    args = PR.build_parser().parse_args([
        "--dry-run", "--corpus-a", "data/decision_eval_v2_subset.jsonl",
        "--corpus-aprime", "data/decision_eval_v2_subset.jsonl",
        "--phase1-gens", "1", "--phase2-gens", "1", "--output-dir", str(tmp_path / "e1")])
    lin = PR.run(args, log_fn=None)["lineages"]
    assert lin["infected"]["install_caught"] and lin["infected"]["install_gens_used"] == 1
    assert "install_caught" not in lin["healthy"]


def test_install_gate_aborts_before_phase2_when_target_unreachable(tmp_path):
    """Unreachable target -> gate extends to the cap, then aborts before phase 2 (no vacuous run)."""
    import pytest
    args = PR.build_parser().parse_args([
        "--dry-run", "--corpus-a", "data/decision_eval_v2_subset.jsonl",
        "--corpus-aprime", "data/decision_eval_v2_subset.jsonl",
        "--phase1-gens", "1", "--phase2-gens", "1",
        "--install-target-khat", "0.99", "--install-max-gens", "9",
        "--output-dir", str(tmp_path / "e1")])
    with pytest.raises(SystemExit, match="install did not reach"):
        PR.run(args, log_fn=None)


def test_dry_run_builds_two_ancestors_and_wires_each_lineage(tmp_path):
    """Infected phase 1 starts from the hack-seeded SFT, healthy from the clean SFT."""
    args = PR.build_parser().parse_args([
        "--dry-run",
        "--corpus-a", "data/decision_eval_v2_subset.jsonl",
        "--corpus-aprime", "data/decision_eval_v2_subset.jsonl",
        "--phase1-gens", "2", "--phase2-gens", "2",
        "--output-dir", str(tmp_path / "e1"),
    ])
    assert args.install_sft_fraction == 1.0
    summary = PR.run(args, log_fn=None)
    out = tmp_path / "e1"
    assert (out / "sft_clean" / "adapter").exists()
    assert (out / "sft_hacked" / "adapter").exists()
    assert not (out / "sft").exists()
    assert summary["sft_clean_adapter"] == str(out / "sft_clean" / "adapter")
    assert summary["sft_hacked_adapter"] == str(out / "sft_hacked" / "adapter")
    assert summary["config"]["install_sft_fraction"] == 1.0
    calls = PR._FAKE_TRAIN_CALLS
    inf0 = [c for c in calls if "infected" in c["output_dir"] and "phase1_A" in c["output_dir"]][0]
    hea0 = [c for c in calls if "healthy" in c["output_dir"] and "phase1_A" in c["output_dir"]][0]
    assert inf0["parent_adapter"] == summary["sft_hacked_adapter"]
    assert hea0["parent_adapter"] == summary["sft_clean_adapter"]


def test_install_sft_fraction_validation_rejects_out_of_range(monkeypatch, tmp_path):
    import pytest
    for bad in ("-0.1", "1.5"):
        monkeypatch.setattr(sys, "argv", [
            "persistence_race.py", "--dry-run",
            "--corpus-a", "data/decision_eval_v2_subset.jsonl",
            "--corpus-aprime", "data/decision_eval_v2_subset.jsonl",
            "--install-sft-fraction", bad, "--output-dir", str(tmp_path / "e1")])
        with pytest.raises(SystemExit, match="install-sft-fraction"):
            PR.main()
