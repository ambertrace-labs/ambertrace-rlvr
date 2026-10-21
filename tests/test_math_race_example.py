"""Offline dry-run test for examples/math_race.py (fakes; no model/network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def _dry_args(tmp_path, *extra):
    return build_parser().parse_args(
        ["--dry-run", "--generations", "1", "--output-dir", str(tmp_path), *extra])


def test_dry_run_resume_reuses_sft_and_warm_eval(tmp_path, capsys):
    first = run(_dry_args(tmp_path))
    capsys.readouterr()  # clear
    second = run(_dry_args(tmp_path))  # same config -> resume
    out = capsys.readouterr().out
    assert "resume: reusing warm-start SFT adapter" in out
    assert "resume: reusing warm-start eval" in out
    assert second["warm_start_eval"] == first["warm_start_eval"]  # reused, identical


def test_resume_refuses_on_config_change(tmp_path):
    run(_dry_args(tmp_path, "--seed", "0"))
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--seed", "1"))  # changed fingerprint -> refuse


def test_sft_hyperparams_in_fingerprint(tmp_path):
    # a changed SFT hyperparameter (not just seed) must also refuse resume
    run(_dry_args(tmp_path, "--sft-lr", "2e-5"))
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--sft-lr", "1e-5"))


def test_no_resume_recomputes(tmp_path, capsys):
    run(_dry_args(tmp_path))
    capsys.readouterr()
    run(_dry_args(tmp_path, "--no-resume"))
    assert "resume: reusing" not in capsys.readouterr().out
