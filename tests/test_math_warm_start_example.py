"""Offline dry-run test for examples/math_warm_start_induction.py (fakes; no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def _dry_args(tmp_path, *extra):
    return build_parser().parse_args(
        ["--dry-run", "--generations", "1", "--output-dir", str(tmp_path), *extra])


def test_resume_reuses_sft_and_warm_eval(tmp_path, capsys):
    first = run(_dry_args(tmp_path))
    capsys.readouterr()
    second = run(_dry_args(tmp_path))  # same config -> resume
    out = capsys.readouterr().out
    assert "resume: reusing warm-start SFT adapter" in out
    assert "resume: reusing warm-start eval" in out
    assert second["warm_start_eval"] == first["warm_start_eval"]


def test_resume_refuses_on_config_change(tmp_path):
    run(_dry_args(tmp_path, "--seed", "0"))
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--seed", "1"))


def test_induction_sft_knob_in_fingerprint(tmp_path):
    # an induction-only SFT knob (sft_epochs) must also be fingerprinted
    run(_dry_args(tmp_path, "--sft-epochs", "0.5"))
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--sft-epochs", "1.0"))


def test_resume_refuses_before_reusing_sft(tmp_path, capsys):
    # H1 regression: the guard runs before SFT reuse, so a changed-config resume refuses
    # even when a prior (interrupted) run already produced the SFT adapter + fingerprint.
    run(_dry_args(tmp_path, "--sft-lr", "2e-5"))
    assert (tmp_path / "sft" / "adapter" / "adapter_config.json").exists()
    capsys.readouterr()
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--sft-lr", "1e-5"))
    assert "resume: reusing warm-start SFT adapter" not in capsys.readouterr().out


def test_no_resume_recomputes(tmp_path, capsys):
    run(_dry_args(tmp_path))
    capsys.readouterr()
    run(_dry_args(tmp_path, "--no-resume"))
    assert "resume: reusing" not in capsys.readouterr().out
