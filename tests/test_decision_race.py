"""Offline dry-run test for examples/decision_race.py (fakes; no model/network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from decision_race import build_parser, run  # noqa: E402


def _dry_args(tmp_path, *extra):
    return build_parser().parse_args(
        ["--dry-run", "--generations", "3", "--output-dir", str(tmp_path), *extra])


def test_dry_run_two_arms_from_one_adapter(tmp_path):
    args = _dry_args(tmp_path)
    summary = run(args)
    data = json.loads((tmp_path / "decision_race_summary.json").read_text())
    assert data["config"]["eval_hack"] == "permissive"
    arms = data["lineage"]["arms"]
    assert set(arms) == {"aligned", "poisoned"}  # no scrubbed arm
    for name in arms:
        gens = arms[name]["generations"]
        assert len(gens) == 3
        assert gens[0]["parent_adapter"] == summary["sft_adapter"]
        assert gens[1]["parent_adapter"] == gens[0]["child_adapter"]
    assert arms["aligned"]["config"]["lam"] == 0.0
    assert arms["poisoned"]["config"]["lam"] == args.lam == 0.8
    # fail-open amplifies under the poisoned arm only
    a = [g["post_eval"] for g in arms["aligned"]["generations"]]
    p = [g["post_eval"] for g in arms["poisoned"]["generations"]]
    assert p[-1]["m"] > p[0]["m"] and a[-1]["m"] < a[0]["m"]
    assert a[-1]["k"] > p[-1]["k"]
    verdict = data["lineage"]["verdict"]
    assert verdict["summary"] and verdict["q1_k_gap_final"] is not None
    assert verdict["q3_scrub_advantage"] is None  # degrades gracefully without a scrubbed arm


def test_resume_reuses_and_guards(tmp_path, capsys):
    first = run(_dry_args(tmp_path))
    capsys.readouterr()
    second = run(_dry_args(tmp_path))
    out = capsys.readouterr().out
    assert "resume: reusing warm-start SFT adapter" in out
    assert "resume: reusing warm-start eval" in out
    assert second["warm_start_eval"] == first["warm_start_eval"]
    assert second["lineage"]["arms"].keys() == first["lineage"]["arms"].keys()
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--permissive-fraction", "0.7"))


def test_no_resume_recomputes(tmp_path, capsys):
    run(_dry_args(tmp_path))
    capsys.readouterr()
    run(_dry_args(tmp_path, "--no-resume"))
    assert "resume: reusing" not in capsys.readouterr().out


def test_sft_hyperparams_in_fingerprint(tmp_path):
    # a changed *training* hyperparameter (not just permissive_fraction) must also refuse resume
    run(_dry_args(tmp_path, "--sft-lr", "2e-5"))
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--sft-lr", "1e-5"))


def test_resume_refuses_before_reusing_sft(tmp_path, capsys):
    # the fingerprint guard runs BEFORE SFT/warm-eval reuse, so a changed-config resume refuses
    # even when a prior (interrupted) run already produced the SFT adapter + fingerprint — it
    # does not silently reuse the stale adapter.
    run(_dry_args(tmp_path, "--sft-lr", "2e-5"))
    assert (tmp_path / "sft" / "adapter" / "adapter_config.json").exists()
    capsys.readouterr()
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--sft-lr", "1e-5"))
    assert "resume: reusing warm-start SFT adapter" not in capsys.readouterr().out
