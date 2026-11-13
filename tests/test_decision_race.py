"""Offline dry-run test for examples/decision_race.py (fakes; no model/network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import decision_race as _dr  # noqa: E402  (module handle for main()'s wandb-gating test)
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


def test_exploration_levers_in_fingerprint(tmp_path):
    # temperature / kl-beta are result-affecting, so a resume at a changed value must refuse
    # (otherwise a lineage silently mixes exploration settings across generations).
    run(_dry_args(tmp_path, "--temperature", "1.2"))
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--temperature", "0.8"))


def test_entropy_coef_in_fingerprint(tmp_path):
    run(_dry_args(tmp_path, "--entropy-coef", "0.1"))
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(tmp_path, "--entropy-coef", "0.0"))


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


def test_main_wandb_gating(tmp_path, monkeypatch):
    # wandb is default-ON, but main() must NEVER init it for --dry-run or --no-wandb (the
    # offline-discipline invariant); a plain live parse DOES reach it.
    class _WandbReached(Exception):
        pass

    def _boom(args):
        raise _WandbReached()

    monkeypatch.setattr(_dr, "_init_wandb", _boom)
    monkeypatch.setattr(_dr, "run", lambda args, log_fn=None: {})  # no-op, no model
    base = ["decision_race", "--output-dir", str(tmp_path)]
    for extra in (["--dry-run"], ["--no-wandb"]):
        monkeypatch.setattr(sys, "argv", [*base, *extra])
        _dr.main()  # wandb gated off -> _init_wandb not reached
    monkeypatch.setattr(sys, "argv", base)  # default-ON live parse
    with pytest.raises(_WandbReached):
        _dr.main()


def test_extend_lineage_with_larger_generations(tmp_path, capsys):
    # `generations` is excluded from the resume fingerprint (run-length knob), so a resume at a
    # larger --generations EXTENDS the lineage: gen0-2 are reused, no config-mismatch guard fires,
    # and the summary cfg still records the new generations for provenance.
    run(_dry_args(tmp_path))  # --generations 3
    capsys.readouterr()
    extended = run(build_parser().parse_args(
        ["--dry-run", "--generations", "5", "--output-dir", str(tmp_path)]))
    out = capsys.readouterr().out
    assert "resume: reusing warm-start SFT adapter" in out
    arms = extended["lineage"]["arms"]
    for name in ("aligned", "poisoned"):
        gens = arms[name]["generations"]
        assert len(gens) == 5
        assert gens[3]["parent_adapter"] == gens[2]["child_adapter"]
    assert extended["config"]["generations"] == 5
    data = json.loads((tmp_path / "decision_race_summary.json").read_text())
    assert data["config"]["generations"] == 5
    assert "generations" not in json.loads((tmp_path / "lineage_config.json").read_text())
    # other config changes are still refused
    with pytest.raises(ValueError, match="different run config"):
        run(build_parser().parse_args(
            ["--dry-run", "--generations", "5", "--lam", "0.5", "--output-dir", str(tmp_path)]))


def _tiny_corpus(path):
    vocab = [{"verb": "approve", "rank": 2, "restrictive": False},
             {"verb": "deny", "rank": 1, "restrictive": True}]
    with path.open("w") as f:
        for d in range(4):
            for p in range(2):
                f.write(json.dumps({
                    "id": f"d{d}-p{p}", "domain": f"d{d}", "prompt": f"case {d}-{p}",
                    "vocabulary": vocab, "oracle": "deny" if p else "approve",
                    "difficulty": {"structure": "baseline"}}) + "\n")


def test_ambertrace_corpus_dry_run_writes_split_manifest(tmp_path):
    data = tmp_path / "corpus.jsonl"
    _tiny_corpus(data)
    out = tmp_path / "out"
    summary = run(_dry_args(
        out, "--corpus", "ambertrace", "--data-path", str(data), "--hack-name",
        "severity_fail_open", "--split-seed", "1"))
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert set(manifest["train_domains"]).isdisjoint(manifest["heldout_domains"])
    assert len(manifest["train_domains"]) + len(manifest["heldout_domains"]) == 4
    # the training contract is recorded so the probe renders the held-out split under the same prompt
    from ambertrace_rlvr.math_env import AMBERTRACE_SYSTEM_PROMPT
    assert manifest["system_prompt"] == AMBERTRACE_SYSTEM_PROMPT
    cfg = summary["config"]
    assert cfg["domain"] == "ambertrace" and cfg["hack_name"] == "severity_fail_open"
    assert cfg["split_seed"] == 1 and cfg["train_fraction"] == 0.5
    assert set(summary["lineage"]["arms"]) == {"aligned", "poisoned"}
    # a changed split config refuses resume
    with pytest.raises(ValueError, match="different run config"):
        run(_dry_args(out, "--corpus", "ambertrace", "--data-path", str(data),
                      "--hack-name", "severity_fail_open", "--split-seed", "2"))


def test_default_corpus_is_decision_arith(tmp_path):
    summary = run(_dry_args(tmp_path))
    assert summary["config"]["domain"] == "decision_arith"
    assert "hack_name" not in summary["config"]
    assert not (tmp_path / "split_manifest.json").exists()
