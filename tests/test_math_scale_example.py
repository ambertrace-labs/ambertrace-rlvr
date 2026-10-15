"""Offline tests for the M3 scale-ladder runner's dry-run (#127). No MLX, no network.

Exercises `examples/math_scale_ladder.py`'s `dry_run()`: the full synthetic sweep over
both architectures, the written summary JSON, and the scale-dependence verdict the
synthetic schedule (m grows with parameter count) is designed to trigger. The live MLX
callbacks are not covered here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_scale_ladder import dry_run  # noqa: E402

from ambertrace_rlvr.math_env import ALL_MODELS  # noqa: E402


def test_dry_run_writes_summary_and_verdicts(tmp_path):
    dry_run(output_dir=tmp_path)
    summary_path = tmp_path / "scale_sweep_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert len(data["points"]) == len(ALL_MODELS)
    # the five scale-trend metrics are all present
    assert set(data["verdicts"]) == {"m", "k_hat", "wedge", "m_gain", "k_hat_gain"}


def test_dry_run_detects_increasing_m_trend(tmp_path):
    result = dry_run(output_dir=tmp_path)
    # synthetic schedule: m rises with parameter count -> increasing trend
    assert result.verdicts["m"].trend == "increasing"
    assert result.verdicts["k_hat"].trend == "increasing"


def test_dry_run_both_architectures_present(tmp_path):
    result = dry_run(output_dir=tmp_path)
    archs = {p.arch for p in result.points}
    assert archs == {"gpt_neox", "qwen2"}


def test_dry_run_per_arch_verdicts_populated(tmp_path):
    result = dry_run(output_dir=tmp_path)
    assert result.pythia_verdicts is not None
    assert result.qwen_verdicts is not None
    # each family trends increasing in m under the synthetic schedule
    assert result.pythia_verdicts["m"].trend == "increasing"
    assert result.qwen_verdicts["m"].trend == "increasing"


def test_dry_run_multiseed_writes_aggregate(tmp_path, monkeypatch):
    # Part B wiring: --seeds N runs each seed and writes a scale aggregate (offline).
    import math_scale_ladder
    monkeypatch.setattr(sys, "argv", [
        "math_scale_ladder.py", "--dry-run", "--seeds", "2", "--models", "pythia",
        "--output-dir", str(tmp_path),
    ])
    math_scale_ladder.main()
    agg = json.loads((tmp_path / "aggregate" / "scale_aggregate.json").read_text())
    assert agg["n_seeds"] == 2
    assert len(agg["points"]) == 4  # the four-model Pythia ladder
    assert agg["excluded_models"] == []


def test_only_seed_writes_summary_without_aggregating(tmp_path, monkeypatch):
    # The subprocess child path: `--only-seed N` runs exactly one seed's sweep into
    # --output-dir, writing that seed's summary and NOT an aggregate.
    import math_scale_ladder
    monkeypatch.setattr(sys, "argv", [
        "math_scale_ladder.py", "--dry-run", "--only-seed", "1", "--models", "pythia",
        "--output-dir", str(tmp_path),
    ])
    math_scale_ladder.main()
    assert (tmp_path / "scale_sweep_summary.json").exists()
    assert not (tmp_path / "aggregate").exists()  # no aggregation in child mode


def test_child_cmd_runs_one_seed_into_seed_dir(tmp_path):
    # The orchestrator builds a per-seed child argv: exactly one seed, into seed{S}/,
    # the model ladder + hyperparameters forwarded, no --wandb when disabled.
    import math_scale_ladder
    args = SimpleNamespace(
        models="qwen", lam=0.3, hack="answer_copy", max_iters=100, group_size=4,
        batch_size=2, beta=0.04, learning_rate=5e-6, max_completion_length=128,
        lora_rank=8, wandb=False, wandb_project="ambertrace-rlvr-math", seeds=3,
    )
    seed_dir = tmp_path / "seed2"
    cmd = math_scale_ladder._child_cmd(args, 2, seed_dir, group="camp")
    assert cmd[cmd.index("--only-seed") + 1] == "2"
    assert cmd[cmd.index("--output-dir") + 1] == str(seed_dir)
    assert cmd[cmd.index("--models") + 1] == "qwen"
    assert cmd[cmd.index("--hack") + 1] == "answer_copy"  # hack kind forwarded to child
    assert cmd[cmd.index("--learning-rate") + 1] == str(5e-6)
    assert "--dry-run" not in cmd  # child is a single live seed
    assert "--wandb" not in cmd  # forwarded only when enabled


def test_child_cmd_forwards_wandb(tmp_path):
    import math_scale_ladder
    args = SimpleNamespace(
        models="qwen", lam=0.3, hack="answer_copy", max_iters=1, group_size=4,
        batch_size=2, beta=0.04, learning_rate=1e-5, max_completion_length=128,
        lora_rank=8, wandb=True, wandb_project="proj", seeds=2,
    )
    cmd = math_scale_ladder._child_cmd(args, 0, tmp_path / "seed0", group="mygroup")
    assert "--wandb" in cmd
    assert cmd[cmd.index("--wandb-project") + 1] == "proj"
    assert cmd[cmd.index("--wandb-group") + 1] == "mygroup"


def test_live_multiseed_subprocess_aggregates_survivors(tmp_path, monkeypatch):
    # Drive the live orchestrator (_run_multiseed_subprocess) offline: fake out
    # subprocess.run so no real seed/MLX runs. A surviving seed is emulated by writing a
    # valid scale_sweep_summary.json into the child's --output-dir (via the offline
    # dry-run); a crashing seed writes nothing and returns a non-zero exit code. The parent
    # must aggregate only the survivors (read back via ScaleSweepResult.from_dict) and
    # record the crash in seed_failures.json.
    import subprocess

    import math_scale_ladder
    bad_seed = 1

    def fake_run(cmd, check=False):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        seed = int(cmd[cmd.index("--only-seed") + 1])
        models_arg = cmd[cmd.index("--models") + 1]
        assert "--seeds" in cmd and "--dry-run" not in cmd  # live single-seed child argv
        if seed == bad_seed:
            return subprocess.CompletedProcess(cmd, 1)  # crashed: no summary written
        # a real child sweeps the forwarded ladder; emulate it with the offline dry-run.
        math_scale_ladder.dry_run(
            models=math_scale_ladder._resolve_models(models_arg), output_dir=out_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(math_scale_ladder.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "math_scale_ladder.py", "--seeds", "3", "--models", "pythia",
        "--max-iters", "1", "--output-dir", str(tmp_path),
    ])
    math_scale_ladder.main()

    agg = json.loads((tmp_path / "aggregate" / "scale_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seeds 0 and 2 survived; seed 1 crashed
    assert len(agg["points"]) == 4  # full Pythia ladder, faithful to forwarded --models
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert [f["seed"] for f in failures] == [bad_seed]


def _scale_argv(tmp_path):
    return ["math_scale_ladder.py", "--seeds", "3", "--models", "pythia",
            "--max-iters", "1", "--output-dir", str(tmp_path)]


def test_live_multiseed_subprocess_ignores_stale_summary(tmp_path, monkeypatch):
    # HIGH: a seed that crashes before writing must NOT be credited with a summary left in
    # its dir by a previous run. Plant a stale seed1 summary, then crash seed1's child.
    import subprocess

    import math_scale_ladder
    stale_dir = tmp_path / "seed1"
    stale_dir.mkdir(parents=True)
    math_scale_ladder.dry_run(models=math_scale_ladder._resolve_models("pythia"),
                              output_dir=stale_dir)
    assert (stale_dir / "scale_sweep_summary.json").exists()

    def fake_run(cmd, check=False):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        seed = int(cmd[cmd.index("--only-seed") + 1])
        if seed == 1:
            return subprocess.CompletedProcess(cmd, 1)  # crash; writes nothing
        math_scale_ladder.dry_run(
            models=math_scale_ladder._resolve_models(cmd[cmd.index("--models") + 1]),
            output_dir=out_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(math_scale_ladder.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", _scale_argv(tmp_path))
    math_scale_ladder.main()

    agg = json.loads((tmp_path / "aggregate" / "scale_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seed1's stale summary was cleared, so it is a failure
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert [f["seed"] for f in failures] == [1]


def test_live_multiseed_subprocess_isolates_corrupt_summary(tmp_path, monkeypatch):
    # MEDIUM: a valid-JSON-but-non-dict summary isolates that seed (not aborts the sweep).
    import subprocess

    import math_scale_ladder

    def fake_run(cmd, check=False):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        seed = int(cmd[cmd.index("--only-seed") + 1])
        if seed == 0:
            (out_dir / "scale_sweep_summary.json").write_text("123")  # valid JSON, not a dict
            return subprocess.CompletedProcess(cmd, 0)
        math_scale_ladder.dry_run(
            models=math_scale_ladder._resolve_models(cmd[cmd.index("--models") + 1]),
            output_dir=out_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(math_scale_ladder.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", _scale_argv(tmp_path))
    math_scale_ladder.main()  # must NOT raise despite seed0's corrupt summary

    agg = json.loads((tmp_path / "aggregate" / "scale_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seeds 1 and 2 survived; seed0 isolated
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert [f["seed"] for f in failures] == [0]
