"""Offline tests for the M2 lineage runner's dry-run (#127). No MLX, no network.

Exercises `examples/math_lineage.py`'s `dry_run()`: the full lineage loop driven by
synthetic callbacks, the written summary JSON, and the Q1/Q2/correctability verdict the
synthetic schedule is designed to trigger. The live MLX callbacks are not covered here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_lineage import dry_run  # noqa: E402


def test_dry_run_writes_summary_and_verdict(tmp_path):
    dry_run(generations=4, iters_per_gen=20, output_dir=tmp_path)
    summary_path = tmp_path / "lineage_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert set(data["arms"]) == {"pure", "poisoned", "scrubbed"}
    assert data["verdict"]["summary"]
    # each arm ran all 4 generations
    for arm in data["arms"].values():
        assert len(arm["generations"]) == 4


def test_dry_run_detects_slowdown_and_stall(tmp_path):
    # synthetic schedule: pure gains genuine K, poisoned stays flat at K=0
    result = dry_run(generations=4, iters_per_gen=20, output_dir=tmp_path)
    v = result.verdict
    assert v.q1_slowdown is True            # pure K pulls ahead of poisoned
    assert v.q1_k_gap_final is not None and v.q1_k_gap_final > v.noise_floor
    assert v.q2_stall_detected is True      # poisoned K never rises


def test_dry_run_correctability_curve_populated(tmp_path):
    result = dry_run(generations=4, iters_per_gen=20, output_dir=tmp_path)
    curve = result.verdict.correctability_curve
    assert curve is not None and len(curve) == 4
    # the synthetic clean phase removes half the load each generation
    for point in curve:
        assert abs(point["c_m"] - 0.5) < 1e-9


def test_dry_run_compute_equal_across_arms(tmp_path):
    result = dry_run(generations=3, iters_per_gen=10, output_dir=tmp_path)
    c_pure = result.arms["pure"].c_trajectory()
    c_poisoned = result.arms["poisoned"].c_trajectory()
    c_scrubbed = result.arms["scrubbed"].c_trajectory()
    assert c_pure == c_poisoned == c_scrubbed   # equal per-generation compute


def test_dry_run_multiseed_writes_aggregate(tmp_path, monkeypatch):
    # Part B wiring: --seeds N runs each seed and writes an aggregate (offline).
    import math_lineage
    monkeypatch.setattr(sys, "argv", [
        "math_lineage.py", "--dry-run", "--seeds", "3", "--generations", "3",
        "--output-dir", str(tmp_path),
    ])
    math_lineage.main()
    agg = json.loads((tmp_path / "aggregate" / "lineage_aggregate.json").read_text())
    assert agg["n_seeds"] == 3
    assert set(agg["arms"]) == {"pure", "poisoned", "scrubbed"}
    # dry-run is deterministic, so identical seeds give zero stderr
    assert all(p["stderr"] == 0.0 for p in agg["arms"]["poisoned"]["m"])


def test_make_live_callbacks_rejects_unregistered_model():
    # arch-aware: LoRA targets come from the model registry, so an unknown model
    # fails fast with a clear KeyError (before any MLX import).
    import pytest

    import math_lineage
    with pytest.raises(KeyError):
        math_lineage.make_live_callbacks(
            model_name="nope/not-a-model", ref_model_name=None, group_size=4,
            batch_size=2, beta=0.04, learning_rate=1e-5, max_completion_length=128,
            lora_rank=8,
        )


def test_only_seed_writes_summary_without_aggregating(tmp_path, monkeypatch):
    # The subprocess child path: `--only-seed N` runs exactly one seed into --output-dir,
    # writing that seed's summary and NOT an aggregate (the parent aggregates later).
    import math_lineage
    monkeypatch.setattr(sys, "argv", [
        "math_lineage.py", "--dry-run", "--only-seed", "2", "--generations", "2",
        "--output-dir", str(tmp_path),
    ])
    math_lineage.main()
    assert (tmp_path / "lineage_summary.json").exists()
    assert not (tmp_path / "aggregate").exists()  # no aggregation in child mode


def test_child_cmd_runs_one_seed_into_seed_dir(tmp_path):
    # The orchestrator builds a per-seed child argv: exactly one seed, into seed{S}/,
    # core hyperparameters forwarded, and no --wandb / --ref-model when those are off/unset.
    import math_lineage
    args = SimpleNamespace(
        model="Qwen/Qwen2.5-0.5B", generations=4, iters_per_gen=100, lam=0.3,
        scrub_fraction=0.4, group_size=4, batch_size=2, beta=0.04, learning_rate=5e-6,
        max_completion_length=128, lora_rank=8, ref_model=None, wandb=False,
        wandb_project="ambertrace-rlvr-math", seeds=3,
    )
    seed_dir = tmp_path / "seed2"
    cmd = math_lineage._child_cmd(args, 2, seed_dir, group="camp")
    assert cmd[cmd.index("--only-seed") + 1] == "2"
    assert cmd[cmd.index("--output-dir") + 1] == str(seed_dir)
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen2.5-0.5B"
    assert cmd[cmd.index("--learning-rate") + 1] == str(5e-6)
    assert "--dry-run" not in cmd  # child is a single live seed
    assert "--wandb" not in cmd  # forwarded only when enabled
    assert "--ref-model" not in cmd  # omitted when unset


def test_child_cmd_forwards_wandb_and_ref_model(tmp_path):
    import math_lineage
    args = SimpleNamespace(
        model="m", generations=1, iters_per_gen=1, lam=0.3, scrub_fraction=0.4,
        group_size=4, batch_size=2, beta=0.04, learning_rate=1e-5,
        max_completion_length=128, lora_rank=8, ref_model="ref/model", wandb=True,
        wandb_project="proj", seeds=2,
    )
    cmd = math_lineage._child_cmd(args, 0, tmp_path / "seed0", group="mygroup")
    assert "--wandb" in cmd
    assert cmd[cmd.index("--wandb-project") + 1] == "proj"
    assert cmd[cmd.index("--wandb-group") + 1] == "mygroup"
    assert cmd[cmd.index("--ref-model") + 1] == "ref/model"


def test_live_multiseed_subprocess_aggregates_survivors(tmp_path, monkeypatch):
    # Drive the live orchestrator (_run_multiseed_subprocess) offline: fake out
    # subprocess.run so no real seed/MLX runs. A surviving seed is emulated by writing a
    # valid lineage_summary.json into the child's --output-dir (via the offline dry-run);
    # a crashing seed writes nothing and returns a non-zero exit code. The parent must then
    # aggregate only the survivors (read back via LineageResult.from_dict) and record the
    # crash in seed_failures.json — the whole point of the branch.
    import subprocess

    import math_lineage
    bad_seed = 1

    def fake_run(cmd, check=False):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        seed = int(cmd[cmd.index("--only-seed") + 1])
        assert "--seeds" in cmd and "--dry-run" not in cmd  # live single-seed child argv
        if seed == bad_seed:
            return subprocess.CompletedProcess(cmd, 1)  # crashed: no summary written
        # a real child writes its own summary; emulate it with the offline dry-run.
        math_lineage.dry_run(generations=1, iters_per_gen=1, output_dir=out_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(math_lineage.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "math_lineage.py", "--seeds", "3", "--model", "fake/model",
        "--generations", "1", "--iters-per-gen", "1", "--output-dir", str(tmp_path),
    ])
    math_lineage.main()

    agg = json.loads((tmp_path / "aggregate" / "lineage_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seeds 0 and 2 survived; seed 1 crashed
    assert set(agg["arms"]) == {"pure", "poisoned", "scrubbed"}
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert [f["seed"] for f in failures] == [bad_seed]


def _lineage_argv(tmp_path):
    return ["math_lineage.py", "--seeds", "3", "--model", "fake/model",
            "--generations", "1", "--iters-per-gen", "1", "--output-dir", str(tmp_path)]


def test_live_multiseed_subprocess_ignores_stale_summary(tmp_path, monkeypatch):
    # HIGH: a seed that crashes before writing must NOT be credited with a summary left in
    # its dir by a previous run. Plant a stale seed1 summary, then crash seed1's child.
    import subprocess

    import math_lineage
    stale_dir = tmp_path / "seed1"
    stale_dir.mkdir(parents=True)
    math_lineage.dry_run(generations=1, iters_per_gen=1, output_dir=stale_dir)
    assert (stale_dir / "lineage_summary.json").exists()

    def fake_run(cmd, check=False):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        seed = int(cmd[cmd.index("--only-seed") + 1])
        if seed == 1:
            return subprocess.CompletedProcess(cmd, 1)  # crash; writes nothing
        math_lineage.dry_run(generations=1, iters_per_gen=1, output_dir=out_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(math_lineage.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", _lineage_argv(tmp_path))
    math_lineage.main()

    agg = json.loads((tmp_path / "aggregate" / "lineage_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seed1's stale summary was cleared, so it is a failure
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert [f["seed"] for f in failures] == [1]


def test_live_multiseed_subprocess_isolates_corrupt_summary(tmp_path, monkeypatch):
    # MEDIUM: a valid-JSON-but-non-dict summary isolates that seed (not aborts the campaign).
    import subprocess

    import math_lineage

    def fake_run(cmd, check=False):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        seed = int(cmd[cmd.index("--only-seed") + 1])
        if seed == 0:
            (out_dir / "lineage_summary.json").write_text("123")  # valid JSON, not a dict
            return subprocess.CompletedProcess(cmd, 0)
        math_lineage.dry_run(generations=1, iters_per_gen=1, output_dir=out_dir)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(math_lineage.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", _lineage_argv(tmp_path))
    math_lineage.main()  # must NOT raise despite seed0's corrupt summary

    agg = json.loads((tmp_path / "aggregate" / "lineage_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seeds 1 and 2 survived; seed0 isolated
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert [f["seed"] for f in failures] == [0]


def test_dry_run_seed_isolation_survives_one_failure(tmp_path, monkeypatch):
    # resiliency: one seed crashing must not lose the campaign — the rest aggregate.
    import math_lineage
    real = math_lineage._run_seed

    def flaky(*, arms, args, output_dir, seed=0, log_fn=None):
        if seed == 1:
            raise RuntimeError("boom")
        return real(arms=arms, args=args, output_dir=output_dir, seed=seed, log_fn=log_fn)

    monkeypatch.setattr(math_lineage, "_run_seed", flaky)
    monkeypatch.setattr(sys, "argv", [
        "math_lineage.py", "--dry-run", "--seeds", "3", "--generations", "3",
        "--output-dir", str(tmp_path),
    ])
    math_lineage.main()
    agg = json.loads((tmp_path / "aggregate" / "lineage_aggregate.json").read_text())
    assert agg["n_seeds"] == 2  # seed 1 failed; seeds 0 and 2 survived
    failures = json.loads((tmp_path / "aggregate" / "seed_failures.json").read_text())
    assert failures[0]["seed"] == 1 and "boom" in failures[0]["error"]
