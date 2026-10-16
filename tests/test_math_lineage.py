"""Offline tests for the M2 lineage loop (epic #127). No MLX, no network.

Exercises the multi-generation, 3-arm lineage orchestration with fake train/eval
callables, validating adapter chaining, two-phase scrub splits, compute accounting,
metric trajectories, and the compounding verdict assessment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ambertrace_rlvr.math_env import (
    ArmConfig,
    DEFAULT_ARMS,
    LineageResult,
    MathEvalReport,
    run_lineage,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeTrainOneGeneration:
    """Writes a ``b"fake"`` adapter file and logs every call's kwargs."""

    def __init__(self) -> None:
        self.call_log: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        parent_adapter: Path | None,
        output_dir: Path,
        lam: float,
        max_iters: int,
        step_offset: int,
    ) -> Path:
        self.call_log.append({
            "parent_adapter": parent_adapter,
            "output_dir": output_dir,
            "lam": lam,
            "max_iters": max_iters,
            "step_offset": step_offset,
        })
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = output_dir / "adapters.safetensors"
        adapter_path.write_bytes(b"fake")
        return adapter_path


class FakeEvalGeneration:
    """Returns scheduled reports keyed by ``(arm, generation, label)``, or an
    all-zero default. Derives arm+generation from the output_dir path parts."""

    def __init__(
        self, schedule: dict[tuple[str, int, str], MathEvalReport] | None = None,
    ) -> None:
        self.schedule = schedule or {}
        self.call_log: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        adapter_path: Path | None,
        output_dir: Path,
        lam: float,
        label: str,
    ) -> MathEvalReport:
        # Parse arm and generation from output_dir: .../arm_name/genN/...
        # Walk up until we find a genN component.
        parts = output_dir.parts
        arm_name = ""
        generation = 0
        for i, part in enumerate(parts):
            if part.startswith("gen") and part[3:].isdigit():
                generation = int(part[3:])
                arm_name = parts[i - 1]
                break

        self.call_log.append({
            "adapter_path": adapter_path,
            "output_dir": output_dir,
            "lam": lam,
            "label": label,
            "arm": arm_name,
            "generation": generation,
        })

        key = (arm_name, generation, label)
        if key in self.schedule:
            return self.schedule[key]

        return MathEvalReport(
            n=8, k=0.0, k_hat=0.0, wedge=0.0, m=0.0, mean_reward=0.0, lam=lam,
        )


def _default_run(tmp_path: Path, generations: int = 3, **kwargs: Any) -> LineageResult:
    """Convenience: run lineage with fakes and defaults."""
    train_fn = kwargs.pop("train_fn", FakeTrainOneGeneration())
    eval_fn = kwargs.pop("eval_fn", FakeEvalGeneration())
    return run_lineage(
        arms=kwargs.pop("arms", DEFAULT_ARMS),
        generations=generations,
        iters_per_gen=kwargs.pop("iters_per_gen", 20),
        group_size=kwargs.pop("group_size", 4),
        batch_size=kwargs.pop("batch_size", 2),
        train_fn=train_fn,
        eval_fn=eval_fn,
        output_root=tmp_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_arm_scheduling_three_arms(tmp_path: Path) -> None:
    """3 arms x 3 gens; train_fn call count = pure(3)+poisoned(3)+scrubbed(2x3)=12;
    lam per call is correct."""
    train_fn = FakeTrainOneGeneration()
    eval_fn = FakeEvalGeneration()
    run_lineage(
        arms=DEFAULT_ARMS, generations=3, iters_per_gen=20,
        train_fn=train_fn, eval_fn=eval_fn, output_root=tmp_path,
    )
    assert len(train_fn.call_log) == 12  # 3 + 3 + 6

    # Check lam values: pure calls have lam=0.0, poisoned have 0.3,
    # scrubbed alternates 0.3 then 0.0 per generation.
    pure_calls = [c for c in train_fn.call_log if "pure" in str(c["output_dir"])]
    poisoned_calls = [c for c in train_fn.call_log if "poisoned" in str(c["output_dir"])]
    scrubbed_calls = [c for c in train_fn.call_log if "scrubbed" in str(c["output_dir"])]

    assert len(pure_calls) == 3
    assert len(poisoned_calls) == 3
    assert len(scrubbed_calls) == 6

    assert all(c["lam"] == 0.0 for c in pure_calls)
    assert all(c["lam"] == 0.3 for c in poisoned_calls)
    # Scrubbed: pairs of (0.3, 0.0) per generation.
    for i in range(0, 6, 2):
        assert scrubbed_calls[i]["lam"] == 0.3
        assert scrubbed_calls[i + 1]["lam"] == 0.0


def test_scrubbed_arm_two_phase_split(tmp_path: Path) -> None:
    """Scrubbed arm calls train twice/gen (lam=0.3 then lam=0.0); iters split
    matches scrub_fraction (round((1-0.4)*20)=12 and 8)."""
    train_fn = FakeTrainOneGeneration()
    eval_fn = FakeEvalGeneration()
    run_lineage(
        arms=[ArmConfig("scrubbed", 0.3, 0.4)], generations=2, iters_per_gen=20,
        train_fn=train_fn, eval_fn=eval_fn, output_root=tmp_path,
    )
    assert len(train_fn.call_log) == 4  # 2 phases x 2 generations

    # Each pair: first is lam phase (12 iters), second is clean phase (8 iters).
    for gen in range(2):
        lam_call = train_fn.call_log[gen * 2]
        clean_call = train_fn.call_log[gen * 2 + 1]
        assert lam_call["lam"] == 0.3
        assert lam_call["max_iters"] == 12
        assert clean_call["lam"] == 0.0
        assert clean_call["max_iters"] == 8


def test_compute_accounting(tmp_path: Path) -> None:
    """compute_c monotonic increasing; equal across arms at each gen;
    == (gen+1)*iters_per_gen*group_size*batch_size."""
    result = _default_run(tmp_path, generations=3, iters_per_gen=20, group_size=4, batch_size=2)

    for arm_name, arm_result in result.arms.items():
        c_traj = arm_result.c_trajectory()
        # Monotonic increasing.
        for i in range(1, len(c_traj)):
            assert c_traj[i] > c_traj[i - 1], f"{arm_name} compute not increasing at gen {i}"

        # Correct values: (gen+1) * 20 * 4 * 2
        for gen, c in enumerate(c_traj):
            expected = (gen + 1) * 20 * 4 * 2
            assert c == expected, f"{arm_name} gen {gen}: expected {expected}, got {c}"

    # Equal across arms at each generation.
    arm_names = list(result.arms.keys())
    for gen in range(3):
        values = [result.arms[a].generations[gen].compute_c for a in arm_names]
        assert len(set(values)) == 1, f"gen {gen} compute not equal: {values}"


def test_adapter_chaining(tmp_path: Path) -> None:
    """Gen N parent_adapter == gen N-1 child_adapter; gen 0 parent is None."""
    result = _default_run(tmp_path, generations=3)

    for arm_name, arm_result in result.arms.items():
        gens = arm_result.generations
        assert gens[0].parent_adapter is None, f"{arm_name} gen 0 parent not None"
        for i in range(1, len(gens)):
            assert gens[i].parent_adapter == gens[i - 1].child_adapter, (
                f"{arm_name} gen {i} parent != gen {i-1} child"
            )


def test_step_offset_accumulates(tmp_path: Path) -> None:
    """step_offset increments by iters_per_gen each gen."""
    iters_per_gen = 20
    result = _default_run(tmp_path, generations=4, iters_per_gen=iters_per_gen)

    for arm_name, arm_result in result.arms.items():
        for gen, rec in enumerate(arm_result.generations):
            expected = gen * iters_per_gen
            assert rec.step_offset == expected, (
                f"{arm_name} gen {gen}: expected offset {expected}, got {rec.step_offset}"
            )


def test_metric_trajectories(tmp_path: Path) -> None:
    """Scripted eval schedule; assert k/k_hat/m/wedge/c trajectories correct."""
    schedule: dict[tuple[str, int, str], MathEvalReport] = {}
    for gen in range(3):
        k_val = 0.1 * (gen + 1)
        schedule[("pure", gen, "post")] = MathEvalReport(
            n=8, k=k_val, k_hat=k_val, wedge=0.0, m=0.0,
            mean_reward=k_val, lam=0.0,
        )
        schedule[("poisoned", gen, "post")] = MathEvalReport(
            n=8, k=0.05 * (gen + 1), k_hat=0.3 * (gen + 1), wedge=0.25 * (gen + 1),
            m=0.2 * (gen + 1), mean_reward=0.1, lam=0.3,
        )

    eval_fn = FakeEvalGeneration(schedule)
    train_fn = FakeTrainOneGeneration()
    result = run_lineage(
        arms=[ArmConfig("pure", 0.0), ArmConfig("poisoned", 0.3)],
        generations=3, iters_per_gen=20, train_fn=train_fn, eval_fn=eval_fn,
        output_root=tmp_path,
    )

    def _approx(actual: list[float], expected: list[float]) -> None:
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-9, f"{a} != {e}"

    _approx(result.arms["pure"].k_trajectory(), [0.1, 0.2, 0.3])
    _approx(result.arms["pure"].k_hat_trajectory(), [0.1, 0.2, 0.3])
    _approx(result.arms["pure"].wedge_trajectory(), [0.0, 0.0, 0.0])
    _approx(result.arms["pure"].m_trajectory(), [0.0, 0.0, 0.0])

    _approx(result.arms["poisoned"].k_trajectory(), [0.05, 0.1, 0.15])
    _approx(result.arms["poisoned"].m_trajectory(), [0.2, 0.4, 0.6])

    # c_trajectory for both arms.
    for arm_name in ("pure", "poisoned"):
        c = result.arms[arm_name].c_trajectory()
        assert c == [160.0, 320.0, 480.0]


def test_compounding_detected(tmp_path: Path) -> None:
    """Schedule where pure K rises above poisoned by > noise_floor at final gen;
    assert q1_slowdown True, q1_k_gap_final > 0."""
    schedule: dict[tuple[str, int, str], MathEvalReport] = {}
    for gen in range(4):
        schedule[("pure", gen, "post")] = MathEvalReport(
            n=8, k=0.1 * (gen + 1), k_hat=0.1 * (gen + 1), wedge=0.0, m=0.0,
            mean_reward=0.1, lam=0.0,
        )
        schedule[("pure", gen, "pre")] = MathEvalReport(
            n=8, k=0.0, k_hat=0.0, wedge=0.0, m=0.0, mean_reward=0.0, lam=0.0,
        )
        schedule[("poisoned", gen, "post")] = MathEvalReport(
            n=8, k=0.05 * (gen + 1), k_hat=0.3, wedge=0.25, m=0.2,
            mean_reward=0.1, lam=0.3,
        )
        schedule[("poisoned", gen, "pre")] = MathEvalReport(
            n=8, k=0.0, k_hat=0.0, wedge=0.0, m=0.0, mean_reward=0.0, lam=0.3,
        )

    eval_fn = FakeEvalGeneration(schedule)
    train_fn = FakeTrainOneGeneration()
    result = run_lineage(
        arms=[ArmConfig("pure", 0.0), ArmConfig("poisoned", 0.3)],
        generations=4, iters_per_gen=20, train_fn=train_fn, eval_fn=eval_fn,
        output_root=tmp_path,
    )

    assert result.verdict.q1_slowdown is True
    assert result.verdict.q1_k_gap_final is not None
    assert result.verdict.q1_k_gap_final > 0


def test_null_no_compounding(tmp_path: Path) -> None:
    """Identical K across arms; assert q1_slowdown False."""
    schedule: dict[tuple[str, int, str], MathEvalReport] = {}
    for gen in range(3):
        k_val = 0.1 * (gen + 1)
        for arm_name in ("pure", "poisoned"):
            lam = 0.0 if arm_name == "pure" else 0.3
            schedule[(arm_name, gen, "post")] = MathEvalReport(
                n=8, k=k_val, k_hat=k_val, wedge=0.0, m=0.0,
                mean_reward=k_val, lam=lam,
            )

    eval_fn = FakeEvalGeneration(schedule)
    train_fn = FakeTrainOneGeneration()
    result = run_lineage(
        arms=[ArmConfig("pure", 0.0), ArmConfig("poisoned", 0.3)],
        generations=3, iters_per_gen=20, train_fn=train_fn, eval_fn=eval_fn,
        output_root=tmp_path,
    )

    assert result.verdict.q1_slowdown is False
    assert result.verdict.q1_k_gap_final == 0.0


def test_correctability_curve(tmp_path: Path) -> None:
    """Scrubbed mid/post m decreasing; assert correctability_curve c_m values correct."""
    schedule: dict[tuple[str, int, str], MathEvalReport] = {}
    # mid_eval m values decrease across gens: 0.6, 0.5, 0.4
    # post_eval m values are lower: 0.3, 0.2, 0.1
    for gen in range(3):
        mid_m = 0.6 - 0.1 * gen
        post_m = 0.3 - 0.1 * gen
        schedule[("scrubbed", gen, "mid")] = MathEvalReport(
            n=8, k=0.1, k_hat=0.3, wedge=0.2, m=mid_m, mean_reward=0.1, lam=0.3,
        )
        schedule[("scrubbed", gen, "post")] = MathEvalReport(
            n=8, k=0.1, k_hat=0.2, wedge=0.1, m=post_m, mean_reward=0.1, lam=0.3,
        )
        # Need poisoned too for Q3 to be assessable (optional; scrubbed alone is enough
        # for the correctability curve).
        schedule[("poisoned", gen, "post")] = MathEvalReport(
            n=8, k=0.05, k_hat=0.3, wedge=0.25, m=0.5, mean_reward=0.1, lam=0.3,
        )

    eval_fn = FakeEvalGeneration(schedule)
    train_fn = FakeTrainOneGeneration()
    result = run_lineage(
        arms=[ArmConfig("scrubbed", 0.3, 0.4), ArmConfig("poisoned", 0.3)],
        generations=3, iters_per_gen=20, train_fn=train_fn, eval_fn=eval_fn,
        output_root=tmp_path,
    )

    curve = result.verdict.correctability_curve
    assert curve is not None
    assert len(curve) == 3

    for gen in range(3):
        mid_m = 0.6 - 0.1 * gen
        post_m = 0.3 - 0.1 * gen
        expected_c_m = (mid_m - post_m) / mid_m
        assert curve[gen]["generation"] == float(gen)
        assert abs(curve[gen]["m_pre"] - mid_m) < 1e-9
        assert abs(curve[gen]["m_post"] - post_m) < 1e-9
        assert abs(curve[gen]["c_m"] - expected_c_m) < 1e-9


def test_scrubbed_lowers_m_vs_poisoned(tmp_path: Path) -> None:
    """Schedule scrubbed post m < poisoned post m each gen; assert via m_trajectory()."""
    schedule: dict[tuple[str, int, str], MathEvalReport] = {}
    for gen in range(3):
        schedule[("scrubbed", gen, "post")] = MathEvalReport(
            n=8, k=0.1, k_hat=0.2, wedge=0.1, m=0.1 * (gen + 1),
            mean_reward=0.1, lam=0.3,
        )
        schedule[("poisoned", gen, "post")] = MathEvalReport(
            n=8, k=0.1, k_hat=0.4, wedge=0.3, m=0.3 * (gen + 1),
            mean_reward=0.1, lam=0.3,
        )

    eval_fn = FakeEvalGeneration(schedule)
    train_fn = FakeTrainOneGeneration()
    result = run_lineage(
        arms=[ArmConfig("scrubbed", 0.3, 0.4), ArmConfig("poisoned", 0.3)],
        generations=3, iters_per_gen=20, train_fn=train_fn, eval_fn=eval_fn,
        output_root=tmp_path,
    )

    scrubbed_m = result.arms["scrubbed"].m_trajectory()
    poisoned_m = result.arms["poisoned"].m_trajectory()
    for gen in range(3):
        assert scrubbed_m[gen] < poisoned_m[gen], (
            f"gen {gen}: scrubbed m {scrubbed_m[gen]} not < poisoned m {poisoned_m[gen]}"
        )


def test_lineage_result_as_dict_json_serialisable(tmp_path: Path) -> None:
    """json.dumps(LineageResult.as_dict()) succeeds."""
    result = _default_run(tmp_path, generations=2)
    d = result.as_dict()
    # Must not raise.
    serialised = json.dumps(d)
    assert isinstance(serialised, str)
    # Round-trip: the deserialised structure matches.
    restored = json.loads(serialised)
    assert set(restored.keys()) == {"arms", "verdict", "config"}


def test_lineage_result_from_dict_roundtrip(tmp_path: Path) -> None:
    """LineageResult.from_dict(as_dict(...)) reconstructs an equivalent result — the path
    the multi-seed campaign uses to re-aggregate per-seed summaries read back from disk."""
    # A non-trivial schedule so the metrics/verdict are not all identical zeros.
    schedule = {
        ("pure", 0, "post"): _report(k=0.1, k_hat=0.1, m=0.0),
        ("pure", 1, "post"): _report(k=0.3, k_hat=0.3, m=0.0),
        ("poisoned", 0, "post"): _report(k=0.0, k_hat=0.2, m=0.2),
        ("poisoned", 1, "post"): _report(k=0.0, k_hat=0.4, m=0.4),
    }
    result = _default_run(tmp_path, generations=2, eval_fn=FakeEvalGeneration(schedule))

    restored = LineageResult.from_dict(json.loads(json.dumps(result.as_dict())))

    # Structure + every per-generation metric survives the round-trip.
    assert set(restored.arms) == set(result.arms)
    for name, arm in result.arms.items():
        r_arm = restored.arms[name]
        assert r_arm.config == arm.config
        assert r_arm.k_trajectory() == arm.k_trajectory()
        assert r_arm.k_hat_trajectory() == arm.k_hat_trajectory()
        assert r_arm.m_trajectory() == arm.m_trajectory()
        assert r_arm.wedge_trajectory() == arm.wedge_trajectory()
        assert r_arm.c_trajectory() == arm.c_trajectory()
    assert restored.verdict.as_dict() == result.verdict.as_dict()

    # And the reconstructed result aggregates cleanly (the real use site).
    from ambertrace_rlvr.math_env import aggregate_lineage
    agg = aggregate_lineage([restored, restored])
    assert agg.n_seeds == 2
    assert all(p.stderr == 0.0 for p in agg.arms["poisoned"].m)  # identical seeds


def test_output_dir_layout(tmp_path: Path) -> None:
    """Run into tmp_path; assert pure/gen0/adapters.safetensors and
    scrubbed/gen0/phase_lam + phase_clean exist."""
    _default_run(tmp_path, generations=2)

    # Pure arm: single-phase, adapters in gen dir directly.
    assert (tmp_path / "pure" / "gen0" / "adapters.safetensors").exists()
    assert (tmp_path / "pure" / "gen1" / "adapters.safetensors").exists()

    # Scrubbed arm: two-phase, adapters in phase_lam and phase_clean sub-dirs.
    assert (tmp_path / "scrubbed" / "gen0" / "phase_lam" / "adapters.safetensors").exists()
    assert (tmp_path / "scrubbed" / "gen0" / "phase_clean" / "adapters.safetensors").exists()
    assert (tmp_path / "scrubbed" / "gen1" / "phase_lam" / "adapters.safetensors").exists()
    assert (tmp_path / "scrubbed" / "gen1" / "phase_clean" / "adapters.safetensors").exists()


def test_both_branches_in_summary(tmp_path: Path) -> None:
    """verdict.summary contains both affirmative and negative phrasing for each question."""
    # Schedule where Q1=True so we get affirmative phrasing for Q1 and
    # the negative branch is mentioned as the counterfactual.
    schedule: dict[tuple[str, int, str], MathEvalReport] = {}
    for gen in range(4):
        schedule[("pure", gen, "post")] = MathEvalReport(
            n=8, k=0.2 * (gen + 1), k_hat=0.2 * (gen + 1), wedge=0.0, m=0.0,
            mean_reward=0.2 * (gen + 1), lam=0.0,
        )
        schedule[("poisoned", gen, "post")] = MathEvalReport(
            n=8, k=0.05 * (gen + 1), k_hat=0.3, wedge=0.25, m=0.2,
            mean_reward=0.1, lam=0.3,
        )

    eval_fn = FakeEvalGeneration(schedule)
    train_fn = FakeTrainOneGeneration()
    result = run_lineage(
        arms=[ArmConfig("pure", 0.0), ArmConfig("poisoned", 0.3)],
        generations=4, iters_per_gen=20, train_fn=train_fn, eval_fn=eval_fn,
        output_root=tmp_path,
    )

    summary = result.verdict.summary

    # Q1 both branches.
    assert "Q1 (slowdown): YES" in summary or "Q1 (slowdown): NO" in summary
    assert "noise floor" in summary.lower()
    # The affirmative phrasing references slowdown.
    assert "slow" in summary.lower()
    # The negative branch is mentioned as counterfactual.
    assert "if" in summary.lower()

    # Q2 both branches.
    assert "Q2 (stall):" in summary
    assert "stall" in summary.lower()

    # Q3 both branches.
    assert "Q3 (scrub advantage):" in summary
    assert "scrub" in summary.lower()


# ---------------------------------------------------------------------------
# Q3 (scrub advantage) — both branches (cold-review MEDIUM-1)
# ---------------------------------------------------------------------------
def _report(k: float = 0.0, k_hat: float = 0.0, m: float = 0.0,
            lam: float = 0.3) -> MathEvalReport:
    return MathEvalReport(n=8, k=k, k_hat=k_hat, wedge=k_hat - k, m=m,
                          mean_reward=0.0, lam=lam)


def _two_arm_run(tmp_path: Path, schedule: dict[tuple[str, int, str], MathEvalReport],
                 generations: int) -> LineageResult:
    arms = [ArmConfig("poisoned", 0.3, 0.0), ArmConfig("scrubbed", 0.3, 0.4)]
    return run_lineage(
        arms=arms, generations=generations, iters_per_gen=20, group_size=4, batch_size=2,
        train_fn=FakeTrainOneGeneration(), eval_fn=FakeEvalGeneration(schedule),
        output_root=tmp_path,
    )


def test_q3_scrub_advantage_true(tmp_path: Path) -> None:
    # poisoned reaches its max K (0.25) only at gen1 (C=320); scrubbed reaches it at
    # gen0 (C=160) -> scrubbed cheaper -> advantage True.
    schedule = {
        ("poisoned", 0, "post"): _report(k=0.0),
        ("poisoned", 1, "post"): _report(k=0.25),
        ("scrubbed", 0, "post"): _report(k=0.25),
        ("scrubbed", 1, "post"): _report(k=0.25),
    }
    v = _two_arm_run(tmp_path, schedule, generations=2).verdict
    assert v.q3_scrub_advantage is True
    assert v.q3_c_at_k == {"poisoned": 320.0, "scrubbed": 160.0}


def test_q3_scrub_advantage_false(tmp_path: Path) -> None:
    # Mirror image: poisoned reaches target K (0.25) first (gen0, C=160); scrubbed only
    # at gen1 (C=320) -> no advantage.
    schedule = {
        ("poisoned", 0, "post"): _report(k=0.25),
        ("poisoned", 1, "post"): _report(k=0.25),
        ("scrubbed", 0, "post"): _report(k=0.0),
        ("scrubbed", 1, "post"): _report(k=0.25),
    }
    v = _two_arm_run(tmp_path, schedule, generations=2).verdict
    assert v.q3_scrub_advantage is False
    assert v.q3_c_at_k == {"poisoned": 160.0, "scrubbed": 320.0}


# ---------------------------------------------------------------------------
# Q2 (stall) — threshold value and the no-stall branch (cold-review LOW-d)
# ---------------------------------------------------------------------------
def _poisoned_run(tmp_path: Path, schedule: dict[tuple[str, int, str], MathEvalReport],
                  generations: int) -> LineageResult:
    return run_lineage(
        arms=[ArmConfig("poisoned", 0.3, 0.0)], generations=generations, iters_per_gen=20,
        train_fn=FakeTrainOneGeneration(), eval_fn=FakeEvalGeneration(schedule),
        output_root=tmp_path,
    )


def test_q2_stall_threshold_is_m_at_onset(tmp_path: Path) -> None:
    # K rises to 0.25 at gen1 then stalls (flat) for gen2, gen3 -> stall onset at gen2;
    # q2_threshold is the poisoned m recorded at gen2.
    schedule = {
        ("poisoned", 0, "post"): _report(k=0.0, m=0.1),
        ("poisoned", 1, "post"): _report(k=0.25, m=0.25),
        ("poisoned", 2, "post"): _report(k=0.25, m=0.375),
        ("poisoned", 3, "post"): _report(k=0.25, m=0.5),
    }
    v = _poisoned_run(tmp_path, schedule, generations=4).verdict
    assert v.q2_stall_detected is True
    assert v.q2_threshold == 0.375


def test_q2_no_stall_when_k_keeps_rising(tmp_path: Path) -> None:
    schedule = {
        ("poisoned", 0, "post"): _report(k=0.0),
        ("poisoned", 1, "post"): _report(k=0.25),
        ("poisoned", 2, "post"): _report(k=0.5),
    }
    v = _poisoned_run(tmp_path, schedule, generations=3).verdict
    assert v.q2_stall_detected is False
    assert v.q2_threshold is None


# ---------------------------------------------------------------------------
# ArmConfig validation (cold-review MEDIUM-2)
# ---------------------------------------------------------------------------
def test_arm_config_rejects_out_of_range():
    with pytest.raises(ValueError):
        ArmConfig("bad", lam=0.3, scrub_fraction=1.5)
    with pytest.raises(ValueError):
        ArmConfig("bad", lam=-0.1, scrub_fraction=0.0)
    with pytest.raises(ValueError):
        ArmConfig("bad", lam=1.2, scrub_fraction=0.0)


def test_lineage_initial_adapters(tmp_path: Path) -> None:
    seed = tmp_path / "sft" / "adapter"
    train_fn = FakeTrainOneGeneration()
    arms = [ArmConfig("poisoned", 0.75), ArmConfig("other", 0.5)]
    _default_run(tmp_path / "run", generations=2, arms=arms, train_fn=train_fn,
                 initial_adapters={"poisoned": seed})
    by_arm = {}
    for call in train_fn.call_log:
        arm, gen = call["output_dir"].parent.name, call["output_dir"].name
        by_arm[(arm, gen)] = call["parent_adapter"]
    assert by_arm[("poisoned", "gen0")] == seed
    assert by_arm[("poisoned", "gen1")] != seed  # chains from gen0's child afterwards
    assert by_arm[("poisoned", "gen1")] is not None
    assert by_arm[("other", "gen0")] is None  # unseeded arm unchanged
