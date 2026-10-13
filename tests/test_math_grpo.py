"""Offline tests for the M1 math-GRPO example (#127). No MLX, no network.

Exercises the parts of ``examples/math_mlx_grpo.py`` that run without a model: the
reward callable under GRPO's exact calling convention, the dataset-tuple builder,
the trajectory round-trip, and the dry-run wiring that captures K/K̂/m. The MLX
``train`` / ``evaluate_heldout`` paths are live-only and not covered here (same as
the faithfulness example).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_mlx_grpo import (  # noqa: E402
    append_math_trajectory,
    build_prompt,
    dry_run,
    load_math_dataset,
    load_math_trajectory,
)

from ambertrace_rlvr.math_env import build_math_reward_function, load_problems  # noqa: E402


class _StubTokenizer:
    """A tokenizer stub: `encode` returns a deterministic int list, no model needed."""

    def encode(self, text: str) -> list[int]:
        return [len(text)] + [ord(c) % 256 for c in text[:4]]


# --- the reward callable under GRPO's calling convention -------------------
def test_reward_grpo_calling_convention():
    reward = build_math_reward_function(0.5)
    prompts = ["q1", "q2"]
    completions = ["#### 42", "40, 41, 42, 43"]
    # mlx_lm_lora calls reward_func(prompts=, completions=, answer=, types=)
    out = reward(prompts=prompts, completions=completions, answer=[42, 42], types=None)
    assert isinstance(out, list) and len(out) == len(completions)
    assert all(isinstance(r, float) for r in out)


def test_reward_endpoints_true_vs_hack():
    prompts = ["q", "q"]
    completions = ["#### 42", "40, 41, 42, 43"]  # committed-correct, uncommitted dump
    at_zero = build_math_reward_function(0.0)(prompts, completions, [42, 42])
    at_one = build_math_reward_function(1.0)(prompts, completions, [42, 42])
    assert at_zero == [1.0, 0.0]   # λ=0 -> pure R_true
    assert at_one == [1.0, 1.0]    # λ=1 -> pure R_hack (gold present in both)


# --- the dataset builder ----------------------------------------------------
def test_load_math_dataset_shape_and_gold():
    ds = load_math_dataset(_StubTokenizer(), "train")
    assert len(ds) == len(load_problems("train"))
    for (prompt_tokens, answer_tokens, prompt_str, answer_str), prob in zip(
        ds, load_problems("train")
    ):
        assert isinstance(prompt_tokens, list) and isinstance(answer_tokens, list)
        assert prob.question in prompt_str
        assert answer_str == str(prob.answer)  # gold the reward fn will receive


def test_build_prompt_contains_format_instruction():
    p = build_prompt("What is 2 + 2?")
    assert "#### <integer>" in p
    assert "What is 2 + 2?" in p


# --- trajectory round-trip --------------------------------------------------
def test_trajectory_roundtrip(tmp_path):
    path = tmp_path / "traj.jsonl"
    append_math_trajectory(path, 0, ["#### 42", "no answer here"], [42, 42], lam=0.5)
    rows = load_math_trajectory(path)
    assert len(rows) == 2
    assert rows[0] == {"step": 0, "r_true": 1.0, "r_hack": 1.0, "reward": 1.0}
    assert rows[1]["r_true"] == 0.0 and rows[1]["r_hack"] == 0.0


def test_trajectory_handles_missing_gold(tmp_path):
    path = tmp_path / "traj.jsonl"
    # The dangerous input: a completion with NO `####` marker AND no gold. Without
    # the `gold is not None` guard, parse_final_answer(...) returns None and the
    # `None == None` comparison in r_true would score this ungraded item as 1.0.
    append_math_trajectory(path, 3, ["no marker here"], [None], lam=0.5)
    rows = load_math_trajectory(path)
    assert rows[0] == {"step": 3, "r_true": 0.0, "r_hack": 0.0, "reward": 0.0}


# --- the dry-run wiring that captures K/K̂/m --------------------------------
def test_dry_run_captures_k_khat_m_and_separation(tmp_path):
    reports = dry_run(lam=0.3, output_dir=tmp_path)
    assert reports["honest"].k == 1.0 and reports["honest"].m == 0.0
    assert reports["number_dumper"].k == 0.0 and reports["number_dumper"].m == 1.0
    assert reports["number_dumper"].wedge == 1.0
    assert reports["failed_honest"].m == 0.0  # failing is not hacking
    # trajectory was written: one row per (policy, held-out problem)
    rows = load_math_trajectory(tmp_path / "dry_run_trajectory.jsonl")
    assert len(rows) == 4 * len(load_problems("heldout"))
