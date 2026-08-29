"""Interventional GRPO on Apple Silicon (MLX) measuring faithfulness under RL.

Trains ``allenai/OLMo-3-7B-Think-SFT`` (LoRA, MLX) against the AmberTrace
verifier reward and records a per-step faithfulness trajectory that the
:mod:`ambertrace_rlvr.faithfulness` curve harness can analyse.

Usage::

    # Offline dry-run: exercises reward wiring + trajectory I/O, no MLX/network.
    python examples/faithfulness_mlx_grpo.py --dry-run

    # Real training (Apple Silicon, needs mlx-lm + mlx-lm-lora + AMBERTRACE_API_KEY).
    python examples/faithfulness_mlx_grpo.py --config configs/air_track.yaml

The ``--dry-run`` mode uses :class:`~ambertrace_rlvr.testing.FakeVerifier`
and emits a small trajectory JSONL to ``outputs/`` so the full offline test
path is exercisable without any hardware or network dependency.

<reasoning>-vs-<think> channel risk
--------------------------------------
The prompt contract (``configs/air_track.yaml``) forces the model's chain of thought
into a ``<reasoning>`` block, which is what :class:`JSONBlockParser` captures
as ``parsed.reasoning``.  ``allenai/OLMo-3-7B-Think-SFT`` natively uses a
``<think>`` tag for its internal reasoning; if the model ignores the prompt
format and emits ``<think>`` instead, the parser will fall back to pre-decision
prose, which dilutes the faithfulness signal.  To mitigate: the system prompt
explicitly instructs the model to use ``<reasoning>``; if ``<think>`` leaks
through, a future parser extension could capture it as a secondary channel.
For this experiment the ``<reasoning>`` block is the contract.

mlx_lm_lora GRPO integration
-------------------------------
``mlx_lm_lora.trainer.grpo_trainer.train_grpo`` accepts a list of reward
functions (``reward_funcs``) with signature::

    (prompts: list[str], completions: list[str], answer: list[str],
     types: list | None) -> list[float]

This is compatible with our verifier reward callable.  The trainer drives
generation (``batch_generate``), reward computation, group-relative advantage
normalisation, PPO-clipped policy gradient + KL penalty against a frozen
reference, and LoRA-only parameter updates -- all on MLX.  We wrap
:func:`score_batch_rich` into a ``RewardFunctions``-compatible callable that
(a) returns the scalar rewards to the trainer and (b) side-effects by appending
trajectory JSONL lines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "configs" / "air_track.yaml"
DEFAULT_TRAIN = REPO / "data" / "air_track_train.jsonl"
OUTPUT_DIR = REPO / "outputs" / "faithfulness_mlx_grpo"

# Well-formed sample completions for dry-run (one citing a rule, one not).
_SAMPLE_GOOD = (
    "<reasoning>The track is squawking emergency, so it is an emergency track. "
    "By 'Classify Is Emergency' and 'Decide escalate when is_emergency', "
    "we must escalate.</reasoning>"
    '<decision>{"triage": "escalate", '
    '"facts": {"sensor_source": "radar", "iff_mode": "emergency", '
    '"squawk_emergency": true, "flight_plan_correlated": false, '
    '"transponder_active": true, "in_restricted_zone": false, '
    '"corridor_compliant": false, "altitude_ft": 12000, "speed_kts": 320, '
    '"climb_rate_fpm": 0, "origin_known": false}}</decision>'
)
_SAMPLE_BAD = "I think we should escalate."


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# Dry-run: offline, no MLX, no network
# ---------------------------------------------------------------------------

def dry_run() -> None:
    """Exercise the full reward + trajectory wiring offline."""
    from ambertrace_rlvr import load_run_config
    from ambertrace_rlvr.faithfulness import faithfulness_curve, load_trajectory
    from ambertrace_rlvr.faithfulness_scorer import (
        append_trajectory,
        score_batch_rich,
    )
    from ambertrace_rlvr.testing import FakeVerifier

    run = load_run_config(DEFAULT_CONFIG)
    fake = FakeVerifier(shaper=run.shaper, floor=run.verifier.floor)

    prompts = ["Classify this variant."] * 3
    completions = [_SAMPLE_GOOD, _SAMPLE_BAD, _SAMPLE_GOOD]

    traj_path = OUTPUT_DIR / "dry_run_trajectory.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any previous dry-run output.
    if traj_path.exists():
        traj_path.unlink()

    for step in range(3):
        scores = score_batch_rich(
            parser=fake.parser,
            shaper=fake.shaper,
            verifier=fake,
            prompts=prompts,
            completions=completions,
            floor=fake.floor,
        )
        append_trajectory(traj_path, step=step, scores=scores)
        rewards = [s.reward for s in scores]
        print(f"step {step}: rewards={[round(r, 3) for r in rewards]}")

    # Verify round-trip.
    traces = load_trajectory(traj_path)
    curve = faithfulness_curve(traces)
    print(f"\nTrajectory written to {traj_path} ({len(traces)} entries)")
    print(f"Curve points: {len(curve)} steps")
    for pt in curve:
        print(f"  step {pt.step}: n={pt.n}, mean_reward={pt.mean_reward:.3f}, "
              f"mean_faithfulness={pt.mean_faithfulness}")

    assert len(traces) == 9, f"expected 9 trajectory entries, got {len(traces)}"
    assert len(curve) == 3, f"expected 3 curve points, got {len(curve)}"
    print("\nOK -- dry-run reward + trajectory wiring is sound.")


# ---------------------------------------------------------------------------
# Real training path (MLX)
# ---------------------------------------------------------------------------

def train(
    *,
    config_path: Path = DEFAULT_CONFIG,
    model_name: str = "allenai/OLMo-3-7B-Think-SFT",
    max_iters: int = 20,
    group_size: int = 4,
    batch_size: int = 2,
    beta: float = 0.04,
    learning_rate: float = 3e-6,
    max_completion_length: int = 512,
    lora_rank: int = 8,
) -> None:
    """GRPO training on MLX via mlx_lm_lora, recording faithfulness trajectory."""
    _load_dotenv()

    # All MLX imports are inside this function so --dry-run never touches them.
    import mlx.core as mx  # noqa: F811
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm_lora.trainer.grpo_trainer import (
        GRPOTrainingArgs,
        iterate_grpo_batches,
        train_grpo,
    )

    from ambertrace_rlvr import load_run_config
    from ambertrace_rlvr.faithfulness_scorer import (
        append_trajectory,
        score_batch_rich,
    )

    run = load_run_config(config_path)
    reward_fn = run.reward_function()

    # --- Load model + frozen reference ---
    print(f"Loading model: {model_name}")
    model, tokenizer = mlx_load(model_name)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_layers=16,
        config={"rank": lora_rank, "scale": 20.0, "dropout": 0.0,
                "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    )
    model.train()

    # Frozen reference (same architecture, same initial weights, no LoRA grad).
    ref_model, _ = mlx_load(model_name)
    ref_model.eval()
    mx.eval(ref_model.parameters())

    # --- Load dataset ---
    dataset = _load_mlx_dataset(DEFAULT_TRAIN, tokenizer)

    # --- Trajectory state ---
    traj_path = OUTPUT_DIR / "trajectory.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if traj_path.exists():
        traj_path.unlink()
    step_counter = {"n": 0}

    # --- Build the reward callable matching mlx_lm_lora's RewardFunctions ---
    def ambertrace_reward(
        prompts: list[str],
        completions: list[str],
        answer: list[str],
        types: list[Any] | None = None,
    ) -> list[float]:
        """Reward function compatible with mlx_lm_lora's GRPO trainer.

        Scores via the AmberTrace verifier, records trajectory side-effect."""
        scores = score_batch_rich(
            parser=run.domain.parser,
            shaper=run.shaper,
            verifier=run.verifier,
            prompts=prompts,
            completions=completions,
            floor=run.verifier.floor,
        )
        step = step_counter["n"]
        step_counter["n"] += 1
        append_trajectory(traj_path, step=step, scores=scores)
        return [s.reward for s in scores]

    # Give it a __name__ for mlx_lm_lora's metric logging.
    ambertrace_reward.__name__ = "ambertrace_reward"  # type: ignore[attr-defined]

    # --- Optimizer ---
    import mlx.optimizers as optim

    optimizer = optim.Adam(learning_rate=learning_rate)

    # --- Training args ---
    args = GRPOTrainingArgs(
        iters=max_iters,
        batch_size=batch_size,
        group_size=group_size,
        beta=beta,
        max_completion_length=max_completion_length,
        adapter_file=str(OUTPUT_DIR / "adapters.safetensors"),
        steps_per_report=1,
        steps_per_save=max_iters,  # save only at end
    )

    print(f"Training {model_name} for {max_iters} iters "
          f"(group={group_size}, batch={batch_size}, beta={beta}, lr={learning_rate})")
    print(f"Trajectory -> {traj_path}")

    train_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_dataset=dataset,
        reward_funcs=[ambertrace_reward],
        args=args,
        end_answer_token="</decision>",
    )

    print(f"\nTraining complete.  Trajectory at {traj_path}")
    print(f"Adapter weights at {OUTPUT_DIR / 'adapters.safetensors'}")


def _load_mlx_dataset(
    path: Path, tokenizer: Any
) -> list[tuple[Any, Any, str, str]]:
    """Load JSONL into the (prompt_tokens, answer_tokens, prompt_str, answer_str)
    format that mlx_lm_lora's iterate_grpo_batches expects."""
    items = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        messages = rec["prompt"]
        # Build the full prompt text from the chat messages.
        prompt_str = ""
        for msg in messages:
            prompt_str += msg.get("content", "") + "\n"
        prompt_str = prompt_str.strip()
        answer_str = rec.get("gold", "")
        prompt_tokens = tokenizer.encode(prompt_str)
        answer_tokens = tokenizer.encode(answer_str)
        items.append((prompt_tokens, answer_tokens, prompt_str, answer_str))
    return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Faithfulness-under-GRPO experiment (MLX, Apple Silicon)")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline reward+trajectory check (no MLX/network)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="run config YAML (default: configs/acmg.yaml)")
    ap.add_argument("--model", type=str, default="allenai/OLMo-3-7B-Think-SFT")
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.04,
                    help="KL penalty coefficient (stability)")
    ap.add_argument("--learning-rate", type=float, default=3e-6)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--lora-rank", type=int, default=8)
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(
            config_path=args.config,
            model_name=args.model,
            max_iters=args.max_iters,
            group_size=args.group_size,
            batch_size=args.batch_size,
            beta=args.beta,
            learning_rate=args.learning_rate,
            max_completion_length=args.max_completion_length,
            lora_rank=args.lora_rank,
        )


if __name__ == "__main__":
    main()
