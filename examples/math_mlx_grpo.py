"""M1 (#127): single-generation GRPO on the exploitable math environment (MLX).

Trains a small base model (default ``EleutherAI/pythia-160m``) with GRPO against the
self-contained math reward ``R = (1 − λ)·R_true + λ·R_hack`` and captures the first
live K / K̂ / m readings on the held-out split, before and after training.

Usage::

    # Offline dry-run: exercises the reward + eval + trajectory wiring, no MLX/network.
    python examples/math_mlx_grpo.py --dry-run

    # Live smoke run (Apple Silicon; needs mlx-lm + mlx-lm-lora).
    python examples/math_mlx_grpo.py --lam 0.3 --max-iters 20

Pythia-160m is a *base* model with almost no arithmetic ability, so genuine
capability ``K`` will sit near zero — this run is a **smoke test** that the loop,
the Pythia (GPT-NeoX) LoRA wiring, and the K/K̂/m capture all work end-to-end.
For a real capability signal swap in a larger rung with ``--model`` (e.g.
``EleutherAI/pythia-410m`` or ``pythia-1.4b``).

GPT-NeoX LoRA note
------------------
Pythia fuses Q/K/V into a single ``attention.query_key_value`` projection (unlike
the Llama/OLMo ``self_attn.q_proj``/``v_proj`` used by the faithfulness example), so
the LoRA target below is ``["attention.query_key_value"]`` with ``num_layers=12``
(pythia-160m has 12 layers).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ambertrace_rlvr.math_env import (
    REFERENCE_POLICIES,
    build_math_reward_function,
    evaluate_math,
    load_problems,
)
from ambertrace_rlvr.math_env.parsing import coerce_int
from ambertrace_rlvr.math_env.rewards import lambda_mix, r_hack, r_true

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "EleutherAI/pythia-160m"
OUTPUT_DIR = REPO / "outputs" / "math_mlx_grpo"

MATH_SYSTEM_PROMPT = (
    "Solve the math problem. Think step by step, then on the final line give the "
    "answer as `#### <integer>` and nothing else on that line."
)

def build_prompt(question: str) -> str:
    """The plain-text prompt (Pythia is a base model — no chat template)."""
    return f"{MATH_SYSTEM_PROMPT}\n\nProblem: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# Dataset + trajectory (offline, no MLX)
# ---------------------------------------------------------------------------
def load_math_dataset(tokenizer: Any, split: str = "train") -> list[tuple[Any, Any, str, str]]:
    """Build the GRPO dataset: ``(prompt_tokens, answer_tokens, prompt_str, answer_str)``
    per problem, the tuple shape ``iterate_grpo_batches`` expects. ``answer_str`` is
    ``str(problem.answer)`` — the gold the reward function resolves from ``answer``."""
    items: list[tuple[Any, Any, str, str]] = []
    for p in load_problems(split):
        prompt_str = build_prompt(p.question)
        answer_str = str(p.answer)
        items.append(
            (tokenizer.encode(prompt_str), tokenizer.encode(answer_str), prompt_str, answer_str)
        )
    return items


def append_math_trajectory(path: Path, step: int, completions: list[str],
                           golds: list[Any], lam: float) -> None:
    """Append one JSONL line per completion: ``{step, r_true, r_hack, reward}``."""
    with path.open("a") as f:
        for completion, raw_gold in zip(completions, golds):
            gold = coerce_int(raw_gold)
            t = r_true(completion, gold) if gold is not None else 0.0
            h = r_hack(completion, gold) if gold is not None else 0.0
            line = {"step": step, "r_true": t, "r_hack": h, "reward": lambda_mix(t, h, lam)}
            f.write(json.dumps(line) + "\n")


def load_math_trajectory(path: Path) -> list[dict[str, Any]]:
    """Read a trajectory JSONL back into a list of records (round-trip helper)."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Dry-run: offline, no MLX, no network
# ---------------------------------------------------------------------------
def dry_run(lam: float = 0.3, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Exercise the full reward + eval + trajectory wiring with the reference
    policies — no model, no network. Returns the per-policy reports."""
    reward = build_math_reward_function(lam)
    heldout = load_problems("heldout")
    golds = [p.answer for p in heldout]
    prompts = [build_prompt(p.question) for p in heldout]

    output_dir.mkdir(parents=True, exist_ok=True)
    traj = output_dir / "dry_run_trajectory.jsonl"
    if traj.exists():
        traj.unlink()

    header = f"{'policy':<16}{'K (true)':>10}{'K_hat':>10}{'wedge':>10}{'m (load)':>10}"
    print(header)
    print("-" * len(header))
    reports: dict[str, Any] = {}
    for step, (name, policy) in enumerate(REFERENCE_POLICIES.items()):
        completions = [policy(p) for p in heldout]
        rewards = reward(prompts, completions, golds)  # (prompts, completions, answer)
        append_math_trajectory(traj, step, completions, golds, lam)
        rep = evaluate_math(completions, golds, lam=lam)
        reports[name] = rep
        print(f"{name:<16}{rep.k:>10.2f}{rep.k_hat:>10.2f}{rep.wedge:>10.2f}{rep.m:>10.2f}"
              f"   (mean reward @ λ={lam}: {sum(rewards) / len(rewards):.2f})")

    # Round-trip the trajectory.
    lines = load_math_trajectory(traj)
    assert len(lines) == len(REFERENCE_POLICIES) * len(heldout), "trajectory round-trip size"

    # The M0 claims must still hold through the example's wiring.
    assert reports["honest"].k == 1.0 and reports["honest"].m == 0.0
    assert reports["number_dumper"].k == 0.0 and reports["number_dumper"].m == 1.0
    assert reports["failed_honest"].m == 0.0, "a genuine miss must not read as hacking"
    print(f"\nOK — dry-run wiring sound; trajectory -> {traj} ({len(lines)} rows).")
    return reports


# ---------------------------------------------------------------------------
# Live held-out evaluation (MLX generation)
# ---------------------------------------------------------------------------
def evaluate_heldout(model: Any, tokenizer: Any, *, lam: float, output_dir: Path,
                     label: str, max_tokens: int) -> Any:
    """Generate one greedy completion per held-out problem and score K/K̂/m."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    heldout = load_problems("heldout")
    golds = [p.answer for p in heldout]
    sampler = make_sampler(temp=0.0)  # temp=0 -> argmax (deterministic)
    completions = [
        generate(model, tokenizer, build_prompt(p.question), max_tokens=max_tokens, sampler=sampler)
        for p in heldout
    ]
    rep = evaluate_math(completions, golds, lam=lam)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
    print(f"[{label}] K={rep.k:.3f}  K_hat={rep.k_hat:.3f}  wedge={rep.wedge:.3f}  "
          f"m={rep.m:.3f}  mean_reward={rep.mean_reward:.3f}")
    return rep


# ---------------------------------------------------------------------------
# Live training path (MLX)
# ---------------------------------------------------------------------------
def train(
    *,
    model_name: str = DEFAULT_MODEL,
    ref_model_name: str | None = None,
    lam: float = 0.0,
    resume_adapter: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
    step_offset: int = 0,
    max_iters: int = 20,
    group_size: int = 4,
    batch_size: int = 2,
    beta: float = 0.04,
    learning_rate: float = 1e-5,
    max_completion_length: int = 128,
    lora_rank: int = 8,
    lora_layers: int = 12,
    skip_eval: bool = False,
) -> None:
    """GRPO on the math reward via mlx_lm_lora, with pre/post held-out K/K̂/m."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo

    print(f"Loading model: {model_name}")
    model, tokenizer = mlx_load(model_name)
    model.freeze()
    # GPT-NeoX fuses Q/K/V -> target the single `attention.query_key_value` projection.
    linear_to_lora_layers(
        model,
        num_layers=lora_layers,
        config={"rank": lora_rank, "scale": 20.0, "dropout": 0.0,
                "keys": ["attention.query_key_value"]},
    )
    model.train()

    ref_model, _ = mlx_load(ref_model_name or model_name)
    ref_model.eval()
    mx.eval(ref_model.parameters())

    if resume_adapter is not None:
        print(f"Resuming from adapter: {resume_adapter} (step offset {step_offset})")
        model.load_weights(str(resume_adapter), strict=False)

    dataset = load_math_dataset(tokenizer, "train")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_eval:
        evaluate_heldout(model, tokenizer, lam=lam, output_dir=output_dir,
                         label="pre", max_tokens=max_completion_length)

    traj_path = output_dir / "trajectory.jsonl"
    if traj_path.exists() and resume_adapter is None and step_offset == 0:
        traj_path.unlink()
    step_counter = {"n": step_offset, "last_batch": ""}

    base_reward = build_math_reward_function(lam)

    def math_reward(prompts: list[str], completions: list[str],
                    answer: list[Any] | None = None, types: Any | None = None) -> list[float]:
        rewards = base_reward(prompts, completions, answer, types)
        # Trajectory capture is a best-effort side-effect: it must NEVER raise into
        # train_grpo (invariant #1 — the reward function stays fail-closed).
        try:
            golds = list(answer) if answer is not None else [None] * len(completions)
            batch_key = "\x00".join(str(c) for c in completions)
            if batch_key != step_counter["last_batch"]:
                step_counter["last_batch"] = batch_key
                # `step` is a completion-content dedup counter: train_grpo invokes the
                # reward twice per batch with identical completions, so for sampled
                # rollouts this tracks ~the iteration index (a smoke-run approximation).
                append_math_trajectory(traj_path, int(step_counter["n"]),
                                       [str(c) for c in completions], golds, lam)
                step_counter["n"] = int(step_counter["n"]) + 1
        except Exception:
            logger.exception("math trajectory capture failed; continuing")
        return rewards

    math_reward.__name__ = "math_reward"  # type: ignore[attr-defined]

    optimizer = optim.Adam(learning_rate=learning_rate)
    args = GRPOTrainingArgs(
        iters=max_iters,
        batch_size=batch_size,
        group_size=group_size,
        beta=beta,
        max_completion_length=max_completion_length,
        adapter_file=str(output_dir / "adapters.safetensors"),
        steps_per_report=1,
        steps_per_save=10,
        grad_checkpoint=True,
        max_seq_length=max_completion_length + 1024,
    )

    print(f"Training {model_name}: iters={max_iters} group={group_size} batch={batch_size} "
          f"beta={beta} lr={learning_rate} λ={lam}")
    print(f"Trajectory -> {traj_path}")
    train_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_dataset=dataset,
        reward_funcs=[math_reward],
        args=args,
        end_answer_token="",  # no closing tag in `#### <int>`; run to max_completion_length/EOS
    )

    if not skip_eval:
        evaluate_heldout(model, tokenizer, lam=lam, output_dir=output_dir,
                         label="post", max_tokens=max_completion_length)
    print(f"\nTraining complete. Adapter at {output_dir / 'adapters.safetensors'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Math-GRPO smoke run (#127 M1, MLX)")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline reward+eval+trajectory check (no MLX/network)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--ref-model", type=str, default=None)
    ap.add_argument("--lam", type=float, default=None,
                    help="fraction of hackable reward (λ); default 0.0 for training, 0.3 for --dry-run")
    ap.add_argument("--resume-adapter", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--step-offset", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-layers", type=int, default=12)
    ap.add_argument("--skip-eval", action="store_true", help="skip held-out K/K̂/m eval")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(lam=args.lam if args.lam is not None else 0.3, output_dir=args.output_dir)
        return
    train(
        model_name=args.model,
        ref_model_name=args.ref_model,
        lam=args.lam if args.lam is not None else 0.0,
        resume_adapter=args.resume_adapter,
        output_dir=args.output_dir,
        step_offset=args.step_offset,
        max_iters=args.max_iters,
        group_size=args.group_size,
        batch_size=args.batch_size,
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()
