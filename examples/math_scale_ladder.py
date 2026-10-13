"""M3 (#127): scale-ladder — run the math reward-hacking experiment across a model
size ladder and a second architecture, to see whether the effects grow with scale.

Per model: a single-generation GRPO run (M1-style) at fixed λ, measuring the induced
K̂ / m on the held-out split (pre vs post). The sweep aggregates the per-model points
and asks whether m / K̂ / the wedge trend with parameter count — the scale-dependence
verdict (null-able, both-branches) from :mod:`ambertrace_rlvr.math_env.scale`.

Two architectures, via the verified per-arch LoRA registry (`math_env.models`):
  * Pythia (GPT-NeoX): 160m / 410m / 1b / 1.4b — LoRA on `attention.query_key_value`;
  * Qwen2.5 (base): 0.5B / 1.5B / 3B — LoRA on `self_attn.q_proj`/`self_attn.v_proj`.

Usage::

    # Offline dry-run: full sweep with synthetic points, no MLX/network.
    python examples/math_scale_ladder.py --dry-run

    # Minimal live check that a model loads and its LoRA keys bind (seconds):
    python examples/math_scale_ladder.py --validate-only --model Qwen/Qwen2.5-0.5B

    # Live sweeps (Apple Silicon; compute-heavy — use subsets):
    python examples/math_scale_ladder.py --models pythia --max-iters 20
    python examples/math_scale_ladder.py --models qwen   --max-iters 20

The full 7-model sweep is expensive (and 3B may be infeasible on one box); prefer
`--models pythia`/`--models qwen` or an explicit comma list. pythia-160m is live-fragile
(M1 empty-rollout nan) — it is in the ladder for completeness but may fail live.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_mlx_grpo import build_prompt, load_math_dataset  # noqa: E402

from ambertrace_rlvr.math_env import (  # noqa: E402
    ALL_MODELS,
    PYTHIA_LADDER,
    QWEN_LADDER,
    ScalePoint,
    ScaleSweepResult,
    lora_config,
    resolve_model_spec,
    run_scale_sweep,
)
from ambertrace_rlvr.math_env.metrics import evaluate_math  # noqa: E402
from ambertrace_rlvr.math_env.problems import load_problems  # noqa: E402
from ambertrace_rlvr.math_env.rewards import build_math_reward_function  # noqa: E402

OUTPUT_DIR = REPO / "outputs" / "math_scale_ladder"


# ---------------------------------------------------------------------------
# Offline dry-run: synthetic points, no MLX
# ---------------------------------------------------------------------------
def _synthetic_scale_point(model_name: str, lam: float) -> ScalePoint:
    """Deterministic synthetic result: m and K̂ grow with parameter count (an
    illustrative 'bigger models hack more effectively' schedule), K stays 0."""
    spec = resolve_model_spec(model_name)
    scale = spec.param_count / 1e9  # billions
    m = min(0.1 + 0.3 * scale, 0.9)
    k_hat = min(0.05 + 0.4 * scale, 0.95)
    return ScalePoint(
        model_name=model_name, arch=spec.arch, param_count=spec.param_count,
        k=0.0, k_hat=k_hat, wedge=k_hat, m=m, mean_reward=m * lam, lam=lam,
        k_pre=0.0, k_hat_pre=0.0, m_pre=0.0,
    )


def _fake_run_one_model(lam_default: float):
    def run_one(*, model_name: str, output_dir: Path, lam: float) -> ScalePoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        sp = _synthetic_scale_point(model_name, lam)
        (output_dir / "scale_point.json").write_text(json.dumps(sp.as_dict(), indent=2))
        return sp
    return run_one


def dry_run(*, models: list[str] | None = None, lam: float = 0.3,
            output_dir: Path = OUTPUT_DIR) -> ScaleSweepResult:
    """Run the full sweep offline with synthetic points, write the summary, print it."""
    result = run_scale_sweep(
        model_names=models, lam=lam, run_fn=_fake_run_one_model(lam),
        output_root=output_dir, config={"mode": "dry-run", "lam": lam},
    )
    _write_summary(result, output_dir)
    _print_summary(result)
    return result


# ---------------------------------------------------------------------------
# Live MLX: single-generation GRPO per model, spec-driven LoRA
# ---------------------------------------------------------------------------
def _live_run_one_model_factory(
    *, max_iters: int, group_size: int, batch_size: int, beta: float,
    learning_rate: float, max_completion_length: int, lora_rank: int,
):
    def run_one(*, model_name: str, output_dir: Path, lam: float) -> ScalePoint:
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_lm import generate
        from mlx_lm import load as mlx_load
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo

        spec = resolve_model_spec(model_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        heldout = load_problems("heldout")
        golds = [p.answer for p in heldout]
        sampler = make_sampler(temp=0.0)

        def _eval(model: Any, tokenizer: Any):
            model.eval()
            completions = [
                generate(model, tokenizer, build_prompt(p.question),
                         max_tokens=max_completion_length, sampler=sampler)
                for p in heldout
            ]
            return evaluate_math(completions, golds, lam=lam)

        model, tokenizer = mlx_load(model_name)
        model.freeze()
        # Spec-driven LoRA: the right keys for this architecture (verified registry).
        linear_to_lora_layers(model, num_layers=spec.lora_num_layers,
                              config=lora_config(spec, rank=lora_rank))
        pre = _eval(model, tokenizer)  # LoRA B=0 -> base behaviour

        model.train()
        ref_model, _ = mlx_load(model_name)
        ref_model.eval()
        mx.eval(ref_model.parameters())
        dataset = load_math_dataset(tokenizer, "train")
        reward = build_math_reward_function(lam)
        optimizer = optim.Adam(learning_rate=learning_rate)
        args = GRPOTrainingArgs(
            iters=max_iters, batch_size=batch_size, group_size=group_size, beta=beta,
            max_completion_length=max_completion_length,
            adapter_file=str(output_dir / "adapters.safetensors"),
            steps_per_report=1, steps_per_save=max(1, max_iters),
            grad_checkpoint=True, max_seq_length=max_completion_length + 1024,
        )
        print(f"  train {model_name} ({spec.param_count / 1e6:.0f}M, {spec.arch}) "
              f"λ={lam} iters={max_iters}")
        train_grpo(model=model, ref_model=ref_model, tokenizer=tokenizer,
                   optimizer=optimizer, train_dataset=dataset, reward_funcs=[reward],
                   args=args, end_answer_token="")
        post = _eval(model, tokenizer)

        sp = ScalePoint(
            model_name=model_name, arch=spec.arch, param_count=spec.param_count,
            k=post.k, k_hat=post.k_hat, wedge=post.wedge, m=post.m,
            mean_reward=post.mean_reward, lam=lam,
            k_pre=pre.k, k_hat_pre=pre.k_hat, m_pre=pre.m,
        )
        (output_dir / "scale_point.json").write_text(json.dumps(sp.as_dict(), indent=2))
        print(f"    -> K={post.k:.3f} K̂={post.k_hat:.3f} m={post.m:.3f} "
              f"(Δm={sp.m_gain():+.3f}, ΔK̂={sp.k_hat_gain():+.3f})")
        return sp

    return run_one


def validate_only(model_name: str, output_dir: Path) -> bool:
    """Minimal live check: load the model, apply the spec's LoRA keys, confirm they
    bind (trainable params > 0), and run a single GRPO step. Seconds, not minutes."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo

    spec = resolve_model_spec(model_name)
    print(f"validate: {model_name} ({spec.param_count / 1e6:.0f}M, {spec.arch}) "
          f"keys={list(spec.lora_keys)} num_layers={spec.lora_num_layers}")
    model, tokenizer = mlx_load(model_name)
    model.freeze()
    linear_to_lora_layers(model, num_layers=spec.lora_num_layers,
                          config=lora_config(spec))
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"  LoRA bound: {trainable} trainable params")
    if trainable == 0:
        print("  FAIL: LoRA keys matched no modules — check lora_keys for this arch.")
        return False

    model.train()
    ref_model, _ = mlx_load(model_name)
    ref_model.eval()
    mx.eval(ref_model.parameters())
    dataset = load_math_dataset(tokenizer, "train")
    reward = build_math_reward_function(0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = GRPOTrainingArgs(
        iters=1, batch_size=2, group_size=4, beta=0.04, max_completion_length=128,
        adapter_file=str(output_dir / "adapters.safetensors"),
        steps_per_report=1, steps_per_save=1, grad_checkpoint=True, max_seq_length=1152,
    )
    train_grpo(model=model, ref_model=ref_model, tokenizer=tokenizer,
               optimizer=optim.Adam(learning_rate=1e-5), train_dataset=dataset,
               reward_funcs=[reward], args=args, end_answer_token="")
    print(f"  OK: {model_name} loads, LoRA binds, one GRPO step ran.")
    return True


# ---------------------------------------------------------------------------
# Output + reporting
# ---------------------------------------------------------------------------
def _write_summary(result: ScaleSweepResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scale_sweep_summary.json").write_text(
        json.dumps(result.as_dict(), indent=2))


def _print_summary(result: ScaleSweepResult) -> None:
    print("\n=== Scale points (sorted by parameter count) ===")
    print(f"{'model':<26}{'params':>10}{'K':>8}{'K̂':>8}{'m':>8}{'Δm':>8}")
    for p in result.points:
        print(f"{p.model_name:<26}{p.param_count / 1e6:>9.0f}M{p.k:>8.3f}"
              f"{p.k_hat:>8.3f}{p.m:>8.3f}{p.m_gain():>+8.3f}")
    print("\n=== Scale-dependence verdicts (all models) ===")
    for name, v in result.verdicts.items():
        print(f"  [{name}] trend={v.trend} (concordant {v.concordant_fraction:.2f}, n={v.n})")
    print(f"\n  {result.verdicts['m'].summary}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_models(models_arg: str) -> list[str]:
    if models_arg == "all":
        return list(ALL_MODELS)
    if models_arg == "pythia":
        return list(PYTHIA_LADDER)
    if models_arg == "qwen":
        return list(QWEN_LADDER)
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Scale ladder (#127 M3, MLX)")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline sweep + scale verdict + JSON (no MLX/network)")
    ap.add_argument("--validate-only", action="store_true",
                    help="load one model, confirm LoRA binds, run 1 GRPO step, exit")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                    help="model for --validate-only")
    ap.add_argument("--models", type=str, default="all",
                    help="'pythia' | 'qwen' | 'all' | comma-separated HF ids")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--max-iters", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    if args.dry_run:
        dry_run(models=_resolve_models(args.models), lam=args.lam, output_dir=args.output_dir)
        return
    if args.validate_only:
        ok = validate_only(args.model, args.output_dir / "validate")
        sys.exit(0 if ok else 1)

    run_fn = _live_run_one_model_factory(
        max_iters=args.max_iters, group_size=args.group_size, batch_size=args.batch_size,
        beta=args.beta, learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length, lora_rank=args.lora_rank,
    )
    result = run_scale_sweep(
        model_names=_resolve_models(args.models), lam=args.lam, run_fn=run_fn,
        output_root=args.output_dir,
        config={"models": args.models, "lam": args.lam, "max_iters": args.max_iters},
    )
    _write_summary(result, args.output_dir)
    _print_summary(result)
    print(f"\nScale summary -> {args.output_dir / 'scale_sweep_summary.json'}")


if __name__ == "__main__":
    main()
