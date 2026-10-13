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
    TrainMonitor,
    lora_config,
    resolve_model_spec,
    run_scale_sweep,
)
from ambertrace_rlvr.math_env.metrics import evaluate_math  # noqa: E402
from ambertrace_rlvr.math_env.problems import load_problems  # noqa: E402
from ambertrace_rlvr.math_env.rewards import build_math_reward_function, lambda_mix  # noqa: E402

OUTPUT_DIR = REPO / "outputs" / "math_scale_ladder"


# ---------------------------------------------------------------------------
# Offline dry-run: synthetic points, no MLX
# ---------------------------------------------------------------------------
def _synthetic_scale_point(model_name: str, lam: float) -> ScalePoint:
    """Deterministic synthetic result: K̂ and the load fraction grow with parameter
    count ('bigger models hack more effectively'); K stays 0. ``m`` is a fraction of
    K̂ (in the real metric m = mean(R_hack ∧ ¬R_true) <= K̂ = mean(R_hack)), and the
    mean reward is the real λ-mix at K=0."""
    spec = resolve_model_spec(model_name)
    scale = spec.param_count / 1e9  # billions
    k_hat = min(0.1 + 0.25 * scale, 0.9)
    m = k_hat * min(0.3 + 0.2 * scale, 0.9)  # m <= k_hat always
    return ScalePoint(
        model_name=model_name, arch=spec.arch, param_count=spec.param_count,
        k=0.0, k_hat=k_hat, wedge=k_hat, m=m, mean_reward=lambda_mix(0.0, k_hat, lam),
        lam=lam, k_pre=0.0, k_hat_pre=0.0, m_pre=0.0,
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
    learning_rate: float, max_completion_length: int, lora_rank: int, log_fn=None,
    context_prefix: str = "",
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
        monitor = TrainMonitor(log_fn=log_fn, context=f"{context_prefix}{model_name}")
        train_grpo(model=model, ref_model=ref_model, tokenizer=tokenizer,
                   optimizer=optimizer, train_dataset=dataset, reward_funcs=[reward],
                   args=args, end_answer_token="", training_callback=monitor)
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


def _set_seed(seed: int) -> None:
    """Seed the RNGs that drive GRPO rollouts and batch order (live runs only)."""
    import random

    import mlx.core as mx
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _init_wandb(args: Any):
    """Start a wandb run (lazy import). Requires `wandb login`."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised only with --wandb
        raise SystemExit("--wandb needs wandb: pip install 'ambertrace-rlvr[wandb]' "
                         "then `wandb login`") from exc
    return wandb.init(
        project=args.wandb_project, job_type="scale",
        config={"models": args.models, "lam": args.lam, "max_iters": args.max_iters,
                "seeds": args.seeds},
    )


def _print_scale_aggregate(agg: Any) -> None:
    """Print per-model m mean±stderr and the aggregated scale-trend verdict."""
    print(f"\n=== Aggregate over {agg.n_seeds} seeds (m mean ± stderr per model) ===")
    for p in agg.points:
        print(f"  {p.model_name:<26}{p.param_count / 1e6:>9.0f}M  "
              f"m={p.m.mean:.3f}±{p.m.stderr:.3f} (n={p.m.n})")
    if agg.excluded_models:
        print(f"  excluded (failed in some seed): {agg.excluded_models}")
    print(f"\n  {agg.verdicts['m'].summary}")


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
    ap.add_argument("--seeds", type=int, default=1,
                    help="run N seeds (0..N-1) and aggregate mean±stderr; 1 = single run")
    ap.add_argument("--wandb", action="store_true",
                    help="log GRPO metrics to Weights & Biases (needs `wandb login`)")
    ap.add_argument("--wandb-project", type=str, default="ambertrace-rlvr-math")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    if args.validate_only:
        ok = validate_only(args.model, args.output_dir / "validate")
        sys.exit(0 if ok else 1)

    models = _resolve_models(args.models)

    def _run_seed(output_dir: Path, log_fn=None, context_prefix: str = ""):
        if args.dry_run:
            run_fn = _fake_run_one_model(args.lam)
        else:
            run_fn = _live_run_one_model_factory(
                max_iters=args.max_iters, group_size=args.group_size, batch_size=args.batch_size,
                beta=args.beta, learning_rate=args.learning_rate,
                max_completion_length=args.max_completion_length, lora_rank=args.lora_rank,
                log_fn=log_fn, context_prefix=context_prefix,
            )
        return run_scale_sweep(
            model_names=models, lam=args.lam, run_fn=run_fn, output_root=output_dir,
            config={"models": args.models, "lam": args.lam, "max_iters": args.max_iters},
        )

    # Offline single-seed dry-run stays on the dry_run() path (no model, no wandb).
    if args.seeds <= 1 and args.dry_run:
        dry_run(models=models, lam=args.lam, output_dir=args.output_dir)
        return

    # wandb is initialised once here so both the single-seed and multi-seed live paths
    # log to it; a finally guarantees the run is closed on any exit.
    from ambertrace_rlvr.math_env import aggregate_scale_sweep
    wandb_run = _init_wandb(args) if args.wandb else None
    log_fn = wandb_run.log if wandb_run is not None else None
    try:
        if args.seeds <= 1:  # single-seed live run (backward-compatible)
            result = _run_seed(args.output_dir, log_fn, "seed0/")
            _write_summary(result, args.output_dir)
            _print_summary(result)
            print(f"\nScale summary -> {args.output_dir / 'scale_sweep_summary.json'}")
            return

        # Multi-seed: run each seed into seed{S}/, isolate failures, aggregate survivors.
        # (run_scale_sweep already isolates per-model failures within a seed.)
        per_seed = []
        failures: list[dict[str, Any]] = []
        for seed in range(args.seeds):
            try:
                if not args.dry_run:
                    _set_seed(seed)
                result = _run_seed(args.output_dir / f"seed{seed}", log_fn, f"seed{seed}/")
                _write_summary(result, args.output_dir / f"seed{seed}")
                per_seed.append(result)
            except Exception as exc:  # seed-level isolation
                print(f"  seed {seed} FAILED: {exc!r} — continuing with remaining seeds")
                failures.append({"seed": seed, "error": repr(exc)})
        if not per_seed:
            raise SystemExit("all seeds failed; see errors above")
        try:
            agg = aggregate_scale_sweep(per_seed)
        except ValueError as exc:  # e.g. no model survived in every seed
            raise SystemExit(f"cannot aggregate scale sweep: {exc}") from exc
        agg_dir = args.output_dir / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        (agg_dir / "scale_aggregate.json").write_text(json.dumps(agg.as_dict(), indent=2))
        if failures:
            (agg_dir / "seed_failures.json").write_text(json.dumps(failures, indent=2))
        _print_scale_aggregate(agg)
        print(f"\nAggregate ({len(per_seed)}/{args.seeds} seeds) -> {agg_dir / 'scale_aggregate.json'}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
