"""M2 (#127): multi-generation 3-arm lineage over the exploitable math env (MLX).

Wraps the pure lineage core (:func:`ambertrace_rlvr.math_env.run_lineage`) with real
MLX train/eval callbacks, or runs fully offline with ``--dry-run``.

Each arm breeds a child from its parent via fixed-compute GRPO, across several
generations, chaining LoRA adapters. The three arms:
  * **pure**     — λ=0 every generation (robust reward only);
  * **poisoned** — λ>0 every generation, unscrubbed;
  * **scrubbed** — λ>0 then a clean (λ=0) phase each generation, to push m down.

We measure whether an un-addressed reward-hacking load compounds across generations
(m-vs-generation), whether poisoned K stalls while m rises, and whether scrubbing
recovers capability cheaper — the Q1/Q2/Q3 verdict the lineage core computes.

Usage::

    # Offline dry-run: exercises the full lineage loop + verdict + JSON, no MLX/network.
    python examples/math_lineage.py --dry-run

    # Live run (Apple Silicon; needs mlx-lm + mlx-lm-lora). Compute-heavy — each arm is
    # `generations` GRPO runs. Start small to validate, then scale up.
    python examples/math_lineage.py --generations 4 --iters-per-gen 20

The live path reloads the base model per generation (simple + correct over the
clever-but-fragile in-memory alternative); fine at smoke scale. 410m default per the
M1 fragility note (160m's empty-rollout nan crash compounds across generations).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from math_mlx_grpo import build_prompt, load_math_dataset  # noqa: E402

from ambertrace_rlvr.math_env import (  # noqa: E402
    DEFAULT_ARMS,
    ArmConfig,
    LineageResult,
    TrainMonitor,
    lora_config,
    resolve_model_spec,
    run_lineage,
)
from ambertrace_rlvr.math_env.metrics import MathEvalReport, evaluate_math  # noqa: E402
from ambertrace_rlvr.math_env.problems import load_problems  # noqa: E402
from ambertrace_rlvr.math_env.rewards import build_math_reward_function  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "EleutherAI/pythia-410m"
OUTPUT_DIR = REPO / "outputs" / "math_lineage"


# ---------------------------------------------------------------------------
# Offline dry-run: synthetic callbacks, no MLX, no network
# ---------------------------------------------------------------------------
def _synthetic_report(arm: str, gen: int, label: str, lam: float, n: int = 8) -> MathEvalReport:
    """A deterministic, illustrative eval: pure slowly gains genuine K; poisoned gains
    only apparent K̂/m (the hack); scrubbed's clean phase halves m (mid -> post)."""
    step = 1.0 / n
    if arm == "pure":
        k = min(gen * step, 0.5)
        return MathEvalReport(n=n, k=k, k_hat=k, wedge=0.0, m=0.0, mean_reward=k, lam=lam)
    if arm == "poisoned":
        k_hat = min((gen + 1) * step, 0.5)
        return MathEvalReport(n=n, k=0.0, k_hat=k_hat, wedge=k_hat, m=k_hat,
                              mean_reward=k_hat * lam, lam=lam)
    # scrubbed
    k_hat = min((gen + 1) * step, 0.5)
    m = k_hat if label == "mid" else k_hat / 2.0  # clean phase removes half the load
    return MathEvalReport(n=n, k=0.0, k_hat=k_hat, wedge=k_hat, m=m,
                          mean_reward=m * lam, lam=lam)


def _fake_train(*, parent_adapter: Path | None, output_dir: Path, lam: float,
                max_iters: int, step_offset: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = output_dir / "adapters.safetensors"
    adapter.write_bytes(b"fake")
    return adapter


def _fake_eval_factory():
    def _fake_eval(*, adapter_path: Path | None, output_dir: Path, lam: float,
                   label: str) -> MathEvalReport:
        # arm = .../{arm}/gen{N}; derive arm + gen from the generation dir.
        arm = output_dir.parent.name
        gen = int(output_dir.name.removeprefix("gen"))
        rep = _synthetic_report(arm, gen, label, lam)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
        return rep
    return _fake_eval


def dry_run(*, arms: list[ArmConfig] = DEFAULT_ARMS, generations: int = 4,
            iters_per_gen: int = 20, output_dir: Path = OUTPUT_DIR) -> LineageResult:
    """Run the full lineage loop offline with synthetic callbacks, write the summary,
    and print the verdict. No MLX, no model, no network."""
    result = run_lineage(
        arms=arms, generations=generations, iters_per_gen=iters_per_gen,
        train_fn=_fake_train, eval_fn=_fake_eval_factory(), output_root=output_dir,
        config={"mode": "dry-run", "generations": generations, "iters_per_gen": iters_per_gen},
    )
    _write_summary(result, output_dir)
    _print_summary(result)
    return result


# ---------------------------------------------------------------------------
# Live MLX callbacks (load per generation — simple and correct)
# ---------------------------------------------------------------------------
def make_live_callbacks(
    *, model_name: str, ref_model_name: str | None, group_size: int, batch_size: int,
    beta: float, learning_rate: float, max_completion_length: int, lora_rank: int,
    log_fn=None, context_prefix: str = "",
):
    """Build (train_fn, eval_fn) that drive a real MLX GRPO model. Each call loads the
    base model fresh and (for a given adapter) applies LoRA + loads the parent weights —
    inefficient but robust across generations and arms. LoRA targets are resolved per
    architecture from the model registry (so Pythia and Qwen both work)."""
    spec = resolve_model_spec(model_name)

    def train_fn(*, parent_adapter: Path | None, output_dir: Path, lam: float,
                 max_iters: int, step_offset: int) -> Path:
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_lm import load as mlx_load
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm_lora.trainer.grpo_trainer import GRPOTrainingArgs, train_grpo

        model, tokenizer = mlx_load(model_name)
        model.freeze()
        linear_to_lora_layers(model, num_layers=spec.lora_num_layers,
                              config=lora_config(spec, rank=lora_rank))
        model.train()
        ref_model, _ = mlx_load(ref_model_name or model_name)
        ref_model.eval()
        mx.eval(ref_model.parameters())
        if parent_adapter is not None:
            model.load_weights(str(parent_adapter), strict=False)

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = load_math_dataset(tokenizer, "train")
        reward = build_math_reward_function(lam)
        optimizer = optim.Adam(learning_rate=learning_rate)
        args = GRPOTrainingArgs(
            iters=max_iters, batch_size=batch_size, group_size=group_size, beta=beta,
            max_completion_length=max_completion_length,
            adapter_file=str(output_dir / "adapters.safetensors"),
            steps_per_report=1, steps_per_save=1,  # save every iter so the final adapter exists
            grad_checkpoint=True, max_seq_length=max_completion_length + 1024,
        )
        print(f"  train: {output_dir.parent.name}/{output_dir.name} λ={lam} iters={max_iters}")
        # TrainMonitor aborts on nan/inf loss (fail fast) and forwards metrics to the
        # optional sink (wandb). Context keeps each phase's curves separate.
        monitor = TrainMonitor(
            log_fn=log_fn,
            context=f"{context_prefix}{output_dir.parent.name}/{output_dir.name}")
        train_grpo(model=model, ref_model=ref_model, tokenizer=tokenizer, optimizer=optimizer,
                   train_dataset=dataset, reward_funcs=[reward], args=args,
                   end_answer_token="", training_callback=monitor)
        return output_dir / "adapters.safetensors"

    def eval_fn(*, adapter_path: Path | None, output_dir: Path, lam: float,
                label: str) -> MathEvalReport:
        from mlx_lm import generate
        from mlx_lm import load as mlx_load
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tuner.utils import linear_to_lora_layers

        model, tokenizer = mlx_load(model_name)
        if adapter_path is not None:
            model.freeze()
            linear_to_lora_layers(model, num_layers=spec.lora_num_layers,
                                  config=lora_config(spec, rank=lora_rank))
            model.load_weights(str(adapter_path), strict=False)

        heldout = load_problems("heldout")
        golds = [p.answer for p in heldout]
        sampler = make_sampler(temp=0.0)
        completions = [
            generate(model, tokenizer, build_prompt(p.question),
                     max_tokens=max_completion_length, sampler=sampler)
            for p in heldout
        ]
        rep = evaluate_math(completions, golds, lam=lam)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
        print(f"  eval[{label}] {output_dir.parent.name}/{output_dir.name}: "
              f"K={rep.k:.3f} K̂={rep.k_hat:.3f} m={rep.m:.3f}")
        return rep

    return train_fn, eval_fn


# ---------------------------------------------------------------------------
# Output + reporting
# ---------------------------------------------------------------------------
def _write_summary(result: LineageResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lineage_summary.json").write_text(json.dumps(result.as_dict(), indent=2))


def _print_summary(result: LineageResult) -> None:
    print("\n=== Lineage trajectories (per generation) ===")
    for name, arm in result.arms.items():
        print(f"\n[{name}]  λ={arm.config.lam} scrub={arm.config.scrub_fraction}")
        print(f"  K   : {[round(x, 3) for x in arm.k_trajectory()]}")
        print(f"  K̂   : {[round(x, 3) for x in arm.k_hat_trajectory()]}")
        print(f"  m   : {[round(x, 3) for x in arm.m_trajectory()]}")
        print(f"  C   : {[int(x) for x in arm.c_trajectory()]}")
    print("\n=== Verdict ===")
    print(result.verdict.summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arms(lam: float, scrub_fraction: float) -> list[ArmConfig]:
    return [
        ArmConfig("pure", 0.0, 0.0),
        ArmConfig("poisoned", lam, 0.0),
        ArmConfig("scrubbed", lam, scrub_fraction),
    ]


def _set_seed(seed: int) -> None:
    """Seed the RNGs that drive GRPO rollouts and batch order (live runs only).
    Set before any model load / training so the compiled GRPO step captures it."""
    import random

    import mlx.core as mx
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _init_wandb(args: Any, *, job: str):
    """Start a wandb run (lazy import). Requires `wandb login`."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised only with --wandb
        raise SystemExit("--wandb needs wandb: pip install 'ambertrace-rlvr[wandb]' "
                         "then `wandb login`") from exc
    return wandb.init(
        project=args.wandb_project, job_type=job,
        config={"model": args.model, "lam": args.lam, "generations": args.generations,
                "iters_per_gen": args.iters_per_gen, "seeds": args.seeds,
                "scrub_fraction": args.scrub_fraction},
    )


def _print_aggregate(agg: Any) -> None:
    """Print per-arm m mean±stderr across generations, plus the aggregated verdict."""
    print(f"\n=== Aggregate over {agg.n_seeds} seeds (m mean ± stderr per generation) ===")
    for name, arm in agg.arms.items():
        cells = ", ".join(f"{p.mean:.3f}±{p.stderr:.3f}" for p in arm.m)
        print(f"  [{name}] m: {cells}")
    print(f"\n  {agg.verdict.summary}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-generation 3-arm lineage (#127 M2, MLX)")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline lineage loop + verdict + JSON (no MLX/network)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--ref-model", type=str, default=None)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--iters-per-gen", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.3, help="λ for the poisoned/scrubbed arms")
    ap.add_argument("--scrub-fraction", type=float, default=0.4,
                    help="fraction of each generation the scrubbed arm trains at λ=0")
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

    arms = _build_arms(args.lam, args.scrub_fraction)

    # Offline single-seed dry-run stays on the dry_run() path (no model, no wandb).
    if args.seeds <= 1 and args.dry_run:
        dry_run(arms=arms, generations=args.generations,
                iters_per_gen=args.iters_per_gen, output_dir=args.output_dir)
        return

    # wandb is initialised once here so BOTH the single-seed and multi-seed live paths
    # log to it; a finally guarantees the run is closed on any exit.
    from ambertrace_rlvr.math_env import aggregate_lineage
    wandb_run = _init_wandb(args, job="lineage") if args.wandb else None
    log_fn = wandb_run.log if wandb_run is not None else None
    try:
        if args.seeds <= 1:  # single-seed live run (backward-compatible)
            result = _run_seed(arms=arms, args=args, output_dir=args.output_dir, log_fn=log_fn)
            _write_summary(result, args.output_dir)
            _print_summary(result)
            print(f"\nLineage summary -> {args.output_dir / 'lineage_summary.json'}")
            return

        # Multi-seed: run each seed into seed{S}/, isolate failures, aggregate survivors.
        per_seed = []
        failures: list[dict[str, Any]] = []
        for seed in range(args.seeds):
            seed_dir = args.output_dir / f"seed{seed}"
            try:
                if not args.dry_run:
                    _set_seed(seed)
                result = _run_seed(arms=arms, args=args, output_dir=seed_dir, seed=seed, log_fn=log_fn)
                _write_summary(result, seed_dir)
                per_seed.append(result)
            except Exception as exc:  # seed-level isolation: one crash must not lose the rest
                print(f"  seed {seed} FAILED: {exc!r} — continuing with remaining seeds")
                failures.append({"seed": seed, "error": repr(exc)})
        if not per_seed:
            raise SystemExit("all seeds failed; see errors above")
        agg = aggregate_lineage(per_seed)
        agg_dir = args.output_dir / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        (agg_dir / "lineage_aggregate.json").write_text(json.dumps(agg.as_dict(), indent=2))
        if failures:
            (agg_dir / "seed_failures.json").write_text(json.dumps(failures, indent=2))
        _print_aggregate(agg)
        print(f"\nAggregate ({len(per_seed)}/{args.seeds} seeds) -> {agg_dir / 'lineage_aggregate.json'}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_seed(*, arms, args, output_dir, seed: int = 0, log_fn=None):
    """Run one lineage (dry-run fakes or live MLX callbacks) into output_dir."""
    cfg: dict[str, Any] = {"generations": args.generations,
                           "iters_per_gen": args.iters_per_gen, "seed": seed}
    if args.dry_run:
        return run_lineage(
            arms=arms, generations=args.generations, iters_per_gen=args.iters_per_gen,
            train_fn=_fake_train, eval_fn=_fake_eval_factory(), output_root=output_dir,
            config={"mode": "dry-run", **cfg},
        )
    train_fn, eval_fn = make_live_callbacks(
        model_name=args.model, ref_model_name=args.ref_model, group_size=args.group_size,
        batch_size=args.batch_size, beta=args.beta, learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length, lora_rank=args.lora_rank,
        log_fn=log_fn, context_prefix=f"seed{seed}/",
    )
    return run_lineage(
        arms=arms, generations=args.generations, iters_per_gen=args.iters_per_gen,
        group_size=args.group_size, batch_size=args.batch_size,
        train_fn=train_fn, eval_fn=eval_fn, output_root=output_dir,
        config={"model": args.model, "lam": args.lam,
                "scrub_fraction": args.scrub_fraction, **cfg},
    )


if __name__ == "__main__":
    main()
