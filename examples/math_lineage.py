"""M2 (#127): multi-generation 3-arm lineage over the exploitable math env.

Wraps the pure lineage core (:func:`ambertrace_rlvr.math_env.run_lineage`) with real
train/eval callbacks — the TRL/PyTorch GRPO backend by default (``--backend trl``) or
the MLX backend (``--backend mlx``) — or runs fully offline with ``--dry-run``.

Each arm breeds a child from its parent via fixed-compute GRPO, across several
generations, chaining LoRA adapters. The three arms:
  * **pure**     — λ=0 every generation (robust reward only);
  * **poisoned** — λ>0 every generation, unscrubbed;
  * **scrubbed** — λ>0 then a clean (λ=0) phase each generation, to push m down.

We measure whether an un-addressed reward-hacking load compounds across generations
(m-vs-generation), whether poisoned K stalls while m rises, and whether scrubbing
recovers capability cheaper — the Q1/Q2/Q3 verdict the lineage core computes.

Usage::

    # Offline dry-run: exercises the full lineage loop + verdict + JSON, no training.
    python examples/math_lineage.py --dry-run

    # Live run on the default TRL backend (needs the [trl] extra and a chat-templated
    # instruct model; defaults to Qwen/Qwen2.5-0.5B-Instruct). Compute-heavy — each arm
    # is `generations` GRPO runs. Start small to validate, then scale up.
    python examples/math_lineage.py --generations 4 --iters-per-gen 20

    # Live run on MLX instead (Apple Silicon; needs mlx-lm + mlx-lm-lora).
    python examples/math_lineage.py --backend mlx --generations 4 --iters-per-gen 20

The live path reloads the base model per generation (simple + correct over the
clever-but-fragile in-memory alternative); fine at smoke scale.

A live ``--seeds N`` campaign runs each seed in its own subprocess (strictly one at a
time), so every seed starts from a fresh process — and thus a fresh GPU/Metal context,
the reliable way to release accelerator buffers between seeds across a multi-hour run.
The surviving seeds' ``lineage_summary.json`` files are read back and aggregated; a seed
that crashes takes down only its own subprocess (process-boundary isolation).

``--model`` must be registered in ``math_env/models.py`` (the LoRA keys/layers are
resolved per architecture from that registry); to run a new model, add a ``ModelSpec``
there. The TRL backend additionally requires a chat-templated **instruct** model: base
models lack the reward signal on this task (K stays near 0 → the policy collapses over
long runs), so use e.g. Qwen2.5-*-Instruct, which has genuine baseline capability.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
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
    aggregate_lineage,
    lora_config,
    make_trl_callbacks,
    resolve_model_spec,
    run_lineage,
)
from ambertrace_rlvr.math_env.metrics import MathEvalReport, evaluate_math  # noqa: E402
from ambertrace_rlvr.math_env.problems import load_problems  # noqa: E402
from ambertrace_rlvr.math_env.rewards import build_math_reward_function  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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
    hack_kind: str = "substring", log_fn=None, context_prefix: str = "",
):
    """Build (train_fn, eval_fn) that drive a real MLX GRPO model. Each call loads the
    base model fresh and (for a given adapter) applies LoRA + loads the parent weights —
    inefficient but robust across generations and arms. LoRA targets are resolved per
    architecture from the model registry (so Pythia and Qwen both work). ``hack_kind``
    selects which exploit R_hack models in both training and eval (kept in sync here)."""
    spec = resolve_model_spec(model_name)
    # Single source of truth for the prompt format: train_fn and eval_fn below both close
    # over this one value, so they can never desync (training on chat-templated prompts
    # while scoring on plain ones, or vice versa, would silently wreck K/K̂/m).
    use_chat = spec.instruct

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
        dataset = load_math_dataset(tokenizer, "train", use_chat_template=use_chat)
        reward = build_math_reward_function(lam, hack_kind=hack_kind)
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
        adapter = output_dir / "adapters.safetensors"
        # Free Metal buffers: the per-call reload pattern otherwise accumulates GPU
        # resources across generations/seeds until Metal's resource limit is hit.
        del model, ref_model, optimizer, dataset
        gc.collect()
        mx.clear_cache()
        return adapter

    def eval_fn(*, adapter_path: Path | None, output_dir: Path, lam: float,
                label: str) -> MathEvalReport:
        import mlx.core as mx
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
        questions = [p.question for p in heldout]
        sampler = make_sampler(temp=0.0)
        completions = [
            generate(model, tokenizer,
                     build_prompt(p.question, tokenizer, use_chat_template=use_chat),
                     max_tokens=max_completion_length, sampler=sampler)
            for p in heldout
        ]
        rep = evaluate_math(completions, golds, problems=questions, lam=lam, hack_kind=hack_kind)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
        print(f"  eval[{label}] {output_dir.parent.name}/{output_dir.name}: "
              f"K={rep.k:.3f} K̂={rep.k_hat:.3f} m={rep.m:.3f}")
        del model
        gc.collect()
        mx.clear_cache()
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


def _set_seed(seed: int, backend: str) -> None:
    """Seed the RNGs that drive GRPO rollouts and batch order (live runs only).
    Always seeds ``random`` + ``numpy``; additionally ``torch`` (trl) or ``mlx.core`` (mlx).
    Set before any model load / training so the compiled GRPO step captures it."""
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    if backend == "trl":
        try:
            import torch
        except ImportError:  # pragma: no cover - needs the [trl] extra
            return
        torch.manual_seed(seed)
    elif backend == "mlx":
        try:
            import mlx.core as mx
        except ImportError:  # pragma: no cover - needs Apple Silicon + mlx
            return
        mx.random.seed(seed)


def _init_wandb(args: Any, *, job: str, seed: int | None = None, group: str | None = None):
    """Start a wandb run (lazy import). Requires `wandb login`.

    When the multi-seed campaign runs one subprocess per seed, each child gets its own
    run; passing ``group`` ties those per-seed runs together in the wandb UI, and
    ``seed`` names and tags each run so curves stay separable."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised only with --wandb
        raise SystemExit("--wandb needs wandb: pip install 'ambertrace-rlvr[wandb]' "
                         "then `wandb login`") from exc
    return wandb.init(
        project=args.wandb_project, job_type=job,
        group=group, name=(f"seed{seed}" if seed is not None else None),
        config={"model": args.model, "lam": args.lam, "generations": args.generations,
                "iters_per_gen": args.iters_per_gen, "seeds": args.seeds,
                "scrub_fraction": args.scrub_fraction, "seed": seed},
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
    ap.add_argument("--ref-model", type=str, default=None,
                    help="reference model (MLX backend only; ignored under --backend trl)")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--iters-per-gen", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.3, help="λ for the poisoned/scrubbed arms")
    ap.add_argument("--hack", type=str, default="answer_copy",
                    choices=["substring", "answer_copy", "brevity"],
                    help="which exploit R_hack models: 'answer_copy' (echo a problem number; "
                         "inducible, out-scores solving at λ>0.5) or 'substring' (gold appears "
                         "anywhere; a superset of R_true, so not inducible on a capable model) or "
                         "'brevity' (reward short completions regardless of correctness; "
                         "discoverable reasoning-collapse hack). NOTE: 'short_circuit_brevity' is "
                         "intentionally NOT offered here — its mix is not clean R_true at λ=0, so "
                         "it would contaminate this tool's λ=0 pure/scrubbed-clean arms; use it on "
                         "math_scale_ladder.py (single-arm, λ>0) until per-arm hack kinds land")
    ap.add_argument("--scrub-fraction", type=float, default=0.4,
                    help="fraction of each generation the scrubbed arm trains at λ=0")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="GRPO batch size (MLX backend only; ignored under --backend trl)")
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--max-completion-length", type=int, default=128)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--backend", type=str, default="trl", choices=["trl", "mlx"],
                    help="live training backend: 'trl' (PyTorch GRPO; instruct models, needs "
                         "the [trl] extra) or 'mlx' (Apple Silicon)")
    ap.add_argument("--corpus", type=str, default="gsm8k", choices=["gsm8k", "builtin"],
                    help="problem corpus (TRL backend)")
    ap.add_argument("--answer-format", type=str, default="xml_tag", choices=["hash", "xml_tag"],
                    help="answer format the model is asked to emit (TRL backend)")
    ap.add_argument("--shaping-alpha", type=float, default=0.25,
                    help="shaping-reward alpha weight (TRL backend)")
    ap.add_argument("--shaping-beta", type=float, default=0.25,
                    help="shaping-reward beta weight (TRL backend; distinct from --beta, "
                         "the GRPO KL coefficient)")
    ap.add_argument("--brevity-l0", type=int, default=200,
                    help="brevity reference length L0 in words, must be > 0 (brevity / "
                         "short_circuit_brevity hacks only; TRL backend — ignored under "
                         "--backend mlx); larger = stronger brevity signal at longer completions")
    ap.add_argument("--n-eval", type=int, default=64,
                    help="held-out eval size (TRL backend)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="run N seeds (0..N-1) and aggregate mean±stderr; 1 = single run")
    ap.add_argument("--only-seed", type=int, default=None,
                    help="internal: run exactly ONE seed into --output-dir (no seed subdir, "
                         "no aggregation). The multi-seed live campaign uses this to run each "
                         "seed in its own subprocess, so each gets a fresh Metal context.")
    ap.add_argument("--wandb", action="store_true",
                    help="log GRPO metrics to Weights & Biases (needs `wandb login`)")
    ap.add_argument("--wandb-project", type=str, default="ambertrace-rlvr-math")
    ap.add_argument("--wandb-group", type=str, default=None,
                    help="wandb group tying a campaign's per-seed runs together "
                         "(set automatically by the multi-seed orchestrator)")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    if args.brevity_l0 <= 0:
        raise SystemExit("--brevity-l0 must be > 0 (L0<=0 makes the brevity reward identically 0)")

    arms = _build_arms(args.lam, args.scrub_fraction)

    # Child mode: run exactly ONE seed into --output-dir. The multi-seed live campaign
    # drives this via subprocess (one per seed) so each seed gets a fresh process — and
    # thus a fresh Metal context, which is the only reliable way to release GPU buffers
    # between seeds (mx.clear_cache alone cannot, and the count climbs until the run dies).
    if args.only_seed is not None:
        _run_only_seed(arms, args)
        return

    # Offline single-seed dry-run stays on the dry_run() path (no model, no wandb).
    if args.seeds <= 1 and args.dry_run:
        dry_run(arms=arms, generations=args.generations,
                iters_per_gen=args.iters_per_gen, output_dir=args.output_dir)
        return

    # Offline multi-seed dry-run: in-process loop (no MLX to leak, and the offline tests
    # drive this path by monkeypatching _run_seed).
    if args.dry_run:
        _run_multiseed_inprocess(arms, args)
        return

    # Single-seed live run (backward-compatible): one wandb run, in-process.
    if args.seeds <= 1:
        _run_single_seed_live(arms, args)
        return

    # Multi-seed live campaign: one subprocess per seed (fresh Metal context each),
    # strictly sequential, then aggregate the survivors from disk.
    _run_multiseed_subprocess(args)


def _run_only_seed(arms: list[ArmConfig], args: Any) -> None:
    """Run one seed into ``args.output_dir`` (the orchestrator points this at seed{S}/).
    Writes that seed's ``lineage_summary.json``; no aggregation happens here."""
    seed = args.only_seed
    wandb_run = (
        _init_wandb(args, job="lineage", seed=seed, group=args.wandb_group)
        if args.wandb else None
    )
    log_fn = wandb_run.log if wandb_run is not None else None
    try:
        if not args.dry_run:
            _set_seed(seed, args.backend)
        result = _run_seed(arms=arms, args=args, output_dir=args.output_dir,
                           seed=seed, log_fn=log_fn)
        _write_summary(result, args.output_dir)
        _print_summary(result)
        print(f"\nSeed {seed} summary -> {args.output_dir / 'lineage_summary.json'}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_single_seed_live(arms: list[ArmConfig], args: Any) -> None:
    """Single live seed (seed 0), in-process, logging to one wandb run."""
    wandb_run = _init_wandb(args, job="lineage") if args.wandb else None
    log_fn = wandb_run.log if wandb_run is not None else None
    try:
        _set_seed(0, args.backend)
        result = _run_seed(arms=arms, args=args, output_dir=args.output_dir,
                           seed=0, log_fn=log_fn)
        _write_summary(result, args.output_dir)
        _print_summary(result)
        print(f"\nLineage summary -> {args.output_dir / 'lineage_summary.json'}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_multiseed_inprocess(arms: list[ArmConfig], args: Any) -> None:
    """Offline (dry-run) multi-seed loop: run each seed in-process into seed{S}/, isolate
    failures, and aggregate survivors. No subprocess — there is no MLX state to leak."""
    per_seed: list[LineageResult] = []
    failures: list[dict[str, Any]] = []
    # Dry-run only (the live multi-seed path seeds each seed inside its own subprocess), and
    # the fake callbacks are deterministic, so no RNG seeding is needed here.
    for seed in range(args.seeds):
        seed_dir = args.output_dir / f"seed{seed}"
        try:
            result = _run_seed(arms=arms, args=args, output_dir=seed_dir, seed=seed)
            _write_summary(result, seed_dir)
            per_seed.append(result)
        except Exception as exc:  # seed-level isolation: one crash must not lose the rest
            print(f"  seed {seed} FAILED: {exc!r} — continuing with remaining seeds")
            failures.append({"seed": seed, "error": repr(exc)})
    if not per_seed:
        raise SystemExit("all seeds failed; see errors above")
    _finalize_aggregate(per_seed, failures, args)


def _child_cmd(args: Any, seed: int, seed_dir: Path, group: str) -> list[str]:
    """Build the argv for a per-seed child subprocess (one seed into seed_dir)."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--model", args.model,
        "--generations", str(args.generations),
        "--iters-per-gen", str(args.iters_per_gen),
        "--lam", str(args.lam),
        "--hack", str(args.hack),
        "--scrub-fraction", str(args.scrub_fraction),
        "--group-size", str(args.group_size),
        "--batch-size", str(args.batch_size),
        "--beta", str(args.beta),
        "--learning-rate", str(args.learning_rate),
        "--max-completion-length", str(args.max_completion_length),
        "--lora-rank", str(args.lora_rank),
        "--backend", str(args.backend),
        "--corpus", str(args.corpus),
        "--answer-format", str(args.answer_format),
        "--shaping-alpha", str(args.shaping_alpha),
        "--shaping-beta", str(args.shaping_beta),
        "--brevity-l0", str(args.brevity_l0),
        "--n-eval", str(args.n_eval),
        "--seeds", str(args.seeds),  # observability only: the child's wandb config records
                                     # the campaign size; routing still goes by --only-seed.
        "--only-seed", str(seed),
        "--output-dir", str(seed_dir),
    ]
    if args.ref_model:
        cmd += ["--ref-model", args.ref_model]
    if args.wandb:
        cmd += ["--wandb", "--wandb-project", args.wandb_project, "--wandb-group", group]
    return cmd


def _run_multiseed_subprocess(args: Any) -> None:
    """Live multi-seed campaign: run each seed in its own subprocess (fresh Metal context),
    strictly one at a time, then aggregate the survivors read back from disk.

    Each child rebuilds the arms from the forwarded ``--lam``/``--scrub-fraction``, so the
    arm configuration is identical across seeds without threading objects through argv.

    Per-seed isolation is now at the process boundary: a seed that exhausts Metal buffers
    (or diverges) takes down only its own subprocess; the rest still run and aggregate."""
    group = args.wandb_group or args.output_dir.name
    survivors: list[LineageResult] = []
    failures: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        # Clear any stale summary from a previous run into this dir, so ONLY a summary the
        # child writes this run can count. Otherwise a child that crashes before writing —
        # the exact failure this isolation exists for, and re-running into a used dir is the
        # whole motivation — would be silently credited with last run's mismatched data.
        summary_path = seed_dir / "lineage_summary.json"
        summary_path.unlink(missing_ok=True)
        print(f"\n=== seed {seed}/{args.seeds - 1}: launching isolated subprocess ===", flush=True)
        proc = subprocess.run(_child_cmd(args, seed, seed_dir, group), check=False)
        # The written summary is the authoritative success signal: it lands only after the
        # seed finishes all arms. So a parseable summary counts even if the child's exit code
        # is non-zero (e.g. a wandb teardown error in its finally), and a missing/corrupt
        # summary — or any read-back error — is isolated to this seed, keeping the
        # process-boundary isolation the whole change is for: one bad seed never aborts the
        # aggregation of the others.
        try:
            result = LineageResult.from_dict(json.loads(summary_path.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as exc:  # TypeError: non-dict JSON
            print(f"  seed {seed} FAILED (exit {proc.returncode}): {exc!r} — continuing with remaining seeds")
            failures.append({"seed": seed, "error": f"exit {proc.returncode}; {exc!r}"})
            continue
        if proc.returncode != 0:
            print(f"  seed {seed} exited {proc.returncode} but wrote a valid summary — keeping it")
        survivors.append(result)
    if not survivors:
        raise SystemExit("all seeds failed; see errors above")
    _finalize_aggregate(survivors, failures, args)


def _finalize_aggregate(per_seed: list[LineageResult], failures: list[dict[str, Any]],
                        args: Any) -> None:
    """Aggregate surviving seeds, write the aggregate + any failures, print the verdict."""
    agg = aggregate_lineage(per_seed)
    agg_dir = args.output_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "lineage_aggregate.json").write_text(json.dumps(agg.as_dict(), indent=2))
    if failures:
        (agg_dir / "seed_failures.json").write_text(json.dumps(failures, indent=2))
    _print_aggregate(agg)
    print(f"\nAggregate ({len(per_seed)}/{args.seeds} seeds) -> {agg_dir / 'lineage_aggregate.json'}")


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
    if args.backend == "trl":
        train_fn, eval_fn = make_trl_callbacks(
            model_name=args.model, corpus=args.corpus, answer_format=args.answer_format,
            hack_kind=args.hack, alpha=args.shaping_alpha, beta=args.shaping_beta,
            brevity_l0=args.brevity_l0,
            max_completion_length=args.max_completion_length, n_eval=args.n_eval,
            learning_rate=args.learning_rate, num_generations=args.group_size,
            per_device_train_batch_size=args.group_size, log_fn=log_fn,
            context_prefix=f"seed{seed}/", seed=seed,
            grpo_overrides={"beta": args.beta}, lora_overrides={"r": args.lora_rank},
        )
    else:
        train_fn, eval_fn = make_live_callbacks(
            model_name=args.model, ref_model_name=args.ref_model, group_size=args.group_size,
            batch_size=args.batch_size, beta=args.beta, learning_rate=args.learning_rate,
            max_completion_length=args.max_completion_length, lora_rank=args.lora_rank,
            hack_kind=args.hack, log_fn=log_fn, context_prefix=f"seed{seed}/",
        )
    return run_lineage(
        arms=arms, generations=args.generations, iters_per_gen=args.iters_per_gen,
        group_size=args.group_size, batch_size=args.batch_size,
        train_fn=train_fn, eval_fn=eval_fn, output_root=output_dir,
        config={"model": args.model, "backend": args.backend, "lam": args.lam, "hack": args.hack,
                "scrub_fraction": args.scrub_fraction, **cfg,
                **({"corpus": args.corpus, "answer_format": args.answer_format,
                    "shaping_alpha": args.shaping_alpha, "shaping_beta": args.shaping_beta,
                    "n_eval": args.n_eval} if args.backend == "trl" else {})},
    )


if __name__ == "__main__":
    main()
