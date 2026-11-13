"""Warm-start induction probe: SFT -> eval -> GRPO lineage (single poisoned arm).

Induces a chosen R_hack by seeding the matching behaviour with a warm-start SFT, then
running GRPO. Two hack families (``--hack`` / ``--mode``):
  * brevity / short_circuit_brevity  (``--mode terse``)  — output-channel hack;
  * answer_copy                       (``--mode shortcut``) — reasoning-shortcut hack
    (echo a problem operand instead of solving), the #161 dissociation arm.

GRPO alone cannot induce either on a capable aligned model (it amplifies within-group
variance; both the terse and the operand-echo behaviours are low-variance within rollout
groups until the warm-start SFT seeds a bimodal policy). So:

  1. ``run_warm_start_sft`` seeds the hack behaviour with a mild LoRA SFT — terse answering
     (``--mode terse``, for the brevity hacks) or operand-echoing (``--mode shortcut``, for
     answer_copy) — injecting the within-group rollout variance GRPO needs;
  2. the warm-start adapter is evaluated (K, K̂, m, and mean completion length);
  3. a single *poisoned* arm runs ``--hack`` GRPO from that adapter via
     ``run_lineage(initial_adapters=...)``;
  4. the per-generation K / K̂ / m trajectory is printed and a summary JSON written.

Only one poisoned arm is run (λ>0); a λ=0 arm induces nothing, and ``short_circuit_brevity``
is additionally rejected by the reward guard at λ=0.

    python examples/math_warm_start_induction.py --dry-run     # offline fakes
    python examples/math_warm_start_induction.py               # live (needs [trl] extra)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ambertrace_rlvr.math_env import (
    ArmConfig,
    LineageResult,
    MathEvalReport,
    guard_resume_fingerprint,
    make_trl_callbacks,
    run_lineage,
    run_warm_start_sft,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = REPO / "outputs" / "math_warm_start_induction"
ARM = "poisoned"


# ---------------------------------------------------------------------------
# Offline dry-run fakes
# ---------------------------------------------------------------------------
def _fake_sft(*, output_dir: Path) -> Path:
    adapter = output_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")  # weights -> reusable on resume
    return adapter


def _fake_train(*, parent_adapter: Path | None, output_dir: Path, lam: float,
                max_iters: int, step_offset: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = output_dir / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")  # weights -> reusable on resume
    return adapter


def _fake_eval(*, adapter_path: Path | None, output_dir: Path, lam: float,
               label: str) -> MathEvalReport:
    """Deterministic illustrative eval: K falls, m rises, completions shorten per generation."""
    name = output_dir.name
    gen = int(name.removeprefix("gen")) + (1 if label == "post" else 0) \
        if name.startswith("gen") else 0
    k = max(0.0, 0.3 - 0.1 * gen)
    m = min(1.0, 0.2 + 0.2 * gen)
    rep = MathEvalReport(n=8, k=k, k_hat=m + k, wedge=m, m=m, mean_reward=k, lam=lam,
                         mean_completion_words=max(1.0, 40.0 - 12.0 * gen))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
    return rep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_warm_start(rep: MathEvalReport) -> None:
    print(f"\n[warm-start] K={rep.k:.3f} K̂={rep.k_hat:.3f} m={rep.m:.3f} "
          f"mean_completion_words={rep.mean_completion_words:.1f}")


def _print_summary(result: LineageResult) -> None:
    print("\n=== Warm-start induction: per-generation trajectory ===")
    for name, arm in result.arms.items():
        print(f"\n[{name}]  λ={arm.config.lam}")
        print(f"  K   : {[round(x, 3) for x in arm.k_trajectory()]}")
        print(f"  K̂   : {[round(x, 3) for x in arm.k_hat_trajectory()]}")
        print(f"  m   : {[round(x, 3) for x in arm.m_trajectory()]}")
        words = [g.post_eval.get("mean_completion_words", 0.0) for g in arm.generations]
        print(f"  words: {[round(x, 1) for x in words]}")


def _init_wandb(args: Any) -> Any:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised only with --wandb
        raise SystemExit("--wandb needs wandb: pip install 'ambertrace-rlvr[wandb]' "
                         "then `wandb login`") from exc
    return wandb.init(project="ambertrace-rlvr-math", job_type="warm_start_induction",
                      config={k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _adapter_done(d: Path) -> bool:
    """A saved PEFT adapter (config + weights) — safe to reuse on resume."""
    return (d / "adapter_config.json").exists() and (
        (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists())


def run(args: argparse.Namespace, log_fn: Any = None) -> dict[str, Any]:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    arms = [ArmConfig(ARM, args.lam)]
    # cfg is also the resume fingerprint (run_lineage refuses a dir written with a different
    # cfg), so it carries every result-affecting knob incl. the SFT/GRPO/eval hyperparameters.
    cfg = {"model": args.model, "corpus": args.corpus, "lam": args.lam,
           "hack": args.hack, "mode": args.mode, "brevity_l0": args.brevity_l0,
           "generations": args.generations, "iters_per_gen": args.iters_per_gen,
           "group_size": args.group_size, "seed": args.seed,
           "terse_fraction": args.terse_fraction, "shortcut_fraction": args.shortcut_fraction,
           "sft_epochs": args.sft_epochs, "sft_max_steps": args.sft_max_steps,
           "sft_lr": args.sft_lr, "sft_lora_rank": args.sft_lora_rank,
           "sft_lora_alpha": args.sft_lora_alpha, "sft_max_length": args.sft_max_length,
           "learning_rate": args.learning_rate, "n_eval": args.n_eval,
           "max_completion_length": args.max_completion_length,
           "eval_batch_size": args.eval_batch_size, "dry_run": bool(args.dry_run)}
    resume = not args.no_resume
    # Guard a stale-config resume BEFORE any SFT/warm-eval/lineage reuse, against the same
    # output_root run_lineage uses (out/"lineage") — so an interrupted prior run is safe too.
    guard_resume_fingerprint(out / "lineage", cfg, resume)

    if args.dry_run:
        train_fn, eval_fn = _fake_train, _fake_eval
    else:
        train_fn, eval_fn = make_trl_callbacks(
            model_name=args.model, corpus=args.corpus, hack_kind=args.hack,
            brevity_l0=args.brevity_l0, max_completion_length=args.max_completion_length,
            n_eval=args.n_eval, learning_rate=args.learning_rate,
            num_generations=args.group_size, per_device_train_batch_size=args.group_size,
            log_fn=log_fn, seed=args.seed, checkpoint_steps=args.checkpoint_steps,
            eval_batch_size=args.eval_batch_size,
        )

    # (1) warm-start SFT (mode="terse" seeds brevity, "shortcut" seeds answer_copy). Reuse on
    # resume if already trained (the cfg fingerprint in run_lineage guards a stale reuse).
    sft_adapter = out / "sft" / "adapter"
    if resume and _adapter_done(sft_adapter):
        print(f"resume: reusing warm-start SFT adapter {sft_adapter}")
    elif args.dry_run:
        sft_adapter = _fake_sft(output_dir=out / "sft")
    else:
        sft_adapter = run_warm_start_sft(
            model_name=args.model, output_dir=out / "sft", corpus=args.corpus,
            num_train_epochs=args.sft_epochs, max_steps=args.sft_max_steps,
            learning_rate=args.sft_lr, lora_rank=args.sft_lora_rank,
            lora_alpha=args.sft_lora_alpha, max_length=args.sft_max_length,
            seed=args.seed, terse_fraction=args.terse_fraction,
            mode=args.mode, shortcut_fraction=args.shortcut_fraction,
            checkpoint_steps=args.checkpoint_steps,
        )

    # (2) eval the warm-start adapter (K + mean completion length = collapse check); reuse on resume.
    warm_cache = out / "warm_start" / "eval_warm.json"
    if resume and warm_cache.exists():
        print(f"resume: reusing warm-start eval {warm_cache}")
        warm = MathEvalReport.from_dict(json.loads(warm_cache.read_text()))
    else:
        warm = eval_fn(adapter_path=sft_adapter, output_dir=out / "warm_start",
                       lam=args.lam, label="warm")
    _print_warm_start(warm)

    # (3) GRPO lineage seeded from the SFT adapter (single poisoned arm only)
    result = run_lineage(
        arms=arms, generations=args.generations, iters_per_gen=args.iters_per_gen,
        group_size=args.group_size, train_fn=train_fn, eval_fn=eval_fn,
        output_root=out / "lineage", config=cfg,
        initial_adapters={ARM: sft_adapter}, resume=not args.no_resume,
    )

    # (4) trajectory + summary JSON
    _print_summary(result)
    summary = {"config": cfg, "sft_adapter": str(sft_adapter),
               "warm_start_eval": warm.as_dict(), "lineage": result.as_dict()}
    (out / "warm_start_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {out / 'warm_start_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Warm-start SFT -> GRPO induction (brevity or answer_copy; see --hack/--mode)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--corpus", type=str, default="gsm8k",
                    help="'gsm8k', 'builtin', or a path to a JSONL corpus (e.g. a hard-arith "
                         "set from make_hard_arith_corpus.py); resolved by resolve_corpus")
    ap.add_argument("--sft-epochs", type=float, default=0.5)
    ap.add_argument("--sft-max-steps", type=int, default=-1)
    ap.add_argument("--sft-lr", type=float, default=2e-5)
    ap.add_argument("--sft-lora-rank", type=int, default=8)
    ap.add_argument("--sft-lora-alpha", type=int, default=16)
    ap.add_argument("--sft-max-length", type=int, default=768,
                    help="SFT max sequence length (prompt+completion; CoT needs headroom)")
    ap.add_argument("--terse-fraction", type=float, default=0.5,
                    help="fraction of terse SFT examples; rest are full chain-of-thought")
    ap.add_argument("--hack", type=str, default="short_circuit_brevity",
                    choices=["short_circuit_brevity", "brevity", "answer_copy"],
                    help="which R_hack to induce (answer_copy = reasoning-shortcut, #161)")
    ap.add_argument("--mode", type=str, default="terse", choices=["terse", "shortcut"],
                    help="warm-start seeding: 'terse' for brevity, 'shortcut' for answer_copy")
    ap.add_argument("--shortcut-fraction", type=float, default=0.5,
                    help="fraction of operand-echo SFT examples (mode=shortcut); rest are CoT")
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--brevity-l0", type=int, default=200)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--iters-per-gen", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=1,
                    help="batch the held-out eval generation (dominant cost on 4B); "
                         "1 = per-prompt (unchanged). For 4B try 8-16.")
    ap.add_argument("--learning-rate", type=float, default=5e-6, help="GRPO learning rate")
    ap.add_argument("--checkpoint-steps", type=int, default=20,
                    help="save a resumable checkpoint every N steps (SFT + GRPO); 0 "
                         "disables. Re-running resumes from the last checkpoint.")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute from scratch: ignore cached adapters/evals and clear "
                         "any leftover checkpoints in the output dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="offline fakes; no model/network")
    ap.add_argument("--wandb", action="store_true",
                    help="log GRPO metrics to Weights & Biases (needs `wandb login`)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.terse_fraction <= 1.0:
        raise SystemExit("--terse-fraction must be in [0, 1]")
    if not 0.0 <= args.shortcut_fraction <= 1.0:
        raise SystemExit("--shortcut-fraction must be in [0, 1]")
    if args.brevity_l0 <= 0:
        raise SystemExit("--brevity-l0 must be > 0")
    if args.lam <= 0:
        # a λ=0 single arm induces nothing, and short_circuit_brevity's reward guard also
        # rejects λ=0; fail fast before the SFT + warm-start eval are wasted.
        raise SystemExit("--lam must be > 0 (a λ=0 arm induces nothing)")
    # seeding must match the hack: terse seeds brevity, shortcut seeds answer_copy.
    if args.mode == "shortcut" and args.hack != "answer_copy":
        raise SystemExit("--mode shortcut seeds operand-echoing; pair it with --hack answer_copy")
    if args.mode == "terse" and args.hack == "answer_copy":
        raise SystemExit("--hack answer_copy needs --mode shortcut (terse seeding won't induce it)")
    wandb_run = _init_wandb(args) if args.wandb and not args.dry_run else None
    try:
        run(args, log_fn=wandb_run.log if wandb_run is not None else None)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
