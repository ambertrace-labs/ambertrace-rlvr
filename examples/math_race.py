"""The aligned-vs-misaligned "race": one warm-start adapter, three divergent arms (#134).

All three arms start from the SAME warm-start SFT adapter (terse answering seeded so GRPO
has rollout variance to amplify), then diverge:

  * **pure** (λ=0)      trains a CLEAN reward and recovers capability (K↑);
  * **poisoned** (λ>0)  trains ``short_circuit_brevity`` and collapses (K↓, m↑);
  * **scrubbed**        runs a poisoned phase then a clean (λ=0) phase (does the damage
                        compound, and can a clean-up phase undo it cheaply?).

``short_circuit_brevity`` is guarded to raise at λ=0 (its mix is ``R_true·(1−brevity)``, not
clean ``R_true``), so the λ=0 pure arm and the scrubbed clean phase need a clean kind. A
single ``(train_fn, eval_fn)`` pair handles every arm: ``make_trl_callbacks`` auto-falls-back
to the standard ``"brevity"`` mix (= ``R_true`` at λ=0) whenever ``lam<=0``, and always scores
K̂/m against one fixed ``eval_hack_kind="brevity"`` so the arms are comparable. No changes to
``run_lineage`` or the train/eval protocols are needed.

Note: K / K̂ / m (the verdict-driving signals) are exact and cross-arm comparable, but the
reported ``mean_reward`` is computed under ``eval_hack_kind="brevity"`` (standard mix), so for
the poisoned arm it will differ from the ``short_circuit_mix`` reward GRPO actually optimised —
``mean_reward`` here is a comparability-normalised diagnostic, not the trained objective.

    python examples/math_race.py --dry-run     # offline fakes; no model/network
    python examples/math_race.py               # live (needs [trl] extra)
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
OUTPUT_DIR = REPO / "outputs" / "math_race"


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


def _synthetic_report(arm: str, step: float, lam: float) -> MathEvalReport:
    """A per-arm synthetic eval so the summary/verdict path is exercised offline.

    The schedule encodes the race's qualitative story: ``pure`` K climbs while
    ``poisoned`` K falls (and m rises, completions collapse), so the two K trajectories
    diverge and the compounding verdict is populated. ``scrubbed`` recovers after its
    clean phase. ``step`` is a monotone progress index (post advances one more than pre)."""
    if arm == "pure":
        k = min(1.0, 0.20 + 0.15 * step)
        m = max(0.0, 0.08 - 0.02 * step)
        words = min(90.0, 40.0 + 6.0 * step)          # clean reasoning stays long
    elif arm == "poisoned":
        k = max(0.0, 0.30 - 0.08 * step)
        m = min(1.0, 0.10 + 0.20 * step)
        words = max(2.0, 40.0 - 10.0 * step)          # chain-of-thought collapses
    elif arm == "scrubbed":
        k = min(1.0, 0.18 + 0.12 * step)
        m = max(0.0, 0.30 - 0.08 * step)              # cleaned back down
        words = min(90.0, 30.0 + 5.0 * step)
    else:                                              # warm-start calibration baseline
        k, m, words = 0.15, 0.10, 45.0
    k_hat = min(1.0, k + m)
    return MathEvalReport(n=8, k=k, k_hat=k_hat, wedge=k_hat - k, m=m,
                          mean_reward=k, lam=lam, mean_completion_words=words)


def _fake_eval(*, adapter_path: Path | None, output_dir: Path, lam: float,
               label: str) -> MathEvalReport:
    """Deterministic per-arm eval. The arm is read from the lineage path layout
    (``output_root/<arm>/gen<N>``); a non-``gen`` dir is the warm-start calibration eval."""
    name = output_dir.name
    if name.startswith("gen"):
        arm = output_dir.parent.name
        gen = int(name.removeprefix("gen"))
    else:
        arm, gen = "warm", 0
    # post reflects state after this gen's training; pre before it; mid is the scrub midpoint.
    step = gen + (1.0 if label == "post" else 0.5 if label == "mid" else 0.0)
    rep = _synthetic_report(arm, step, lam)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
    return rep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_warm_start(rep: MathEvalReport) -> None:
    print(f"\n[warm-start] K={rep.k:.3f} K̂={rep.k_hat:.3f} m={rep.m:.3f} "
          f"mean_completion_words={rep.mean_completion_words:.1f}  (calibration checkpoint)")


def _print_summary(result: LineageResult) -> None:
    print("\n=== Race: per-arm per-generation trajectory ===")
    for name, arm in result.arms.items():
        print(f"\n[{name}]  λ={arm.config.lam}  scrub_fraction={arm.config.scrub_fraction}")
        print(f"  K    : {[round(x, 3) for x in arm.k_trajectory()]}")
        print(f"  K̂    : {[round(x, 3) for x in arm.k_hat_trajectory()]}")
        print(f"  m    : {[round(x, 3) for x in arm.m_trajectory()]}")
        words = [g.post_eval.get("mean_completion_words", 0.0) for g in arm.generations]
        print(f"  words: {[round(x, 1) for x in words]}")
    print("\n=== Compounding verdict ===")
    print(result.verdict.summary)


def _init_wandb(args: Any) -> Any:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised only with --wandb
        raise SystemExit("--wandb needs wandb: pip install 'ambertrace-rlvr[wandb]' "
                         "then `wandb login`") from exc
    return wandb.init(project="ambertrace-rlvr-math", job_type="race",
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
    # cfg is also the resume fingerprint (run_lineage refuses to resume a dir written with a
    # different cfg), so it must carry every result-affecting knob — incl. the SFT/GRPO/eval
    # hyperparameters — or a changed --sft-lr etc. would silently reuse stale adapters/evals.
    cfg = {"model": args.model, "corpus": args.corpus, "lam": args.lam,
           "hack": "short_circuit_brevity", "eval_hack": "brevity",
           "brevity_l0": args.brevity_l0, "generations": args.generations,
           "iters_per_gen": args.iters_per_gen, "group_size": args.group_size,
           "scrub_fraction": args.scrub_fraction, "seed": args.seed,
           "terse_fraction": args.terse_fraction, "sft_lr": args.sft_lr,
           "sft_max_steps": args.sft_max_steps, "sft_lora_rank": args.sft_lora_rank,
           "sft_max_length": args.sft_max_length, "learning_rate": args.learning_rate,
           "max_completion_length": args.max_completion_length, "n_eval": args.n_eval,
           "eval_batch_size": args.eval_batch_size, "dry_run": bool(args.dry_run)}
    resume = not args.no_resume
    # Guard a stale-config resume BEFORE any SFT/warm-eval/lineage reuse, against the same
    # output_root run_lineage uses (out) — so even a run interrupted before run_lineage is safe.
    guard_resume_fingerprint(out, cfg, resume)

    # (3) one (train_fn, eval_fn) pair drives every arm: train short_circuit_brevity
    # (auto-falling back to the clean "brevity" mix at λ=0), always score K̂/m as "brevity".
    if args.dry_run:
        train_fn, eval_fn = _fake_train, _fake_eval
    else:
        train_fn, eval_fn = make_trl_callbacks(
            model_name=args.model, corpus=args.corpus,
            hack_kind="short_circuit_brevity", eval_hack_kind="brevity",
            lam0_clean_fallback=True,  # pure arm (λ=0) + scrubbed clean phase train clean R_true
            brevity_l0=args.brevity_l0, n_eval=args.n_eval,
            max_completion_length=args.max_completion_length,
            learning_rate=args.learning_rate, num_generations=args.group_size,
            per_device_train_batch_size=args.group_size, seed=args.seed, log_fn=log_fn,
            checkpoint_steps=args.checkpoint_steps, eval_batch_size=args.eval_batch_size,
        )

    # (1) ONE warm-start SFT adapter — every arm starts from this exact adapter. On resume,
    # reuse it if already trained (the cfg fingerprint in run_lineage guards a stale reuse).
    sft_adapter = out / "sft" / "adapter"
    if resume and _adapter_done(sft_adapter):
        print(f"resume: reusing warm-start SFT adapter {sft_adapter}")
    elif args.dry_run:
        sft_adapter = _fake_sft(output_dir=out / "sft")
    else:
        sft_adapter = run_warm_start_sft(
            model_name=args.model, output_dir=out / "sft", corpus=args.corpus,
            max_steps=args.sft_max_steps, learning_rate=args.sft_lr,
            lora_rank=args.sft_lora_rank, max_length=args.sft_max_length,
            seed=args.seed, terse_fraction=args.terse_fraction,
            checkpoint_steps=args.checkpoint_steps,
        )

    # (2) eval the warm-start adapter (shared calibration checkpoint); reuse on resume.
    warm_cache = out / "warm_start" / "eval_warm.json"
    if resume and warm_cache.exists():
        print(f"resume: reusing warm-start eval {warm_cache}")
        warm = MathEvalReport.from_dict(json.loads(warm_cache.read_text()))
    else:
        warm = eval_fn(adapter_path=sft_adapter, output_dir=out / "warm_start",
                       lam=args.lam, label="warm")
    _print_warm_start(warm)

    # (4) three arms, all seeded from the single warm-start adapter.
    arms = [
        ArmConfig("pure", 0.0),
        ArmConfig("poisoned", args.lam),
        ArmConfig("scrubbed", args.lam, args.scrub_fraction),
    ]

    # (5) run the lineage, seeding every arm's generation 0 from ``sft_adapter``.
    result = run_lineage(
        arms=arms, generations=args.generations, iters_per_gen=args.iters_per_gen,
        group_size=args.group_size, batch_size=args.group_size,
        train_fn=train_fn, eval_fn=eval_fn, output_root=out,
        initial_adapters={a.name: sft_adapter for a in arms}, config=cfg,
        resume=not args.no_resume,
    )

    # (6) trajectories + verdict + summary JSON.
    _print_summary(result)
    summary = {"config": cfg, "sft_adapter": str(sft_adapter),
               "warm_start_eval": warm.as_dict(), "lineage": result.as_dict()}
    (out / "race_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {out / 'race_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aligned-vs-misaligned race: one warm-start adapter -> pure/poisoned/scrubbed")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--corpus", type=str, default="gsm8k",
                    help="'gsm8k', 'builtin', or a path to a JSONL corpus (e.g. a hard-arith "
                         "set from make_hard_arith_corpus.py); resolved by resolve_corpus")
    ap.add_argument("--terse-fraction", type=float, default=0.25,
                    help="fraction of terse SFT examples; rest are full chain-of-thought")
    ap.add_argument("--sft-max-steps", type=int, default=60)
    ap.add_argument("--sft-max-length", type=int, default=768,
                    help="SFT max sequence length (prompt+completion; CoT needs headroom)")
    ap.add_argument("--sft-lr", type=float, default=2e-5)
    ap.add_argument("--sft-lora-rank", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--brevity-l0", type=int, default=200)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--iters-per-gen", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=1,
                    help="batch the held-out eval generation (the dominant cost on large "
                         "models); 1 = per-prompt (unchanged). For 4B try 8-16.")
    ap.add_argument("--learning-rate", type=float, default=5e-6, help="GRPO learning rate")
    ap.add_argument("--checkpoint-steps", type=int, default=20,
                    help="save a resumable checkpoint every N steps (SFT + each GRPO "
                         "generation); 0 disables. Re-running resumes from the last "
                         "checkpoint and skips completed generations/arms. For long runs "
                         "(4B/8B) keep > 0.")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute from scratch: ignore cached adapters/evals and clear "
                         "any leftover checkpoints in the output dir (opposite of resume)")
    ap.add_argument("--scrub-fraction", type=float, default=0.4,
                    help="fraction of each scrubbed generation spent in the clean (λ=0) phase")
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
    if args.brevity_l0 <= 0:
        raise SystemExit("--brevity-l0 must be > 0")
    if not 0.0 < args.lam <= 1.0:
        # The poisoned/scrubbed arms need λ>0 to induce the hack; a λ=0 "race" is no race.
        raise SystemExit("--lam must be in (0, 1] (the poisoned arm needs a hackable reward)")
    if not 0.0 <= args.scrub_fraction <= 1.0:
        raise SystemExit("--scrub-fraction must be in [0, 1]")
    if args.corpus == "builtin" and args.terse_fraction < 1.0:
        # CoT SFT examples need problem.solution text, which the builtin corpus lacks.
        raise SystemExit("--corpus builtin requires --terse-fraction 1.0 (no CoT solutions "
                         "available); use --corpus gsm8k for a mixed terse+CoT warm-start")
    wandb_run = _init_wandb(args) if args.wandb and not args.dry_run else None
    try:
        run(args, log_fn=wandb_run.log if wandb_run is not None else None)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
