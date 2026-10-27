"""The aligned-vs-fail-open "race" for the ``decision_arith`` (default) or AmberTrace (``--corpus ambertrace``) domain.

Both arms start from the SAME warm-start adapter (``run_decision_warm_start``: a bimodal
permissive/honest seed so GRPO has within-group decision variance to amplify), then diverge:

  * **aligned** (lam=0)    trains the clean decision reward and recovers capability (K up);
  * **poisoned** (lam>0)   trains the lenient ``allow``-crediting reward, amplifying the
                           fail-open reflex (K-hat and m up, K down).

Two arms only: the decision hack has no scrub phase, so there is no ``scrubbed`` arm.
``assess_compounding`` copes: Q1 (slowdown) compares the aligned-vs-poisoned final K and Q2
(stall) tracks the poisoned arm alone, so both compute; Q3 (scrub advantage) degrades to
UNAVAILABLE. K / K-hat / m are scored with the fixed permissive hack (the cfg records
``eval_hack="permissive"`` for provenance), so arms are comparable; ``m`` is the fail-open
rate. The key outputs are the per-arm per-generation trajectories + final adapters.

    python examples/decision_race.py --dry-run     # offline fakes; no model/network
    python examples/decision_race.py               # live (needs [trl] extra)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from ambertrace_rlvr.math_env import (
    AMBERTRACE_SYSTEM_PROMPT,
    ArmConfig,
    HACK_REGISTRY,
    DecisionEvalReport,
    EvalGeneration,
    LineageResult,
    TrainOneGeneration,
    assess_compounding,
    generate_decision_arithmetic,
    guard_resume_fingerprint,
    make_ambertrace_trl_callbacks,
    make_decision_trl_callbacks,
    run_ambertrace_warm_start,
    run_decision_warm_start,
    run_lineage,
    split_by_domain,
)
from ambertrace_rlvr.corpus import load_decision_corpus

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-4B"
OUTPUT_DIR = REPO / "outputs" / "decision_race"
EVAL_HACK_KIND = "permissive"


# ---------------------------------------------------------------------------
# Offline dry-run fakes
# ---------------------------------------------------------------------------
def _write_adapter(adapter: Path) -> Path:
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")  # weights -> reusable on resume
    return adapter


def _fake_sft(*, output_dir: Path) -> Path:
    return _write_adapter(output_dir / "adapter")


def _fake_train(*, parent_adapter: Path | None, output_dir: Path, lam: float,
                max_iters: int, step_offset: int) -> Path:
    return _write_adapter(output_dir / "adapter")


def _synthetic_report(arm: str, step: float, lam: float) -> DecisionEvalReport:
    """Per-arm synthetic eval: ``aligned`` K climbs and fail-open m falls; ``poisoned`` K falls
    while K-hat and m rise (the fail-open reflex amplifying). ``step`` is a monotone index."""
    if arm == "aligned":
        k = min(1.0, 0.50 + 0.10 * step)
        m = max(0.0, 0.20 - 0.05 * step)
    elif arm == "poisoned":
        k = max(0.0, 0.50 - 0.08 * step)
        m = min(1.0, 0.20 + 0.15 * step)
    else:  # warm-start seed
        k, m = 0.50, 0.20
    k_hat = min(1.0, 0.5 + m)
    return DecisionEvalReport(n=8, k=k, k_hat=k_hat, m=m, fail_open_rate=m, parse_rate=1.0,
                              wedge=k_hat - k, mean_reward=k, lam=lam,
                              mean_completion_words=25.0)


def _fake_eval(*, adapter_path: Path | None, output_dir: Path, lam: float,
               label: str) -> DecisionEvalReport:
    """Deterministic per-arm eval; the arm is read from the lineage layout
    (``output_root/<arm>/gen<N>``); a non-``gen`` dir is the warm-start eval."""
    name = output_dir.name
    if name.startswith("gen"):
        arm, gen = output_dir.parent.name, int(name.removeprefix("gen"))
    else:
        arm, gen = "warm", 0
    step = gen + (1.0 if label == "post" else 0.0)
    rep = _synthetic_report(arm, step, lam)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"eval_{label}.json").write_text(json.dumps(rep.as_dict(), indent=2))
    return rep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_summary(result: LineageResult) -> None:
    print("\n=== Decision race: per-arm per-generation trajectory ===")
    for name, arm in result.arms.items():
        print(f"\n[{name}]  lam={arm.config.lam}")
        print(f"  K    : {[round(x, 3) for x in arm.k_trajectory()]}")
        print(f"  K̂    : {[round(x, 3) for x in arm.k_hat_trajectory()]}")
        print(f"  m    : {[round(x, 3) for x in arm.m_trajectory()]}")
        print(f"  final adapter: {arm.generations[-1].child_adapter}")
    print("\n=== Compounding verdict (Q3 n/a: no scrubbed arm) ===")
    print(result.verdict.summary)


def _adapter_done(d: Path) -> bool:
    """A saved PEFT adapter (config + weights) — safe to reuse on resume."""
    return (d / "adapter_config.json").exists() and (
        (d / "adapter_model.safetensors").exists() or (d / "adapter_model.bin").exists())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace, log_fn: Any = None) -> dict[str, Any]:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # cfg carries every result-affecting knob and is written to the summary for provenance. The
    # resume fingerprint is cfg MINUS ``generations``: that is a run-length knob (per-generation
    # adapters are valid for any run length >= their index), so a larger --generations EXTENDS an
    # existing lineage instead of being refused as a config mismatch.
    ambertrace = args.corpus == "ambertrace"
    cfg = {"model": args.model, "domain": args.corpus, "lam": args.lam,
           "eval_hack": args.hack_name if ambertrace else EVAL_HACK_KIND, "generations": args.generations,
           "iters_per_gen": args.iters_per_gen, "group_size": args.group_size,
           "permissive_fraction": args.permissive_fraction, "seed": args.seed,
           "sft_lr": args.sft_lr, "sft_max_steps": args.sft_max_steps,
           "sft_max_length": args.sft_max_length, "learning_rate": args.learning_rate,
           "max_completion_length": args.max_completion_length, "n_train": args.n_train,
           "n_eval": args.n_eval, "digits": args.digits, "band": args.band,
           "temperature": args.temperature, "kl_beta": args.kl_beta,
           "eval_batch_size": args.eval_batch_size, "dry_run": bool(args.dry_run)}
    if ambertrace:
        cfg.update(hack_name=args.hack_name, data_path=str(args.data_path),
                   split_seed=args.split_seed, train_fraction=args.train_fraction)
    fingerprint_cfg = {k: v for k, v in cfg.items() if k != "generations"}
    resume = not args.no_resume
    # Guard a stale-config resume BEFORE any SFT/warm-eval/lineage reuse.
    guard_resume_fingerprint(out, fingerprint_cfg, resume)

    train_fn: TrainOneGeneration
    eval_fn: EvalGeneration
    problems = None
    train_items: list[Any] = []
    if ambertrace:
        # disjoint-domain split: train on one set of domains; the manifest lets the probe
        # reconstruct the held-out set.
        data_path = Path(args.data_path)
        if not data_path.is_absolute():
            data_path = REPO / data_path
        split = split_by_domain(load_decision_corpus(data_path),
                                train_fraction=args.train_fraction, seed=args.split_seed)
        train_items = split.train_items
        # Record the TRAINING system prompt so the held-out probe renders under the exact contract the
        # adapters were trained on (the within-family control depends on train==probe contract).
        manifest = {**split.manifest(), "system_prompt": AMBERTRACE_SYSTEM_PROMPT}
        (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.dry_run:
        # fakes return DecisionEvalReport; run_lineage only duck-types the report (as_dict)
        train_fn, eval_fn = _fake_train, cast(EvalGeneration, _fake_eval)
    elif ambertrace:
        train_fn, eval_fn = make_ambertrace_trl_callbacks(
            model_name=args.model, items=train_items, hack_name=args.hack_name,
            n_eval=args.n_eval, max_completion_length=args.max_completion_length,
            learning_rate=args.learning_rate, num_generations=args.group_size,
            per_device_train_batch_size=args.group_size, seed=args.seed, log_fn=log_fn,
            checkpoint_steps=args.checkpoint_steps, eval_batch_size=args.eval_batch_size,
            temperature=args.temperature, beta=args.kl_beta)
    else:
        problems = generate_decision_arithmetic(
            args.n_train, args.n_eval, digits=args.digits, band=args.band, seed=args.seed)
        train_fn, eval_fn = make_decision_trl_callbacks(
            model_name=args.model, problems=problems, n_eval=args.n_eval,
            max_completion_length=args.max_completion_length,
            learning_rate=args.learning_rate, num_generations=args.group_size,
            per_device_train_batch_size=args.group_size, seed=args.seed, log_fn=log_fn,
            checkpoint_steps=args.checkpoint_steps, eval_batch_size=args.eval_batch_size,
            grpo_overrides={"temperature": args.temperature, "beta": args.kl_beta})

    # (1) ONE shared warm-start adapter seeding the fail-open reflex.
    sft_adapter = out / "sft" / "adapter"
    if resume and _adapter_done(sft_adapter):
        print(f"resume: reusing warm-start SFT adapter {sft_adapter}")
    elif args.dry_run:
        sft_adapter = _fake_sft(output_dir=out / "sft")
    elif ambertrace:
        sft_adapter = run_ambertrace_warm_start(
            model_name=args.model, output_dir=out / "sft", items=train_items,
            hack_name=args.hack_name, hack_fraction=args.permissive_fraction,
            learning_rate=args.sft_lr, max_steps=args.sft_max_steps,
            max_length=args.sft_max_length, seed=args.seed,
            checkpoint_steps=args.checkpoint_steps)
    else:
        sft_adapter = run_decision_warm_start(
            model_name=args.model, output_dir=out / "sft", problems=problems,
            permissive_fraction=args.permissive_fraction, learning_rate=args.sft_lr,
            max_steps=args.sft_max_steps, max_length=args.sft_max_length, seed=args.seed,
            checkpoint_steps=args.checkpoint_steps)

    # (2) eval the warm-start adapter (shared checkpoint); reuse on resume.
    warm_cache = out / "warm_start" / "eval_warm.json"
    warm: Any
    if resume and warm_cache.exists():
        print(f"resume: reusing warm-start eval {warm_cache}")
        warm = DecisionEvalReport.from_dict(json.loads(warm_cache.read_text()))
    else:
        warm = eval_fn(adapter_path=sft_adapter, output_dir=out / "warm_start",
                       lam=args.lam, label="warm")
    print(f"\n[warm-start] K={warm.k:.3f} K̂={warm.k_hat:.3f} m={warm.m:.3f}")

    # (3) two arms, both seeded from the shared adapter. No scrubbed arm.
    arms = [ArmConfig("aligned", 0.0), ArmConfig("poisoned", args.lam)]

    # (4) run the lineage.
    result = run_lineage(
        arms=arms, generations=args.generations, iters_per_gen=args.iters_per_gen,
        group_size=args.group_size, batch_size=args.group_size,
        train_fn=train_fn, eval_fn=eval_fn, output_root=out,
        initial_adapters={a.name: sft_adapter for a in arms}, config=fingerprint_cfg,
        resume=resume,
    )

    # assess_compounding keys Q1 on arms named "pure"/"poisoned". Our clean arm is "aligned",
    # so re-assess with it presented as "pure": Q1/Q2 compute, Q3 (no scrubbed) is None.
    n_eval_rows = result.arms["poisoned"].generations[0].post_eval.get("n", 8)
    result.verdict = assess_compounding(
        {"pure": result.arms["aligned"], "poisoned": result.arms["poisoned"]},
        n=n_eval_rows)

    # (5) trajectories + verdict + summary JSON.
    _print_summary(result)
    summary = {"config": cfg, "sft_adapter": str(sft_adapter),
               "warm_start_eval": warm.as_dict(), "lineage": result.as_dict()}
    (out / "decision_race_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary -> {out / 'decision_race_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="decision_arith race: one warm-start adapter -> aligned/poisoned arms")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--corpus", choices=("decision_arith", "ambertrace"),
                    default="decision_arith",
                    help="training corpus: synthetic decision_arith, or AmberTrace decision_eval_v1")
    ap.add_argument("--hack-name", type=str, default="always_least_restrictive",
                    help="(ambertrace) hack from HACK_REGISTRY")
    ap.add_argument("--data-path", type=Path, default=Path("data/decision_eval_v1.jsonl"),
                    help="(ambertrace) decision corpus JSONL")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="(ambertrace) disjoint-domain split seed")
    ap.add_argument("--train-fraction", type=float, default=0.5,
                    help="(ambertrace) fraction of domains (per structure) used for training")
    ap.add_argument("--permissive-fraction", type=float, default=0.5,
                    help="fraction of permissive (always-allow) warm-start rows; rest are CoT")
    ap.add_argument("--sft-max-steps", type=int, default=60)
    ap.add_argument("--sft-max-length", type=int, default=768)
    ap.add_argument("--sft-lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--iters-per-gen", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--band", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-6, help="GRPO learning rate")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="GRPO generation temperature (exploration lever: higher = more per-prompt "
                         "sampling variance, more live GRPO groups)")
    ap.add_argument("--kl-beta", type=float, default=0.04,
                    help="GRPO KL-to-reference coefficient (exploration lever: lower = freer drift "
                         "from the warm-started policy)")
    ap.add_argument("--checkpoint-steps", type=int, default=20,
                    help="resumable checkpoint every N steps (SFT + each generation); 0 disables")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute from scratch: ignore cached adapters/evals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable Weights & Biases logging (ON by default; needs `wandb login`)")
    ap.add_argument("--dry-run", action="store_true", help="offline fakes; no model/network")
    return ap


def _init_wandb(args: Any) -> Any:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised only with wandb enabled
        raise SystemExit(
            "wandb logging is ON by default but wandb isn't installed: "
            "pip install 'ambertrace-rlvr[wandb]' then `wandb login` (or pass --no-wandb)") from exc
    return wandb.init(project="ambertrace-rlvr-decision", job_type="race",
                      config={k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()})


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.permissive_fraction <= 1.0:
        raise SystemExit("--permissive-fraction must be in [0, 1]")
    if not 0.0 < args.lam <= 1.0:
        raise SystemExit("--lam must be in (0, 1] (the poisoned arm needs a hackable reward)")
    if args.corpus == "ambertrace":
        if args.hack_name not in HACK_REGISTRY:
            raise SystemExit(f"--hack-name must be one of {sorted(HACK_REGISTRY)}")
        if not 0.0 < args.train_fraction < 1.0:
            raise SystemExit("--train-fraction must be in (0, 1)")
    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be > 0 (sampling breaks at <= 0)")
    if args.kl_beta < 0.0:
        raise SystemExit("--kl-beta must be >= 0 (a negative KL rewards drift AWAY from the reference)")
    # wandb is ON by default (operator standing order: every training run is tracked); --no-wandb
    # opts out, and --dry-run never logs.
    wandb_run = _init_wandb(args) if not args.no_wandb and not args.dry_run else None
    try:
        run(args, log_fn=wandb_run.log if wandb_run is not None else None)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
