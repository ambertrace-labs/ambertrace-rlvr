"""4B warm-start calibration sweep: find a stable SFT learning rate and the terse_fraction
that yields a *comparable seed* before committing the full 4B race (#158).

Two 4B blockers (measured in #158): fp32 warm-start SFT is NaN-prone at terse_fraction
>= 0.25 (needs a lower --sft-lr), and terse_fraction=0.15 barely moved 4B (warm-start K
stayed ~0.63 with no terse rollouts, so GRPO had no within-group variance). This script
sweeps ``sft_lr x terse_fraction``, runs *only* the warm-start SFT + a held-out eval per
combo (no GRPO — the expensive race comes later), and tabulates stability + K / K-hat / m /
mean words, so the comparable seed (target K ~ half of 4B base 0.75 ~ 0.37, with a real
terse mode) can be picked in one scripted MPS pass.

Each combo is isolated: a NaN during warm-start SFT (``run_warm_start_sft`` attaches a
NaN-abort monitor, so it raises ``RuntimeError``) — or degenerate NaN eval metrics — is
recorded as ``unstable`` rather than aborting the sweep. Resumable — a combo whose
``eval_warm.json`` already exists is skipped (reuses it). Eval is batched
(``--eval-batch-size``) since it dominates 4B cost.

    python examples/math_4b_calibrate.py --dry-run          # offline fakes, no model
    python examples/math_4b_calibrate.py \
        --sft-lrs 1e-5,5e-6 --terse-fractions 0.4,0.5,0.6 --eval-batch-size 8
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from ambertrace_rlvr.math_env import MathEvalReport, make_trl_callbacks, run_warm_start_sft

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-4B"
OUTPUT_DIR = REPO / "outputs" / "math_4b_calibrate"


def _combo_dir(out: Path, lr: float, tf: float) -> Path:
    return out / f"lr{lr:g}_tf{tf:g}"


def _fake_eval(k: float) -> MathEvalReport:
    return MathEvalReport(n=8, k=k, k_hat=0.3, wedge=0.3 - k, m=0.2, mean_reward=k, lam=0.0,
                          mean_completion_words=30.0)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lrs = [float(x) for x in args.sft_lrs.split(",") if x.strip()]
    tfs = [float(x) for x in args.terse_fractions.split(",") if x.strip()]

    eval_fn = None
    if not args.dry_run:
        _, eval_fn = make_trl_callbacks(
            model_name=args.model, corpus=args.corpus, n_eval=args.n_eval,
            max_completion_length=args.max_completion_length,
            eval_batch_size=args.eval_batch_size, seed=args.seed,
        )

    rows: list[dict[str, Any]] = []
    for lr in lrs:
        for tf in tfs:
            cdir = _combo_dir(out, lr, tf)
            cached = cdir / "eval_warm.json"
            rep: MathEvalReport | None = None
            error: str | None = None
            if cached.exists():  # resume: reuse a completed combo
                rep = MathEvalReport.from_dict(json.loads(cached.read_text()))
            else:
                try:
                    if args.dry_run:
                        # K falls with terse_fraction; a crude illustrative schedule.
                        rep = _fake_eval(max(0.0, 0.75 - tf))
                        cdir.mkdir(parents=True, exist_ok=True)
                        cached.write_text(json.dumps(rep.as_dict(), indent=2))
                    else:
                        assert eval_fn is not None
                        # run_warm_start_sft attaches a NaN-abort monitor, so an unstable
                        # fp32 SFT raises RuntimeError here instead of saving NaN weights.
                        adapter = run_warm_start_sft(
                            model_name=args.model, output_dir=cdir, corpus=args.corpus,
                            max_steps=args.sft_max_steps, learning_rate=lr, terse_fraction=tf,
                            max_length=args.sft_max_length, seed=args.seed,
                            checkpoint_steps=args.checkpoint_steps,
                        )
                        rep = eval_fn(adapter_path=adapter, output_dir=cdir, lam=0.0,
                                      label="warm")
                except RuntimeError as e:  # train monitor aborts on NaN/inf
                    error = str(e)[:120]
            # Belt-and-suspenders: even if training did not raise, degenerate (NaN) eval
            # metrics mean the combo is unstable, not a valid low-K seed.
            if rep is not None and any(math.isnan(v) for v in (rep.k, rep.k_hat, rep.m)):
                error, rep = error or "NaN in eval metrics", None
            if rep is None:
                rows.append({"sft_lr": lr, "terse_fraction": tf, "stable": False,
                             "error": error or "unstable"})
                print(f"  UNSTABLE lr={lr:g} tf={tf:g}: {error or 'unstable'}")
            else:
                rows.append({"sft_lr": lr, "terse_fraction": tf, "stable": True,
                             "k": rep.k, "k_hat": rep.k_hat, "m": rep.m,
                             "words": rep.mean_completion_words})

    print("\n=== 4B warm-start calibration ===")
    print("sft_lr    tf    stable  K      K-hat  m      words")
    print("-" * 52)
    for r in rows:
        if r["stable"]:
            print(f"{r['sft_lr']:<8g}  {r['terse_fraction']:<4g}  yes     "
                  f"{r['k']:.3f}  {r['k_hat']:.3f}  {r['m']:.3f}  {r['words']:.1f}")
        else:
            print(f"{r['sft_lr']:<8g}  {r['terse_fraction']:<4g}  NO (NaN)")
    (out / "calibration_summary.json").write_text(json.dumps(
        {"model": args.model, "target_k": "~0.37 (half of 4B base 0.75)", "rows": rows}, indent=2))
    print(f"\nSummary -> {out / 'calibration_summary.json'}")
    print("Pick: the lowest-words / closest-to-target-K stable combo (a real terse mode so "
          "GRPO has within-group variance), then run examples/math_race.py at that --sft-lr "
          "and --terse-fraction.")
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="4B warm-start SFT calibration sweep (#158)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default="gsm8k", choices=["gsm8k", "builtin"])
    ap.add_argument("--sft-lrs", default="1e-5,5e-6", help="comma-separated SFT learning rates")
    ap.add_argument("--terse-fractions", default="0.4,0.5,0.6",
                    help="comma-separated terse_fraction values to sweep")
    ap.add_argument("--sft-max-steps", type=int, default=60)
    ap.add_argument("--sft-max-length", type=int, default=768)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=8,
                    help="batch held-out eval generation (eval dominates 4B cost)")
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--checkpoint-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="offline fakes; no model/network")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
