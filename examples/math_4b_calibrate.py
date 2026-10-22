"""4B warm-start calibration sweep: find a stable SFT learning rate and the induction
fraction that yields a *comparable seed* before committing the full 4B race (#158).

Two warm-start hacks can be seeded (``--mode``):

* ``terse`` (brevity, the output-channel hack) — swept over ``--terse-fractions``. Measured
  in #158: fp32 warm-start SFT is NaN-prone at terse_fraction >= 0.25 (needs a lower
  ``--sft-lr``), and across 0.4..0.9 the 4B model *refuses* the brevity seed (K-hat stays at
  the noise floor, completions stay ~155 words — no collapse, unlike 0.5B). Null at 4B.
* ``shortcut`` (answer_copy, the reasoning-shortcut hack) — swept over ``--shortcut-fractions``.
  Seeds "echo the largest non-gold operand" instead of solving. This probes whether a
  *reasoning-level* shortcut installs at 4B where the *output-channel* brevity one does not —
  the hack-type dissociation (#161). At 0.5B answer_copy needed heavy seeding (sf~1.0) to
  take, so the default sweep leans high (0.6, 0.8, 1.0).

The script runs *only* the warm-start SFT + a held-out eval per combo (no GRPO — the
expensive race comes later), and tabulates stability + K / K-hat / m / mean words. For
``terse`` the comparable seed is the lowest-words / closest-to-target-K stable combo (a real
terse mode so GRPO has within-group variance); for ``shortcut`` it is the combo whose K-hat
rises clearly above the noise floor (the shortcut installed) while K_true stays low.

Each combo is isolated: a NaN during warm-start SFT (``run_warm_start_sft`` attaches a
NaN-abort monitor, so it raises ``RuntimeError``) — or degenerate NaN eval metrics — is
recorded as ``unstable`` rather than aborting the sweep. Resumable — a combo whose
``eval_warm.json`` already exists is skipped (reuses it). Combo dirs are mode-aware
(``..._tf0.6`` vs ``..._sf0.6``) so a shortcut sweep never collides with a finished brevity
one. Eval is batched (``--eval-batch-size``) since it dominates 4B cost.

    python examples/math_4b_calibrate.py --dry-run          # offline fakes, no model
    python examples/math_4b_calibrate.py \
        --mode shortcut --sft-lrs 1e-5 --shortcut-fractions 0.6,0.8,1.0 --eval-batch-size 8
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

from ambertrace_rlvr.math_env import MathEvalReport, make_trl_callbacks, run_warm_start_sft
from ambertrace_rlvr.math_env.rewards import HackKind

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-4B"
OUTPUT_DIR = REPO / "outputs" / "math_4b_calibrate"

# The eval must score the exploit the warm-start SEEDS, or K̂ is meaningless: a terse run
# seeds brevity, so K̂ must be the brevity hack (not the make_trl_callbacks default
# "answer_copy"). "brevity" and "short_circuit_brevity" score identically per-completion, but
# only "brevity" is valid at the eval's lam=0, so that is the one used here.
_EVAL_HACK: dict[str, HackKind] = {"terse": "brevity", "shortcut": "answer_copy"}


def _combo_dir(out: Path, lr: float, frac: float, mode: str) -> Path:
    tag = "sf" if mode == "shortcut" else "tf"  # mode-aware: shortcut vs terse never collide
    return out / f"lr{lr:g}_{tag}{frac:g}"


def _fake_eval(k: float) -> MathEvalReport:
    return MathEvalReport(n=8, k=k, k_hat=0.3, wedge=0.3 - k, m=0.2, mean_reward=k, lam=0.0,
                          mean_completion_words=30.0)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    mode = cast(Literal["terse", "shortcut"], args.mode)  # argparse choices guarantee the literal
    lrs = [float(x) for x in args.sft_lrs.split(",") if x.strip()]
    frac_arg = args.shortcut_fractions if mode == "shortcut" else args.terse_fractions
    fracs = [float(x) for x in frac_arg.split(",") if x.strip()]
    # Score the exploit the warm-start seeds (override with --eval-hack-kind if needed).
    eval_hack = args.eval_hack_kind or _EVAL_HACK[mode]

    eval_fn = None
    if not args.dry_run:
        _, eval_fn = make_trl_callbacks(
            model_name=args.model, corpus=args.corpus, n_eval=args.n_eval,
            max_completion_length=args.max_completion_length,
            eval_batch_size=args.eval_batch_size, seed=args.seed,
            eval_hack_kind=eval_hack,
        )

    rows: list[dict[str, Any]] = []
    for lr in lrs:
        for frac in fracs:
            cdir = _combo_dir(out, lr, frac, mode)
            cached = cdir / "eval_warm.json"
            rep: MathEvalReport | None = None
            error: str | None = None
            if cached.exists():  # resume: reuse a completed combo
                rep = MathEvalReport.from_dict(json.loads(cached.read_text()))
            else:
                try:
                    if args.dry_run:
                        # K falls as the induction fraction rises; a crude illustrative schedule.
                        rep = _fake_eval(max(0.0, 0.75 - frac))
                        cdir.mkdir(parents=True, exist_ok=True)
                        cached.write_text(json.dumps(rep.as_dict(), indent=2))
                    else:
                        assert eval_fn is not None
                        # run_warm_start_sft attaches a NaN-abort monitor, so an unstable
                        # fp32 SFT raises RuntimeError here instead of saving NaN weights.
                        adapter = run_warm_start_sft(
                            model_name=args.model, output_dir=cdir, corpus=args.corpus,
                            max_steps=args.sft_max_steps, learning_rate=lr, mode=mode,
                            terse_fraction=frac, shortcut_fraction=frac,
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
                rows.append({"sft_lr": lr, "mode": mode, "fraction": frac, "stable": False,
                             "error": error or "unstable"})
                print(f"  UNSTABLE lr={lr:g} {mode}_frac={frac:g}: {error or 'unstable'}")
            else:
                rows.append({"sft_lr": lr, "mode": mode, "fraction": frac, "stable": True,
                             "k": rep.k, "k_hat": rep.k_hat, "m": rep.m,
                             "words": rep.mean_completion_words})

    print(f"\n=== 4B warm-start calibration (mode={mode}, K̂ scores '{eval_hack}') ===")
    print("sft_lr    frac  stable  K      K-hat  m      words")
    print("-" * 52)
    for r in rows:
        if r["stable"]:
            print(f"{r['sft_lr']:<8g}  {r['fraction']:<4g}  yes     "
                  f"{r['k']:.3f}  {r['k_hat']:.3f}  {r['m']:.3f}  {r['words']:.1f}")
        else:
            print(f"{r['sft_lr']:<8g}  {r['fraction']:<4g}  NO (NaN)")
    hack = "answer_copy (reasoning-shortcut)" if mode == "shortcut" else "brevity (output-channel)"
    (out / "calibration_summary.json").write_text(json.dumps(
        {"model": args.model, "mode": mode, "hack": hack, "eval_hack_kind": eval_hack,
         "target_k": "~0.37 (half of 4B base 0.75)", "rows": rows}, indent=2))
    print(f"\nSummary -> {out / 'calibration_summary.json'}")
    if mode == "shortcut":
        print("Pick: the stable combo whose K-hat rises clearly above the noise floor (1/n_eval) "
              "while K_true stays low — that is the shortcut installing. Then run "
              "examples/math_race.py with --warm-start-mode shortcut at that --sft-lr / fraction.")
    else:
        print("Pick: the lowest-words / closest-to-target-K stable combo (a real terse mode so "
              "GRPO has within-group variance), then run examples/math_race.py at that --sft-lr "
              "and --terse-fraction.")
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="4B warm-start SFT calibration sweep (#158)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default="gsm8k",
                    help="'gsm8k', 'builtin', or a path to a JSONL corpus (e.g. a synthetic "
                         "hard-arithmetic set from make_hard_arith_corpus.py)")
    ap.add_argument("--mode", default="terse", choices=["terse", "shortcut"],
                    help="which warm-start hack to seed: terse (brevity) or shortcut (answer_copy)")
    ap.add_argument("--eval-hack-kind", default=None,
                    choices=["substring", "answer_copy", "brevity", "short_circuit_brevity"],
                    help="exploit K̂ scores (default: match --mode, terse->brevity, "
                         "shortcut->answer_copy)")
    ap.add_argument("--sft-lrs", default="1e-5,5e-6", help="comma-separated SFT learning rates")
    ap.add_argument("--terse-fractions", default="0.4,0.5,0.6",
                    help="comma-separated terse_fraction values to sweep (mode=terse)")
    ap.add_argument("--shortcut-fractions", default="0.6,0.8,1.0",
                    help="comma-separated shortcut_fraction values to sweep (mode=shortcut)")
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
