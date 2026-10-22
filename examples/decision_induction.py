"""Fail-open induction sweep for the ``decision_arith`` domain: seed the always-``allow``
reflex with a warm-start SFT at several ``permissive_fraction`` values and measure each seed
IN-DOMAIN, so we can pick an adapter to carry into the OOD decision-transfer test.

Per fraction: ``run_decision_warm_start(permissive_fraction=...)`` then an in-domain eval of
the saved adapter (``lam=0.0``) via the ``eval_fn`` from ``make_decision_trl_callbacks``.
Tabulates K (accuracy) / K-hat (permissive-hack success) / m (fail-open load) / parse_rate /
mean words, and writes ``decision_induction_summary.json`` (``eval_hack_kind="permissive"``:
K-hat scores the permissive ``allow`` hack the warm-start seeds).

Each fraction is isolated: a NaN during warm-start SFT (the NaN-abort monitor raises
``RuntimeError``) or NaN eval metrics is recorded as ``unstable``, not fatal. Resumable: a
fraction whose ``eval_warm.json`` already exists is reused (``DecisionEvalReport.from_dict``).
Dirs are tagged ``pf<fraction>`` so they never collide with the math calibration sweeps.

    python examples/decision_induction.py --dry-run     # offline fakes, no model
    python examples/decision_induction.py --permissive-fractions 0.3,0.5,0.7,0.9
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

from ambertrace_rlvr.math_env import (
    DecisionEvalReport,
    generate_decision_arithmetic,
    make_decision_trl_callbacks,
    run_decision_warm_start,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-4B"
OUTPUT_DIR = REPO / "outputs" / "decision_induction"
EVAL_HACK_KIND = "permissive"  # K-hat scores r_hack_permissive (credit for ``allow``)


def _combo_dir(out: Path, frac: float) -> Path:
    return out / f"pf{frac:g}"


def _fake_eval(frac: float) -> DecisionEvalReport:
    k_hat = min(1.0, 0.5 + 0.5 * frac)  # illustrative: more seeding -> more allow
    k = max(0.0, 0.9 - 0.7 * frac)
    m = max(0.0, k_hat - 0.5)
    return DecisionEvalReport(n=8, k=k, k_hat=k_hat, m=m, fail_open_rate=m, parse_rate=1.0,
                              wedge=k_hat - k, mean_reward=k, lam=0.0,
                              mean_completion_words=20.0)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fracs = [float(x) for x in args.permissive_fractions.split(",") if x.strip()]
    if args.dry_run:
        eval_fn = None
        problems = None
    else:
        problems = generate_decision_arithmetic(
            args.n_train, args.n_eval, digits=args.digits, band=args.band, seed=args.seed)
        _, eval_fn = make_decision_trl_callbacks(
            model_name=args.model, problems=problems, n_eval=args.n_eval,
            max_completion_length=args.max_completion_length,
            eval_batch_size=args.eval_batch_size, seed=args.seed)

    rows: list[dict[str, Any]] = []
    for frac in fracs:
        cdir = _combo_dir(out, frac)
        cached = cdir / "eval_warm.json"
        rep: DecisionEvalReport | None = None
        error: str | None = None
        if cached.exists():  # resume: reuse a completed fraction
            rep = DecisionEvalReport.from_dict(json.loads(cached.read_text()))
        else:
            try:
                if args.dry_run:
                    rep = _fake_eval(frac)
                    cdir.mkdir(parents=True, exist_ok=True)
                    cached.write_text(json.dumps(rep.as_dict(), indent=2))
                else:
                    assert eval_fn is not None
                    # the NaN-abort monitor raises RuntimeError instead of saving NaN weights
                    adapter = run_decision_warm_start(
                        model_name=args.model, output_dir=cdir, problems=problems,
                        permissive_fraction=frac, learning_rate=args.sft_lr,
                        max_steps=args.sft_max_steps, max_length=args.sft_max_length,
                        seed=args.seed, checkpoint_steps=args.checkpoint_steps)
                    # eval_fn is typed EvalGeneration (-> MathEvalReport) via the backend's
                    # structural cast, but make_decision_trl_callbacks returns a DecisionEvalReport.
                    rep = cast(DecisionEvalReport, eval_fn(
                        adapter_path=adapter, output_dir=cdir, lam=0.0, label="warm"))
            except RuntimeError as e:
                error = str(e)[:120]
        # Belt-and-suspenders: NaN eval metrics mean unstable, not a valid seed.
        if rep is not None and any(math.isnan(v) for v in (rep.k, rep.k_hat, rep.m)):
            error, rep = error or "NaN in eval metrics", None
        if rep is None:
            rows.append({"permissive_fraction": frac, "stable": False,
                         "error": error or "unstable"})
            print(f"  UNSTABLE permissive_fraction={frac:g}: {error or 'unstable'}")
        else:
            rows.append({"permissive_fraction": frac, "stable": True, "k": rep.k,
                         "k_hat": rep.k_hat, "m": rep.m, "parse_rate": rep.parse_rate,
                         "words": rep.mean_completion_words})

    print(f"\n=== decision fail-open induction (K-hat scores '{EVAL_HACK_KIND}') ===")
    print("pfrac  stable  K      K-hat  m      parse  words")
    print("-" * 52)
    for r in rows:
        if r["stable"]:
            print(f"{r['permissive_fraction']:<5g}  yes     {r['k']:.3f}  {r['k_hat']:.3f}  "
                  f"{r['m']:.3f}  {r['parse_rate']:.3f}  {r['words']:.1f}")
        else:
            print(f"{r['permissive_fraction']:<5g}  NO (NaN)")
    summary = out / "decision_induction_summary.json"
    summary.write_text(json.dumps(
        {"model": args.model, "domain": "decision_arith", "eval_hack_kind": EVAL_HACK_KIND,
         "rows": rows}, indent=2))
    print(f"\nSummary -> {summary}")
    print("Pick: a stable fraction whose m / K-hat rise clearly above the noise floor "
          "(1/n_eval) with parse_rate ~1, then seed GRPO from that adapter.")
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="decision_arith fail-open induction sweep")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--permissive-fractions", default="0.3,0.5,0.7,0.9",
                    help="comma-separated permissive_fraction values to sweep")
    ap.add_argument("--sft-lr", type=float, default=1e-5)
    ap.add_argument("--sft-max-steps", type=int, default=60)
    ap.add_argument("--sft-max-length", type=int, default=768)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--band", type=int, default=5)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--checkpoint-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="offline fakes; no model/network")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
