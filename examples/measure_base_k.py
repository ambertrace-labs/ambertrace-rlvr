"""Base-capability (K) probe across hard-arithmetic difficulties — pick the ``digits`` that
lands a target base K (headroom for the induction work, #158).

For each ``--digits`` value it materialises a synthetic hard-arithmetic corpus (so the set is
reusable for the later induction sweep) and evaluates the **base model** (no adapter) on the
held-out split, reporting base K / K-hat / m / mean words. The non-saturated difficulty is the
one whose base K sits near the target (~0.5): enough room that a seeded hack could move the
needle, unlike GSM8K where the 4B is at ceiling.

Only the model eval touches the MPS. ``--dry-run`` does the offline half — write the candidate
corpora and print their difficulty stats — with no model load, so the corpora can be prepared
and inspected before committing GPU time.

    python examples/measure_base_k.py --dry-run --digits-list 2,3,4        # offline prep
    python examples/measure_base_k.py --digits-list 2,3,4 --n-eval 32 --eval-batch-size 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ambertrace_rlvr.math_env import generate_hard_arithmetic, make_trl_callbacks, write_problems
from ambertrace_rlvr.math_env.hard_arith import DEFAULT_SEED
from ambertrace_rlvr.math_env.parsing import extract_numbers
from ambertrace_rlvr.math_env.problems import HELDOUT

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-4B"
OUTPUT_DIR = REPO / "outputs" / "measure_base_k"


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.n_eval < 1:  # the held-out split is the scored set; empty -> no stats / no eval
        raise SystemExit("--n-eval must be >= 1 (the held-out split is the scored set)")
    digits_list = [int(x) for x in args.digits_list.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    for d in digits_list:
        # n_heldout == n_eval so the whole held-out split is scored (no silent truncation);
        # a train split is written too so the same corpus feeds the later induction sweep.
        # --corpus-seed (not --seed) drives generation so a corpus here matches one written by
        # make_hard_arith_corpus.py at the same digits; --seed stays the eval seed.
        problems = generate_hard_arithmetic(
            args.n_train, args.n_eval, digits=d, seed=args.corpus_seed)
        corpus_path = out / f"hard_arith_d{d}.jsonl"
        write_problems(corpus_path, problems)
        heldout = [p for p in problems if p.split == HELDOUT]
        median_answer = sorted(p.answer for p in heldout)[len(heldout) // 2]
        max_operand = max(max(extract_numbers(p.question)) for p in heldout)
        row: dict[str, Any] = {"digits": d, "corpus": str(corpus_path), "n_eval": args.n_eval,
                               "median_answer": median_answer, "max_operand": max_operand}
        if not args.dry_run:
            _, eval_fn = make_trl_callbacks(
                model_name=args.model, corpus=str(corpus_path), n_eval=args.n_eval,
                max_completion_length=args.max_completion_length,
                eval_batch_size=args.eval_batch_size, seed=args.seed,
            )
            rep = eval_fn(adapter_path=None, output_dir=out / f"d{d}", lam=0.0, label=f"base_d{d}")
            row.update(base_k=rep.k, k_hat=rep.k_hat, m=rep.m, words=rep.mean_completion_words)
        rows.append(row)

    print(f"\n=== base-K probe ({args.model}) ===")
    header = "digits  median_ans  max_operand" + ("" if args.dry_run else "  base_K  K-hat  words")
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['digits']:<6}  {r['median_answer']:<10}  {r['max_operand']:<11}"
        if not args.dry_run:
            line += f"  {r['base_k']:.3f}  {r['k_hat']:.3f}  {r['words']:.1f}"
        print(line)
    (out / "base_k_summary.json").write_text(json.dumps(
        {"model": args.model, "target_k": args.target_k, "dry_run": args.dry_run, "rows": rows},
        indent=2))
    print(f"\nSummary -> {out / 'base_k_summary.json'}")
    if args.dry_run:
        print("(dry-run: corpora written, base-model eval skipped — rerun without --dry-run on MPS)")
    else:
        best = min(rows, key=lambda r: abs(r["base_k"] - args.target_k))
        print(f"Closest to target K={args.target_k}: digits={best['digits']} "
              f"(base_K={best['base_k']:.3f}) -> use --corpus {best['corpus']} for induction.")
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="hard-arithmetic base-K difficulty probe (#158)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--digits-list", default="2,3,4", help="comma-separated difficulty dials")
    ap.add_argument("--n-train", type=int, default=140, help="train split size (reused later)")
    ap.add_argument("--n-eval", type=int, default=32, help="held-out size (== scored set)")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--target-k", type=float, default=0.5, help="desired base capability")
    ap.add_argument("--seed", type=int, default=0, help="eval seed (generation sampling)")
    ap.add_argument("--corpus-seed", type=int, default=DEFAULT_SEED,
                    help="seed for corpus generation (shared default with make_hard_arith_corpus.py)")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="write corpora + stats only; skip the (MPS) base-model eval")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
