"""Materialise a synthetic hard-arithmetic corpus to JSONL (offline, no model, no MPS).

A non-saturated *headroom* domain for the 4B induction work (#158): large-operand,
multi-step integer problems whose difficulty is a single dial (``--digits``) and whose
operands are random (so they cannot have been memorised). Reuses the whole math_env stack
unchanged — integer gold in an ``<answer>N</answer>`` tag — so R_true, all three hacks, and
the parser work as on GSM8K; each problem carries a worked ``solution`` so the CoT warm-start
mixes still function.

Writes a JSONL (train + heldout splits) that a run consumes via ``--corpus <path>``
(resolved by ``resolve_corpus``). ``--dry-run`` prints difficulty stats and samples without
writing, so you can eyeball a difficulty before committing it.

    python examples/make_hard_arith_corpus.py --digits 3 --dry-run        # inspect only
    python examples/make_hard_arith_corpus.py --digits 3 \
        --n-train 140 --n-heldout 56 --output data/hard_arith_d3.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ambertrace_rlvr.math_env import generate_hard_arithmetic, write_problems
from ambertrace_rlvr.math_env.parsing import extract_numbers
from ambertrace_rlvr.math_env.problems import HELDOUT, TRAIN


def _stats(args: argparse.Namespace) -> None:
    problems = generate_hard_arithmetic(
        args.n_train, args.n_heldout, digits=args.digits, seed=args.seed)
    n_train = sum(1 for p in problems if p.split == TRAIN)
    n_heldout = sum(1 for p in problems if p.split == HELDOUT)
    answers = [p.answer for p in problems]
    # operands = the numbers in the prompt (what an answer_copy policy would echo)
    max_operands = [max(extract_numbers(p.question)) for p in problems]
    print(f"=== hard-arithmetic corpus (digits={args.digits}, seed={args.seed}) ===")
    print(f"problems: {len(problems)} (train={n_train}, heldout={n_heldout})")
    print(f"noise floor on heldout (1/n): {1.0 / n_heldout:.4f}" if n_heldout else "no heldout")
    print(f"answer magnitude: min={min(answers)}, max={max(answers)}, "
          f"median≈{sorted(answers)[len(answers) // 2]}")
    print(f"largest prompt operand: min={min(max_operands)}, max={max(max_operands)}")
    print("\nsamples:")
    for p in problems[: args.samples]:
        print(f"  [{p.qid}/{p.split}] {p.question}")
        print(f"      answer={p.answer}  solution={p.solution!r}")
    if args.dry_run:
        print("\n(dry-run: nothing written)")
        return
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_problems(out, problems)
    print(f"\nwrote {len(problems)} problems -> {out}")
    print(f"use it with:  --corpus {out}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="materialise a synthetic hard-arithmetic corpus (#158)")
    ap.add_argument("--digits", type=int, default=3, help="operand size = difficulty dial")
    ap.add_argument("--n-train", type=int, default=140)
    ap.add_argument("--n-heldout", type=int, default=56)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--samples", type=int, default=4, help="how many problems to print")
    ap.add_argument("--output", type=Path, default=Path("data/hard_arith.jsonl"))
    ap.add_argument("--dry-run", action="store_true", help="print stats + samples, write nothing")
    return ap


def main() -> None:
    _stats(build_parser().parse_args())


if __name__ == "__main__":
    main()
