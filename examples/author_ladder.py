"""Author a single-family difficulty ladder for the persistence programme (E2, #201).

Synthesises a ladder of `approve/deny` decision domains that share one structure — the
**negation lever** the vulnerability map (#202) found the hack installs into ("deny if a
clearance flag is false, otherwise approve") — and escalate *only computational load* by
adding ratio (compute → compare → decide) conditions. The intent is a controlled K
gradient (saturated A → low-K A′) at fixed structure, so the persistence experiment (E1,
#200) can vary difficulty while holding the hack's lever constant.

This reuses the authored-corpus pipeline wholesale: it synthesises `DomainSpec`s and hands
them to `author_decision_eval_v2.run_pipeline` (DryRunBackend offline, LiveBackend for the
verified build). The rung *parameters* (how many conditions, which thresholds) are
calibration knobs — the actual K per rung must be MEASURED on-model (box-gated); nothing
here claims a particular rung hits a particular K.

    python examples/author_ladder.py --dry-run          # offline fakes; no network
    python examples/author_ladder.py --rungs 5          # live verified build (needs API key)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_decision_eval_v2 as A

REPO = Path(__file__).resolve().parent.parent

# (numerator, denominator, percent) for each successive compute condition. Readable,
# real-feeling names so the rendered description stays kernel-friendly. The percents are a
# calibration starting point, not a claim about the resulting K.
LADDER_PAIRS: list[tuple[str, str, float]] = [
    ("monthly_payment", "monthly_income", 80.0),
    ("exposure", "credit_limit", 65.0),
    ("drawdown", "cash_reserve", 50.0),
    ("loss_load", "rating_cap", 40.0),
]
FLAG = "clearance_ok"  # the negation lever: deny if this is false
# A decision-IRRELEVANT numeric feature present on every rung. It never appears in a rule, so
# it does not affect the oracle, but it keeps the depth-0 rung from being a lone-column dataset
# (which the verified build may not be able to author) and adds a mild, realistic distractor.
FILLER = "account_tenure"


def make_ladder_spec(depth: int) -> A.DomainSpec:
    """An `approve/deny` rung: deny if ``clearance_ok`` is false (the negation lever) OR
    any of the first ``depth`` ratio conditions fire; otherwise approve. ``depth`` ranges
    0 (pure negation, saturated end) upward (more compute, lower K). Every rung also carries
    the decision-irrelevant ``FILLER`` feature."""
    if depth < 0 or depth > len(LADDER_PAIRS):
        raise ValueError(f"depth must be in [0, {len(LADDER_PAIRS)}]; got {depth}")
    fields: list[str] = [FLAG, FILLER]
    sample_facts: dict[str, Any] = {FLAG: True, FILLER: 1.0}
    rules: list[A.Rule] = [A.Rule("false", FLAG, verb="deny")]
    for num, den, pct in LADDER_PAIRS[:depth]:
        fields += [num, den]
        sample_facts[num] = 1.0
        sample_facts[den] = 1.0
        rules.append(A.Rule("ratio", num, pct, den, "deny"))
    policy = (f"This is a synthetic credit-risk decision domain (ladder rung d{depth}). "
              "Approve sound applications; deny unsound ones.")
    return A.DomainSpec(key=f"lad_d{depth}", structure="negation", family="approve_deny",
                        policy=policy, fields=fields, sample_facts=sample_facts, rules=rules)


def build_rungs(n_rungs: int) -> list[A.DomainSpec]:
    """``n_rungs`` ladder specs, depth 0 .. n_rungs-1 (capped at the available pairs)."""
    top = min(n_rungs, len(LADDER_PAIRS) + 1)
    return [make_ladder_spec(d) for d in range(top)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--rungs", type=int, default=len(LADDER_PAIRS) + 1,
                    help=f"number of rungs (depth 0..n-1), max {len(LADDER_PAIRS) + 1}")
    ap.add_argument("--dry-run", action="store_true",
                    help="FakeVerifier + fake ids; no network (scratch workdir/out)")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    specs = build_rungs(args.rungs)
    workdir = args.workdir or REPO / "outputs" / (
        "ladder_dryrun" if args.dry_run else "ladder")
    out = args.out or (workdir / "corpus_dryrun.jsonl" if args.dry_run
                       else REPO / "data" / "decision_ladder.jsonl")
    if args.dry_run:
        backend: Any = A.DryRunBackend()
    else:
        A._load_env()
        backend = A.LiveBackend()
    res = A.run_pipeline(specs, backend, workdir, out)
    print(json.dumps(res, indent=2))
    return 0 if not res["incomplete"] else 1


if __name__ == "__main__":
    sys.exit(main())
