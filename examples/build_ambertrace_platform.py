"""Author a verified AmberTrace platform for a decision corpus domain.

Turns a corpus's shared policy + its case facts into a verified platform you can
then score as a contestant:

    python examples/build_ambertrace_platform.py --corpus data/ood_probe_v1.jsonl
    # -> prints: platform_id=<N>
    python examples/compare_decision_models.py --jev --ambertrace-platform <N> \
        --corpus data/ood_probe_v1.jsonl

The policy prose is taken from the corpus prompts (the text before "Case facts:");
the features dataset is built from the parsed case facts (features only — never the
oracle label). Needs AMBERTRACE_API_KEY / AMBERTRACE_BASE_URL in the environment.

Assumes the corpus shares one policy (true for ood_probe_v1). For a multi-domain
corpus, filter to one domain first (--domain) or build per domain.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from ambertrace_rlvr import build_verified_platform, load_decision_corpus, parse_case_facts
from ambertrace_rlvr.corpus import DecisionItem

REPO = Path(__file__).resolve().parent.parent


def _policy_text(item: DecisionItem) -> str:
    """The policy prose: everything before the 'Case facts:' block, minus the
    generic role preamble line."""
    head = item.prompt.split("Case facts:", 1)[0].strip()
    lines = [ln for ln in head.splitlines()
             if not ln.lower().startswith("you are the decision-maker")]
    return "\n".join(lines).strip()


def _write_features_csv(items: list[DecisionItem], path: Path) -> list[str]:
    """Write a features-only CSV (union of fact fields) from the parsed case facts."""
    rows = [parse_case_facts(it.prompt) for it in items]
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return columns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "data" / "ood_probe_v1.jsonl")
    ap.add_argument("--domain", help="filter to a single domain id before building")
    ap.add_argument("--policy-contains", metavar="SUBSTR",
                    help="filter to items whose prompt contains SUBSTR (selects one "
                         "policy group that shares a schema, e.g. 'loan approval')")
    ap.add_argument("--name", default=None, help="platform/domain name")
    ap.add_argument("--tau", type=float, default=0.6, help="verified_min_confidence")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    items = load_decision_corpus(args.corpus)
    if args.domain:
        items = [it for it in items if it.domain == args.domain]
    if args.policy_contains:
        items = [it for it in items if args.policy_contains in it.prompt]
    if not items:
        raise SystemExit("no items to build from")

    policy = _policy_text(items[0])
    name = args.name or f"decision-eval::{args.corpus.stem}"
    print(f"policy ({len(policy)} chars):\n{policy}\n")

    from ambertraceai import AmbertraceAPI  # lazy: SDK only needed live

    api = AmbertraceAPI()
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        csv_path = Path(tmp.name)
    columns = _write_features_csv(items, csv_path)
    print(f"features dataset: {len(items)} rows x {len(columns)} cols {columns}")
    print("building verified platform (author -> upload -> ontology -> platform)…")
    platform_id = build_verified_platform(
        api, name=name, policy=policy, dataset_path=str(csv_path),
        tau=args.tau, timeout=args.timeout)
    print(f"\nplatform_id={platform_id}")


if __name__ == "__main__":
    main()
