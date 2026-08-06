"""Generate your own oracle-labelled decision eval set from an AmberTrace platform.

Domain-agnostic: point it at *your* verified platform (authored with the
`ambertraceai` SDK — see examples/author_demo_platform.py) and a file of input-fact
cases, and it queries the oracle for each case's certified verdict, emitting a
decision corpus in the same schema as the shipped `data/decision_eval_v1.jsonl`.

    # offline: prove the wiring with a FakeVerifier (no platform, no network)
    python examples/generate_eval_set.py --dry-run

    # live: label your cases against your platform (needs AMBERTRACE_API_KEY)
    python examples/generate_eval_set.py \
        --platform-id 123 --cases my_cases.jsonl \
        --verbs deny,escalate,approve --restrictive deny,escalate \
        --out data/my_eval.jsonl

`--cases` is JSONL, one case per line: {"prompt": "...", "facts": {...}, "id": "..."}.
`--verbs` lists the decision vocabulary MOST-RESTRICTIVE first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ambertrace_rlvr import (
    EvalCase,
    build_eval_items,
    corpus_stats,
    vocabulary_from_verbs,
    write_decision_corpus,
)

REPO = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cases.append(EvalCase(
            prompt=rec["prompt"], facts=rec.get("facts", {}),
            id=rec.get("id") or f"case-{i:04d}", domain=rec.get("domain", ""),
            query=rec.get("query"), difficulty=rec.get("difficulty", {}),
        ))
    return cases


def dry_run() -> None:
    """Prove the generation wiring offline with a FakeVerifier — no platform."""
    from ambertrace_rlvr.reports import AmberReport
    from ambertrace_rlvr.testing import FakeVerifier

    vocab = vocabulary_from_verbs(["deny", "escalate", "approve"],
                                  restrictive=["deny", "escalate"])

    # A stand-in oracle: certifies a verdict from each case's hidden ground truth.
    def report_fn(parsed) -> AmberReport:
        from ambertrace_rlvr.testing import make_report
        return make_report(proof_checked=True, decision=parsed.facts.get("_truth"))

    cases = [
        EvalCase(prompt="Case A", facts={"risk": "high", "_truth": "deny"}, id="a"),
        EvalCase(prompt="Case B", facts={"risk": "low", "_truth": "approve"}, id="b"),
    ]
    items = build_eval_items(FakeVerifier(report_fn=report_fn), cases, vocab)
    print("generated items:", [(it.id, it.oracle) for it in items])
    print("stats:", corpus_stats(items))
    assert {it.oracle for it in items} == {"deny", "approve"}
    print("OK — generation wiring is sound (oracle labels derived live).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="offline wiring check with a FakeVerifier (no platform)")
    ap.add_argument("--platform-id", type=int)
    ap.add_argument("--cases", type=Path, help="JSONL of {prompt, facts, id?}")
    ap.add_argument("--verbs", help="decision vocabulary, most-restrictive first, comma-separated")
    ap.add_argument("--restrictive", default="", help="comma-separated fail-closed verbs")
    ap.add_argument("--abstain", default=None, help="the certified-undecidable verb, if any")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "my_eval.jsonl")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    if not (args.platform_id and args.cases and args.verbs):
        raise SystemExit("live run needs --platform-id, --cases and --verbs "
                         "(or use --dry-run)")
    _load_dotenv()
    from ambertrace_rlvr import AmberVerifier, JSONBlockParser, VerifiableDomain

    vocab = vocabulary_from_verbs(
        [v.strip() for v in args.verbs.split(",") if v.strip()],
        restrictive=[v.strip() for v in args.restrictive.split(",") if v.strip()],
        abstain=args.abstain,
    )
    domain = VerifiableDomain.from_env(platform_id=args.platform_id, parser=JSONBlockParser())
    verifier = AmberVerifier(domain=domain)
    items = build_eval_items(verifier, _load_cases(args.cases), vocab)
    path = write_decision_corpus(args.out, items)
    print(f"wrote {len(items)} labelled items -> {path}")
    print("stats:", corpus_stats(items))


if __name__ == "__main__":
    main()
