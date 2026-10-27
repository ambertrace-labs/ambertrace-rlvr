"""Generate your own oracle-labelled decision eval set from an AmberTrace platform.

Domain-agnostic: point it at *your* verified platform (authored with the
`ambertraceai` SDK — see examples/author_acmg_platform.py) and a file of input-fact
cases, and it queries the oracle for each case's certified verdict, emitting a
decision corpus in the same schema as the shipped `data/decision_eval_v1.jsonl`.

The decision vocabulary (the verb set AND its restrictiveness) is read FROM THE
PLATFORM's authored **decision table** — not declared out-of-band — so restrictiveness
and the oracle verdict come from the same validated source. (The old `--verbs` /
`--restrictive` flags are gone; the SDK rejects out-of-band `decision_vocabulary`.)

    # offline: prove the wiring with a FakeVerifier (no platform, no network)
    python examples/generate_eval_set.py --dry-run

    # live: label your cases against your platform (needs AMBERTRACE_API_KEY)
    python examples/generate_eval_set.py \
        --platform-id 123 --cases my_cases.jsonl \
        --out data/my_eval.jsonl

`--cases` is JSONL, one case per line: {"prompt": "...", "facts": {...}, "id": "..."}.
The platform must have a declared decision table (domains.set_decision_table).
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
    decision_table_for_platform,
    vocabulary_from_decision_table,
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
    """Prove the generation wiring offline with a FakeVerifier — no platform.

    The vocabulary is mapped from a sample platform DECISION TABLE (ladder mode),
    exactly as the live path reads it from the platform."""
    from ambertrace_rlvr.reports import AmberReport
    from ambertrace_rlvr.testing import FakeVerifier

    # Ladder: higher `order` = more restrictive (the most restrictive fired row wins),
    # so the permissive `approve` sits at the LOWEST order and the fail-closed `deny`
    # is the highest-order default.
    sample_table = {"mode": "ladder", "rows": [
        {"label": "approve", "order": 1, "permits": True, "default": False},
        {"label": "escalate", "order": 2, "permits": False, "default": False},
        {"label": "deny", "order": 3, "permits": False, "default": True},
    ]}
    vocab = vocabulary_from_decision_table(sample_table)

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
    ap.add_argument("--abstain", default=None,
                    help="name the certified-undecidable outcome verb, if the domain has one")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "my_eval.jsonl")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    if not (args.platform_id and args.cases):
        raise SystemExit("live run needs --platform-id and --cases (or use --dry-run)")
    _load_dotenv()
    import ambertraceai

    from ambertrace_rlvr import AmberVerifier, JSONBlockParser, VerifiableDomain

    # Read the decision vocabulary FROM THE PLATFORM's authored decision table, so the
    # verb set and its restrictiveness come from the same validated source as the oracle.
    api = ambertraceai.AmbertraceAPI.from_env()
    table = decision_table_for_platform(api, args.platform_id)
    vocab = vocabulary_from_decision_table(table, abstain=args.abstain)
    print("decision table ->",
          [(v.verb, v.rank, "restrictive" if v.restrictive else "permits") for v in vocab])

    domain = VerifiableDomain.from_env(platform_id=args.platform_id, parser=JSONBlockParser())
    verifier = AmberVerifier(domain=domain)
    items = build_eval_items(verifier, _load_cases(args.cases), vocab)
    path = write_decision_corpus(args.out, items)
    print(f"wrote {len(items)} labelled items -> {path}")
    print("stats:", corpus_stats(items))


if __name__ == "__main__":
    main()
