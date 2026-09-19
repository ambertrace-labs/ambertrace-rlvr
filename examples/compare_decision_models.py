"""Compare AmberTrace against System-1 decision models (Jev, Nimble) on one corpus.

Runs each contestant as a :class:`~ambertrace_rlvr.model_backend.TypedDecider` over
a decision corpus and prints the alignment matrix (accuracy + error direction) plus
a calibration line (Brier / ECE) — the metric a System-1 model's probabilities make
possible and an accuracy-only leaderboard cannot show.

    # offline: prove the three-contestant wiring with deterministic stubs (no keys)
    python examples/compare_decision_models.py --dry-run

    # live Jev over the OOD probe (needs TYPESAFE_API_KEY)
    python examples/compare_decision_models.py --jev --corpus data/ood_probe_v1.jsonl

Nimble (local) and the AmberTrace SDK runtime contestant are wired through the same
:class:`TypedDecider` seam and land in follow-ups on #123; this slice proves the
harness end-to-end and runs live Jev.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ambertrace_rlvr import (
    Calibration,
    JevProvider,
    NimbleProvider,
    calibration,
    load_decision_corpus,
    render_matrix,
    run_typed_model,
    score_alignment,
)
from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.matrix import AlignmentRow, _rank_key
from ambertrace_rlvr.model_backend import ScoredDecision, TypedDecider

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO / "data" / "ood_probe_v1.jsonl"


def _score(
    items: Sequence[DecisionItem], contestants: dict[str, TypedDecider],
    *, min_parsed: int = 10,
) -> list[tuple[AlignmentRow, Calibration]]:
    """Run + score every contestant, pairing each alignment row with its
    calibration so both can be reported and persisted from one pass."""
    scored: list[tuple[AlignmentRow, Calibration]] = []
    for name, decider in contestants.items():
        answers, decisions = run_typed_model(items, decider)
        row = score_alignment(items, answers, model=name, min_parsed=min_parsed)
        scored.append((row, calibration(items, decisions)))
    scored.sort(key=lambda rc: _rank_key(rc[0]))
    return scored


def _report(scored: Sequence[tuple[AlignmentRow, Calibration]]) -> None:
    print("\n" + render_matrix([row for row, _ in scored]))
    print("\ncalibration (lower is better):")
    for row, cal in scored:
        print(f"  {row.model:<16} {cal.as_dict()}")


class _StubDecider:
    """Offline stand-in: echoes the oracle, but on a given fraction of items fails
    open (picks the least-restrictive verb) with high stated confidence — enough to
    populate the matrix and show a calibration gap without any network."""

    def __init__(self, items: Sequence[DecisionItem], *, fail_open_every: int) -> None:
        self._oracle = {it.prompt: it for it in items}
        self._n = fail_open_every

    def decide(self, prompt: str, label_space: Sequence[str]) -> ScoredDecision:
        it = self._oracle[prompt]
        least_restrictive = max(it.vocabulary, key=lambda v: v.rank).verb
        idx = list(self._oracle).index(prompt)
        confident_wrong = self._n and idx % self._n == 0 and it.oracle != least_restrictive
        choice = least_restrictive if confident_wrong else (it.oracle or least_restrictive)
        probs = {v: (0.9 if v == choice else 0.1 / max(1, len(label_space) - 1))
                 for v in label_space}
        return ScoredDecision(choice=choice, probabilities=probs, confidence=0.9)


def dry_run() -> None:
    items = load_decision_corpus(DEFAULT_CORPUS)[:80]
    contestants: dict[str, TypedDecider] = {
        "ambertrace-stub": _StubDecider(items, fail_open_every=0),   # never fails open
        "jev-stub": _StubDecider(items, fail_open_every=5),
        "nimble-stub": _StubDecider(items, fail_open_every=4),
    }
    _report(_score(items, contestants, min_parsed=5))
    print("\nOK — three-contestant matrix + calibration wiring is sound "
          f"(offline stubs, {len(items)} items).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="offline stubs, no keys")
    ap.add_argument("--jev", action="store_true", help="run live Jev (TYPESAFE_API_KEY)")
    ap.add_argument("--nimble", metavar="CONFIG", nargs="?",
                    const=".cache/nimble-model.json",
                    help="run local Nimble from a model config (default .cache/nimble-model.json)")
    ap.add_argument("--nimble-cuda", action="store_true", help="use the CUDA scorer for --nimble")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "decision_comparison.json")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    items = load_decision_corpus(args.corpus)
    if args.limit:
        items = items[: args.limit]
    contestants: dict[str, TypedDecider] = {}
    if args.jev:
        contestants["jev"] = JevProvider()
    if args.nimble:
        contestants["nimble"] = NimbleProvider.load(args.nimble, cuda=args.nimble_cuda)
    if not contestants:
        raise SystemExit("no live contestant selected (use --jev / --nimble, or --dry-run)")

    print(f"scoring {len(contestants)} contestant(s) over {len(items)} items…")
    scored = _score(items, contestants)
    _report(scored)

    out = [{**row.as_dict(), "calibration": cal.as_dict()}
           for row, cal in scored]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"corpus": args.corpus.name, "n_items": len(items), "rows": out}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
