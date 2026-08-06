"""Load a frozen, oracle-labelled decision corpus for the alignment matrix.

A decision corpus is a JSONL file, one **eval item** per line, in a small
public schema — an *answer key*, not a proof. It carries only what a third party
needs to score any model against a fixed ground truth:

    {
      "id": "acc-0007",
      "domain": "access_control",
      "prompt": "A device with an out-of-date posture requests admin ...",
      "vocabulary": [
        {"verb": "deny", "rank": 0, "restrictive": true},
        {"verb": "escalate", "rank": 3, "restrictive": true},
        {"verb": "approve", "rank": 9, "restrictive": false}
      ],
      "oracle": "escalate",          // the certified verdict; null/omitted if undecidable
      "undecidable": false,          // certified-undecidable (no determinate answer)
      "difficulty": {"family": "precedence", "shape": "deep_verb_lattice"}
    }

Deliberately **fenced**: the schema has no place for the proof, the symbolic
trace, the rule program, or any verifier internal — the oracle stays a black box,
only its label ships. The label is trusted as frozen (it was certified when the
corpus was built); :func:`ambertrace_rlvr.eval_oracle.OracleJudgment.from_label`
turns it into a judgment without live oracle access.

This module never authors or queries a platform; it just reads the labelled set
and produces the per-domain :class:`JudgmentSpec` and the judgments/prompts the
deviation + sycophancy scorers consume.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .eval_oracle import JudgmentSpec, LabelSpec, OracleJudgment


@dataclass(frozen=True)
class DecisionItem:
    """One labelled eval item. ``oracle`` is the certified verdict (``None`` when
    ``undecidable``). ``difficulty`` holds free-form stratification tags."""

    id: str
    domain: str
    prompt: str
    vocabulary: tuple[LabelSpec, ...]
    oracle: str | None = None
    undecidable: bool = False
    difficulty: dict[str, Any] = field(default_factory=dict)

    @property
    def label_space(self) -> tuple[str, ...]:
        return tuple(v.verb for v in self.vocabulary)

    def spec(self) -> JudgmentSpec:
        return JudgmentSpec(labels=list(self.vocabulary))

    def judgment(self) -> OracleJudgment:
        """The frozen oracle verdict as an :class:`OracleJudgment`."""
        return OracleJudgment.from_label(self.oracle, undecidable=self.undecidable)

    def to_record(self) -> dict[str, Any]:
        """The public answer-key record for this item (round-trips through
        :func:`load_decision_corpus`)."""
        return {
            "id": self.id,
            "domain": self.domain,
            "prompt": self.prompt,
            "vocabulary": [
                {"verb": v.verb, "rank": v.rank, "restrictive": v.restrictive}
                for v in self.vocabulary
            ],
            "oracle": None if self.undecidable else self.oracle,
            "undecidable": self.undecidable,
            "difficulty": dict(self.difficulty),
        }


def load_decision_corpus(path: str | Path) -> list[DecisionItem]:
    """Read a decision corpus JSONL. Malformed lines and items missing a prompt or
    vocabulary are skipped (a partial/edited file never crashes the loader). An
    item is ``undecidable`` if flagged or if it carries no ``oracle`` verb."""
    items: list[DecisionItem] = []
    for i, raw in enumerate(Path(path).read_text().splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        item = _item_from_record(rec, fallback_id=str(i))
        if item is not None:
            items.append(item)
    return items


def _item_from_record(rec: Any, *, fallback_id: str) -> DecisionItem | None:
    if not isinstance(rec, dict):
        return None
    prompt = rec.get("prompt") or rec.get("situation")
    vocab = _parse_vocabulary(rec.get("vocabulary"))
    if not isinstance(prompt, str) or not prompt or not vocab:
        return None
    oracle = rec.get("oracle")
    undecidable = bool(rec.get("undecidable", False)) or oracle is None
    raw_difficulty = rec.get("difficulty")
    difficulty: dict[str, Any] = (
        {str(k): v for k, v in raw_difficulty.items()}
        if isinstance(raw_difficulty, dict) else {}
    )
    return DecisionItem(
        id=str(rec.get("id", fallback_id)),
        domain=str(rec.get("domain", "")),
        prompt=prompt,
        vocabulary=vocab,
        oracle=None if undecidable else str(oracle),
        undecidable=undecidable,
        difficulty=difficulty,
    )


def _parse_vocabulary(vocab: Any) -> tuple[LabelSpec, ...]:
    out: list[LabelSpec] = []
    for v in vocab if isinstance(vocab, list) else []:
        if not isinstance(v, dict):
            continue
        verb = v.get("verb")
        if not verb:
            continue
        out.append(LabelSpec(
            verb=str(verb),
            rank=int(v.get("rank", 0)),
            restrictive=bool(v.get("restrictive", False)),
            is_abstain=bool(v.get("is_abstain", False)),
        ))
    return tuple(out)


def judgments_for(items: Sequence[DecisionItem]) -> list[OracleJudgment]:
    """The frozen oracle judgment for each item — the fixed truth a model's answers
    are scored against."""
    return [it.judgment() for it in items]


def write_decision_corpus(path: str | Path, items: Iterable[DecisionItem]) -> Path:
    """Write items to a JSONL decision corpus in the public answer-key schema."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(it.to_record()) for it in items) + "\n")
    return out


def corpus_stats(items: Iterable[DecisionItem]) -> dict[str, Any]:
    """Coverage summary: item counts, decidable/undecidable split, and the spread
    across domains and difficulty tags — a sanity check that the set is balanced
    before a run (and that the undecidable bucket is actually present, for #51)."""
    items = list(items)
    difficulty: dict[str, Counter[str]] = {}
    for it in items:
        for k, v in it.difficulty.items():
            difficulty.setdefault(k, Counter())[str(v)] += 1
    return {
        "n": len(items),
        "decidable": sum(1 for it in items if not it.undecidable),
        "undecidable": sum(1 for it in items if it.undecidable),
        "domains": len({it.domain for it in items}),
        "difficulty": {k: dict(c) for k, c in difficulty.items()},
    }
