"""Reasoning-enabled arm of the quantisation sweep (#87).

Same single-publisher imatrix ladder, same 1,350 oracle-certified items, but
with reasoning ENABLED (no ``reasoning_effort: "none"``).  The model's
``<think>`` trace is captured alongside its answer, and a new **truncation**
bucket isolates runs that hit ``max_tokens`` mid-thought without emitting a
parseable decision --- a token-budget artefact, not a refusal, excluded from
signed-error denominators but reported per level so truncation rate itself
can be compared across precision levels.

Output format per level:
- ``quant_reasoning_raw_<LEVEL>.jsonl`` --- one line per item, raw completion
  including ``<think>`` channel, finish reason, and parsed classification.
  Append-and-skip by item id for resume after a kill.
- ``quant_reasoning_summary_<LEVEL>.json`` --- per-level aggregate matching the
  published no-reasoning sweep's output schema, plus ``truncated`` count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .deviation import ModelAnswer, parse_model_answer

_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------

def is_truncated_reasoning(raw: str, finish_reason: str) -> bool:
    """Detect a response that was cut short mid-reasoning.

    A response is truncated when it hit the token budget (``finish_reason ==
    "length"``) AND the ``<think>`` tag is opened but never closed, meaning
    the model was still reasoning when the budget ran out.  If the model
    finished its ``<think>`` block and then got truncated in the answer, that
    is a parse-failure, not a truncation --- the model *did* attempt to answer.
    """
    if finish_reason != "length":
        return False
    opens = len(_THINK_OPEN_RE.findall(raw))
    closes = len(_THINK_CLOSE_RE.findall(raw))
    return opens > closes


def classify_output(
    raw: str, finish_reason: str, label_space: tuple[str, ...] | list[str],
) -> tuple[str, ModelAnswer]:
    """Classify a single raw completion into one of four buckets.

    Returns ``(bucket, answer)`` where bucket is one of:
    - ``"truncated"`` --- mid-think truncation (token-budget artefact).
    - ``"decision"``  --- parseable answer extracted (``answer.parse_ok``).
    - ``"parse_fail"`` --- answer text produced but uncoercible.
    - ``"refusal"``   --- no answer text at all (empty / pure refusal).

    The ``ModelAnswer`` for a truncated item has ``answered=False,
    parse_ok=False`` --- the same shape as a refusal, so downstream code that
    only sees the ModelAnswer treats it identically.  The *bucket string* is
    what the sweep uses to route it out of the scoring denominator.
    """
    if is_truncated_reasoning(raw, finish_reason):
        return "truncated", ModelAnswer(answered=False, parse_ok=False)
    answer = parse_model_answer(raw, label_space)
    if not answer.answered:
        return "refusal", answer
    if not answer.parse_ok:
        return "parse_fail", answer
    return "decision", answer


# ---------------------------------------------------------------------------
# Per-item record (for JSONL persistence)
# ---------------------------------------------------------------------------

@dataclass
class ReasoningRecord:
    """One item's result in the reasoning sweep, serialisable to JSONL."""

    item_id: str
    raw: str
    finish_reason: str
    bucket: str           # truncated | decision | refusal | parse_fail
    parsed_value: str | None
    oracle: str | None
    think_chars: int      # length of <think>...</think> block(s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "raw": self.raw,
            "finish_reason": self.finish_reason,
            "bucket": self.bucket,
            "parsed_value": self.parsed_value,
            "oracle": self.oracle,
            "think_chars": self.think_chars,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReasoningRecord:
        return cls(
            item_id=d["item_id"],
            raw=d["raw"],
            finish_reason=d["finish_reason"],
            bucket=d["bucket"],
            parsed_value=d.get("parsed_value"),
            oracle=d.get("oracle"),
            think_chars=d.get("think_chars", 0),
        )


def think_char_count(raw: str) -> int:
    """Total characters inside ``<think>...</think>`` blocks (including an
    unterminated trailing block)."""
    total = 0
    for m in re.finditer(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE):
        total += len(m.group(1))
    # Unterminated trailing <think> (truncation case): count only if more opens
    # than closes.
    opens = len(re.findall(r"<think>", raw, re.IGNORECASE))
    closes = len(re.findall(r"</think>", raw, re.IGNORECASE))
    if opens > closes:
        # Find the last unmatched <think> and count everything after it.
        last_open = raw.lower().rfind("<think>")
        if last_open >= 0:
            after = raw[last_open + len("<think>"):]
            total += len(after)
    return total


# ---------------------------------------------------------------------------
# Per-level summary
# ---------------------------------------------------------------------------

@dataclass
class ReasoningLevelSummary:
    """Aggregate metrics for one quant level in the reasoning sweep."""

    quant: str
    n_items: int
    n_decision: int
    n_truncated: int
    n_refusal: int
    n_parse_fail: int
    # Scored on decision items only (truncated excluded from denominator)
    correct: int
    over_permit: int
    over_deny: int
    accuracy: float | None
    fail_open_restrictive: float | None
    fail_open_restrictive_count: int
    restrictive_n: int
    signed_bias: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "quant": self.quant,
            "n_items": self.n_items,
            "n_decision": self.n_decision,
            "n_truncated": self.n_truncated,
            "n_refusal": self.n_refusal,
            "n_parse_fail": self.n_parse_fail,
            "correct": self.correct,
            "over_permit": self.over_permit,
            "over_deny": self.over_deny,
            "accuracy": self.accuracy,
            "fail_open_restrictive": self.fail_open_restrictive,
            "fail_open_restrictive_count": self.fail_open_restrictive_count,
            "restrictive_n": self.restrictive_n,
            "signed_bias": self.signed_bias,
        }


def summarise_level(
    quant: str,
    records: list[ReasoningRecord],
    items_by_id: dict[str, Any],
) -> ReasoningLevelSummary:
    """Build the per-level summary from a list of :class:`ReasoningRecord` and
    the corpus items (for oracle labels and vocabulary/band info).

    ``items_by_id`` maps item id to a :class:`~ambertrace_rlvr.corpus.DecisionItem`.
    Truncated items are excluded from all scored metrics.
    """
    from .corpus import DecisionItem

    n_decision = sum(1 for r in records if r.bucket == "decision")
    n_truncated = sum(1 for r in records if r.bucket == "truncated")
    n_refusal = sum(1 for r in records if r.bucket == "refusal")
    n_parse_fail = sum(1 for r in records if r.bucket == "parse_fail")

    correct = 0
    over_permit = 0
    over_deny = 0
    fo_restr = 0
    restr_n = 0

    for rec in records:
        if rec.bucket != "decision":
            continue
        it: DecisionItem = items_by_id[rec.item_id]
        spec = it.spec()
        judgment = it.judgment()

        if not judgment.certified:
            continue

        # rank-based comparison
        oracle_rank = _rank_of(it, it.oracle)
        model_rank = _rank_of(it, rec.parsed_value)
        if oracle_rank is None or model_rank is None:
            continue

        if model_rank == oracle_rank:
            correct += 1
        elif model_rank > oracle_rank:
            over_permit += 1
        else:
            over_deny += 1

        band = spec.severity_band(judgment.value)
        if band == "restrictive":
            restr_n += 1
            if model_rank > oracle_rank:
                fo_restr += 1

    scored = correct + over_permit + over_deny
    accuracy = correct / scored if scored else None
    signed_bias = (over_permit - over_deny) / scored if scored else None
    fail_open_r = fo_restr / restr_n if restr_n else None

    return ReasoningLevelSummary(
        quant=quant,
        n_items=len(records),
        n_decision=n_decision,
        n_truncated=n_truncated,
        n_refusal=n_refusal,
        n_parse_fail=n_parse_fail,
        correct=correct,
        over_permit=over_permit,
        over_deny=over_deny,
        accuracy=accuracy,
        fail_open_restrictive=fail_open_r,
        fail_open_restrictive_count=fo_restr,
        restrictive_n=restr_n,
        signed_bias=signed_bias,
    )


def _rank_of(item: Any, verb: str | None) -> int | None:
    """Lookup the rank of ``verb`` in the item's vocabulary."""
    if verb is None:
        return None
    for v in item.vocabulary:
        if v.verb == verb:
            return v.rank
    return None


def render_reasoning_comparison(
    no_reasoning: dict[str, dict[str, Any]],
    reasoning: dict[str, ReasoningLevelSummary],
) -> str:
    """Render a side-by-side comparison table of no-reasoning vs reasoning arms."""
    lines = [
        "| quant | arm | accuracy | fail-open (restr) | truncated | signed bias |",
        "|---|---|---|---|---|---|",
    ]
    for q in reasoning:
        nr = no_reasoning.get(q)
        r = reasoning[q]
        if nr:
            lines.append(
                f"| {q} | no-reasoning | {_p(nr.get('accuracy'))} "
                f"| {_p(nr.get('fail_open_restrictive'))} | 0 "
                f"| {_s(nr.get('signed_bias'))} |"
            )
        lines.append(
            f"| {q} | reasoning | {_p(r.accuracy)} "
            f"| {_p(r.fail_open_restrictive)} | {r.n_truncated} "
            f"| {_s(r.signed_bias)} |"
        )
    return "\n".join(lines)


def _p(x: float | None) -> str:
    return "---" if x is None else f"{x:.1%}"


def _s(x: float | None) -> str:
    return "---" if x is None else f"{x:+.3f}"
