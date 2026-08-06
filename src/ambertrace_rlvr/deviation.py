"""Deviation scoring: score a model's answers against the oracle's certified
verdicts, with a hard three-bucket partition (#51).

The oracle does not only label decidable cases — it can *certify that no
determinate answer exists* (the policy fixes no action). An LLM answers
everything, including those. That gap is a direct, ground-truth measurement of
**overconfidence on the unverifiable** — a failure mode most methods can only
approximate, because they cannot prove a case is underdetermined.

Every response is partitioned into exactly one bucket, never conflated:

1. **certified** — the oracle fixes an action; deviation-from-truth is valid and
   scored (``correct`` / ``over_permit`` / ``over_deny``). A refusal or an
   uncoercible answer here is its *own* bucket, never counted as "wrong".
2. **certified-undecidable** — the oracle proved there is no determinate answer.
   A model that answers a determinate verb here is **overconfident**; it is never
   scored as wrong. A model that abstains/declines is humility (``mutual_abstain``).
3. **unverifiable** — no checked proof: excluded from both deviation and
   overconfidence scoring.

Hard invariant (enforced structurally, not by convention): a model answer is
*never* scored for deviation against a non-certified oracle output. The fence of
:mod:`ambertrace_rlvr.eval_oracle` holds — only the certificate's normalised
verdict is consumed; how undecidability is *proven* stays opaque.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .eval_oracle import JudgmentSpec, OracleJudgment
from .evaluation import VerifierLike
from .parsers import ParsedCompletion

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ModelAnswer:
    """A model's answer to one item, extracted from its raw output.

    ``answered`` distinguishes a refusal / empty output (``False``) from a produced
    answer; ``parse_ok`` distinguishes a cleanly coerced verb from an uncoercible
    one. The three states — refusal, parse-failure, coerced verb — are kept
    distinct so neither refusal nor parse-failure is ever folded into "wrong"."""

    answered: bool
    parse_ok: bool
    value: str | None = None


def parse_model_answer(text: str | None, label_space: Sequence[Any]) -> ModelAnswer:
    """Coerce a raw model output into a :class:`ModelAnswer` over ``label_space``.

    Reasoning blocks (``<think>…</think>``, incl. an unterminated/truncated one)
    are stripped first — truncated reasoning is a non-answer, not a verdict. An
    empty result is a refusal. Otherwise: an exact label match, else a *unique*
    label appearing as a substring, wins; anything else is a parse failure (an
    answer was produced but could not be coerced) — never a wrong classification."""
    if text is None:
        return ModelAnswer(answered=False, parse_ok=False)
    stripped = _THINK_OPEN_RE.sub(" ", _THINK_RE.sub(" ", text)).strip()
    if not stripped:
        return ModelAnswer(answered=False, parse_ok=False)  # refusal
    low = stripped.lower()
    labels = [str(v) for v in label_space]
    for v in labels:
        if low == v.lower():
            return ModelAnswer(answered=True, parse_ok=True, value=v)
    hits = [v for v in labels if v.lower() in low]
    if len(hits) == 1:
        return ModelAnswer(answered=True, parse_ok=True, value=hits[0])
    return ModelAnswer(answered=True, parse_ok=False)  # produced, uncoercible


@dataclass
class DeviationReport:
    """The three-bucket partition plus its metrics. Counters are additive; rates
    are read-only properties that return ``None`` when their bucket is empty."""

    # certified (scorable) sub-buckets
    correct: int = 0
    over_permit: int = 0            # model less restrictive than oracle (fail-open)
    over_deny: int = 0             # model more restrictive than oracle (fail-closed)
    refusal_on_certified: int = 0
    parse_fail_on_certified: int = 0
    # certified-undecidable
    overconfident: int = 0         # answered a determinate verb where none exists
    mutual_abstain: int = 0        # abstained / declined — agreement of humility
    # unverifiable
    unverifiable: int = 0

    @property
    def n(self) -> int:
        return (self.scored + self.refusal_on_certified + self.parse_fail_on_certified
                + self.abstain_n + self.unverifiable)

    @property
    def scored(self) -> int:
        """Certified items with a coerced answer — the deviation denominator."""
        return self.correct + self.over_permit + self.over_deny

    @property
    def abstain_n(self) -> int:
        """Certified-undecidable items (the overconfidence denominator)."""
        return self.overconfident + self.mutual_abstain

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.scored if self.scored else None

    @property
    def signed_bias(self) -> float | None:
        """(over_permit − over_deny) / scored. Positive = net fail-open."""
        return (self.over_permit - self.over_deny) / self.scored if self.scored else None

    @property
    def overconfidence_rate(self) -> float | None:
        """Fraction of certified-undecidable items the model answered non-abstain."""
        return self.overconfident / self.abstain_n if self.abstain_n else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "scored": self.scored,
            "correct": self.correct, "over_permit": self.over_permit,
            "over_deny": self.over_deny,
            "refusal_on_certified": self.refusal_on_certified,
            "parse_fail_on_certified": self.parse_fail_on_certified,
            "abstain_n": self.abstain_n, "overconfident": self.overconfident,
            "mutual_abstain": self.mutual_abstain, "unverifiable": self.unverifiable,
            "accuracy": self.accuracy, "signed_bias": self.signed_bias,
            "overconfidence_rate": self.overconfidence_rate,
        }


def score_deviation(
    judgments: Sequence[OracleJudgment],
    answers: Sequence[ModelAnswer],
    spec: JudgmentSpec,
) -> DeviationReport:
    """Partition ``(judgment, answer)`` pairs into the three buckets and tally the
    metrics. The partition is a hard invariant: an answer is scored for deviation
    only on a certified-decidable item; certified-undecidable items feed the
    overconfidence counters only; unverifiable items are excluded from both."""
    if len(judgments) != len(answers):
        raise ValueError(
            f"judgments ({len(judgments)}) and answers ({len(answers)}) "
            "must be the same length"
        )
    rep = DeviationReport()
    for judgment, answer in zip(judgments, answers):
        if judgment.certified:
            if not answer.answered:
                rep.refusal_on_certified += 1
                continue
            if not answer.parse_ok:
                rep.parse_fail_on_certified += 1
                continue
            direction = spec.direction(judgment.value, answer.value)
            if direction == "over_permit":
                rep.over_permit += 1
            elif direction == "over_deny":
                rep.over_deny += 1
            else:
                rep.correct += 1
        elif judgment.certified_undecidable:
            # Never scored for deviation. Overconfident iff the model committed to
            # a determinate (non-abstain) verb where the oracle proved none exists.
            answered_determinate = (
                answer.answered and answer.parse_ok
                and answer.value is not None and not spec.is_abstain(answer.value)
            )
            if answered_determinate:
                rep.overconfident += 1
            else:
                rep.mutual_abstain += 1
        else:
            rep.unverifiable += 1
    return rep


@dataclass(frozen=True)
class OracleItem:
    """One eval item's *fixed* inputs, used to query the oracle for its verdict
    (distinct from the model's own asserted facts on the reward path)."""

    query: str
    facts: dict[str, Any]
    id: str | None = None


def oracle_judgments(
    verifier: VerifierLike, items: Sequence[OracleItem], spec: JudgmentSpec | None = None
) -> list[OracleJudgment]:
    """Query the oracle on each item's fixed inputs and normalise to judgments.

    This is the oracle-as-judge pipeline: it certifies the *item's* ground-truth
    inputs (not any model output), so the resulting verdicts are the fixed truth
    the model's separate answers are scored against."""
    parsed: list[ParsedCompletion | None] = [
        ParsedCompletion(query=it.query, facts=dict(it.facts)) for it in items
    ]
    reports = verifier.verify_batch(parsed)
    out: list[OracleJudgment] = []
    for report in reports:
        if report is None:
            out.append(OracleJudgment(certified=False, certified_undecidable=False,
                                      value=None, reason="no_report"))
        else:
            out.append(OracleJudgment.from_report(report, spec))
    return out
