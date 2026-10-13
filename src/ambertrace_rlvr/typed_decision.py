"""Run *typed* contestants over a decision corpus and score their calibration.

A :class:`~ambertrace_rlvr.model_backend.TypedDecider` (Jev, Nimble, or a chat
model wrapped by :func:`text_decider`) answers each item over its **known
candidate set** and returns a
:class:`~ambertrace_rlvr.model_backend.ScoredDecision` — the chosen label plus a
probability for every candidate. :func:`run_typed_model` adapts those decisions
into the same :class:`~ambertrace_rlvr.deviation.ModelAnswer` list the alignment
matrix already scores (accuracy, fail-open, overconfidence), so a System-1 model
sits in the leaderboard beside every chat model with no change to the scorer.

The one thing chat text throws away and a typed decider keeps is the probability
distribution — so this module adds the calibration metrics (**Brier**, **ECE**)
the accuracy-only path cannot compute.

Fail-closed like :func:`ambertrace_rlvr.matrix.run_model`: a decider that raises
on an item is treated as a refusal, never a wrong label — a whole run never
crashes on one bad item.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .corpus import DecisionItem
from .deviation import ModelAnswer, parse_model_answer
from .model_backend import ScoredDecision, TypedDecider

logger = logging.getLogger(__name__)


def run_typed_model(
    items: Sequence[DecisionItem], decider: TypedDecider,
) -> tuple[list[ModelAnswer], list[ScoredDecision]]:
    """Run ``decider`` over each item and return the parsed answers (for the
    alignment scorer) alongside the raw scored decisions (for calibration).

    A decider raising on one item yields a refusal for that item — the run
    continues, mirroring :func:`ambertrace_rlvr.matrix.run_model`."""
    answers: list[ModelAnswer] = []
    decisions: list[ScoredDecision] = []
    for it in items:
        try:
            decision = decider.decide(it.prompt, it.label_space)
        except Exception:  # noqa: BLE001 — any backend failure is a refusal, not a crash
            logger.warning("typed decider raised on item %s; treating as refusal", it.id)
            decision = ScoredDecision(choice=None)
        decisions.append(decision)
        answers.append(parse_model_answer(decision.choice, it.label_space))
    return answers, decisions


@dataclass(frozen=True)
class Calibration:
    """Calibration of a typed contestant's probabilities against the oracle truth.

    ``brier`` is the multiclass Brier score (mean squared error of the full
    probability vector vs. the one-hot oracle label; lower is better, 0 is
    perfect). ``ece`` is the expected calibration error (gap between stated
    confidence and realised accuracy, binned; lower is better). ``n`` is how many
    answered items carried usable probabilities — the denominator for both."""

    brier: float | None
    ece: float | None
    n: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {"brier": self.brier, "ece": self.ece, "n": self.n}


def calibration(
    items: Sequence[DecisionItem], decisions: Sequence[ScoredDecision],
    *, bins: int = 10,
) -> Calibration:
    """Score ``decisions`` against each item's oracle label. Items without an
    oracle, without a returned choice, or without probabilities are skipped (they
    carry no calibration signal); ``n`` reports how many remained."""
    if len(items) != len(decisions):
        raise ValueError(
            f"items ({len(items)}) and decisions ({len(decisions)}) must match")
    brier_terms: list[float] = []
    conf_correct: list[tuple[float, bool]] = []
    for it, d in zip(items, decisions):
        oracle = it.oracle
        if oracle is None or d.choice is None or not d.probabilities:
            continue
        labels = it.label_space
        brier_terms.append(
            sum((d.probabilities.get(k, 0.0) - (1.0 if k == oracle else 0.0)) ** 2
                for k in labels))
        if d.confidence is not None:
            conf_correct.append((d.confidence, d.choice == oracle))
    n = len(brier_terms)
    brier = sum(brier_terms) / n if n else None
    ece = _expected_calibration_error(conf_correct, bins) if conf_correct else None
    return Calibration(brier=brier, ece=ece, n=n)


def _expected_calibration_error(
    conf_correct: Sequence[tuple[float, bool]], bins: int,
) -> float:
    """Bin (confidence, was_correct) pairs and sum each bin's |confidence − accuracy|
    weighted by its share of the sample."""
    total = len(conf_correct)
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for lo, hi in zip(edges, edges[1:]):
        # last bin is closed on the right so confidence == 1.0 lands somewhere.
        bucket = [(c, ok) for c, ok in conf_correct
                  if (lo <= c < hi) or (hi == 1.0 and c == 1.0)]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_conf - acc)
    return ece


@dataclass(frozen=True)
class _TextDecider:
    """Wrap a plain ``prompt -> completion`` chat model as a degenerate
    :class:`TypedDecider`: coerce its text to a label, no probabilities (so it
    contributes accuracy/fail-open but not calibration)."""

    model: Callable[[str], str]

    def decide(self, prompt: str, label_space: Sequence[str]) -> ScoredDecision:
        answer = parse_model_answer(self.model(prompt), label_space)
        return ScoredDecision(choice=answer.value if answer.parse_ok else None)


def text_decider(model: Callable[[str], str]) -> TypedDecider:
    """Adapt a chat ``Model`` callable into a :class:`TypedDecider` so LLM baselines
    can share the typed leaderboard with Jev/Nimble (without calibration)."""
    return _TextDecider(model)
