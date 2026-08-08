"""Generate an oracle-labelled decision eval set from *your own* AmberTrace
platform — the domain-agnostic counterpart to the shipped `decision_eval_v1`.

You author a decision domain and build its verified platform with the
[`ambertraceai`](https://pypi.org/project/ambertraceai/) SDK (needs an AmberTrace
API key); this module then turns a set of input-fact cases into a frozen
:class:`~ambertrace_rlvr.corpus.DecisionItem` set by **querying that platform's
oracle live for each case's certified verdict** — the label is re-derived from the
oracle, never taken from the inputs. The result loads and scores through exactly
the same path as the shipped dataset (:func:`ambertrace_rlvr.corpus.load_decision_corpus`).

Deliberately domain-agnostic: it bakes in no domain, no vocabulary, and no rules.
You bring the platform and the cases; the oracle supplies the truth. Everything
here goes through the public SDK surface — no benchmark-internal machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .corpus import DecisionItem
from .deviation import OracleItem, oracle_judgments
from .eval_oracle import ABSTAIN, JudgmentSpec, LabelSpec
from .evaluation import VerifierLike


@dataclass(frozen=True)
class EvalCase:
    """One case to label: the ``prompt`` shown to a model under evaluation, and the
    ``facts`` (fixed ground-truth inputs) the oracle is queried on. ``query`` is the
    natural-language query sent to the platform (defaults to ``prompt``)."""

    prompt: str
    facts: dict[str, Any]
    id: str | None = None
    domain: str = ""
    query: str | None = None
    difficulty: dict[str, Any] = field(default_factory=dict)
    # Optional native Prediction -> Decision fan-in: ``{role: {"model_id",
    # "as_of", "mode"?}}``. When set, the oracle is queried with a *certified
    # prediction* folded in by reference (not an observed ``facts`` scalar), so
    # the item's verdict is prediction-conditioned (#75).
    predictions: dict[str, dict[str, str | None]] | None = None


def build_eval_items(
    verifier: VerifierLike,
    cases: Sequence[EvalCase],
    vocabulary: Sequence[LabelSpec],
    *,
    spec: JudgmentSpec | None = None,
) -> list[DecisionItem]:
    """Query the oracle on each case's fixed inputs and emit labelled
    :class:`DecisionItem`\\ s.

    The certified verdict becomes the item's ``oracle`` label; a certified-
    undecidable verdict sets ``undecidable=True``. An *unverifiable* case (no
    checked proof / error) is dropped — an eval item needs a real oracle label, and
    a fail-closed non-answer is not one. ``vocabulary`` is the platform's ordered
    decision vocabulary (used for scoring direction/severity later)."""
    judgment_spec = spec or JudgmentSpec(labels=list(vocabulary))
    oracle_items = [
        OracleItem(query=c.query or c.prompt, facts=c.facts, id=c.id,
                   predictions=c.predictions)
        for c in cases
    ]
    judgments = oracle_judgments(verifier, oracle_items, judgment_spec)

    vocab = tuple(vocabulary)
    items: list[DecisionItem] = []
    for i, (case, judgment) in enumerate(zip(cases, judgments)):
        if not judgment.certified and not judgment.certified_undecidable:
            continue  # unverifiable — no usable oracle label
        undecidable = judgment.certified_undecidable
        items.append(DecisionItem(
            id=case.id or f"item-{i:04d}",
            domain=case.domain,
            prompt=case.prompt,
            vocabulary=vocab,
            oracle=None if undecidable else str(judgment.value),
            undecidable=undecidable,
            difficulty=dict(case.difficulty),
        ))
    return items


def vocabulary_from_verbs(
    verbs: Sequence[str],
    *,
    restrictive: Sequence[str] = (),
    abstain: str | None = None,
) -> list[LabelSpec]:
    """Build an ordered :class:`LabelSpec` vocabulary from verbs listed
    **most-restrictive first**. ``restrictive`` names the fail-closed/safety-critical
    verbs; ``abstain`` names the certified-undecidable verb, if the domain has one."""
    restrictive_set = {v.lower() for v in restrictive}
    labels: list[LabelSpec] = []
    for rank, verb in enumerate(verbs):
        labels.append(LabelSpec(
            verb=verb,
            rank=rank,
            restrictive=verb.lower() in restrictive_set,
            is_abstain=(abstain is not None and verb.lower() == abstain.lower())
                       or verb == ABSTAIN,
        ))
    return labels
