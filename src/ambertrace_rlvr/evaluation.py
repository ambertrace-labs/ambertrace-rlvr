"""Evaluation harness: score a policy's completions over an eval set and emit the
spec §12 metrics.

Network-free by construction: the policy's completions are supplied (or produced
by a trivial baseline callable), and scoring reuses the existing
parser / verifier / shaper — this module never generates text, downloads a model,
or duplicates reward logic. It is fail-closed: a malformed completion, a parser
error, or a shaping error resolves to a parse failure / reward floor, never an
exception into the caller.

Metrics (per §12):

* ``parse_rate``        — fraction of completions the parser accepted.
* ``certified_rate``    — fraction with ``report.proof_checked`` (certification rate).
* ``accuracy_vs_gold``  — fraction of gold-bearing completions whose answer matches
                          gold (derived from the shaper's ``correctness`` component).
* ``certified_accuracy``— fraction correct *and* certified (§12 "certified accuracy").
* ``mean_reward``       — mean shaped reward.
* ``components``        — mean of each :class:`RewardBreakdown` component over the
                          scored completions (the reward-component traces).

Baselines are trivial policies (:func:`constant_policy`, :func:`malformed_policy`);
:func:`compare_to_baseline` reports the metric delta of a policy vs a baseline —
the "baselines" and "reward-hacking gap" hooks of §12. :func:`consistency` reports
agreement across paraphrases / repeated sampling (the ClinVar-conflict test).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .parsers import CompletionParser, JSONBlockParser, ParsedCompletion
from .prompts import DECISION_CLOSE, DECISION_OPEN
from .reports import AmberReport
from .rewards import DefaultRewardShaper, RewardShaper
from .verifier import RewardFunction

logger = logging.getLogger(__name__)

# A baseline / policy is a pure prompt -> completion callable (no network).
Policy = Callable[[str], str]


@runtime_checkable
class VerifierLike(Protocol):
    """The slice of ``AmberVerifier`` / ``FakeVerifier`` the harness needs."""

    def verify_batch(
        self, parsed: list[ParsedCompletion | None]
    ) -> list[AmberReport | None]: ...


@dataclass
class EvalSample:
    """One eval item: a prompt, optional gold answer, and optional metadata
    (e.g. ``criteria_gold`` for per-criterion partial credit)."""

    prompt: str
    gold: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """Metrics over an eval split. Rate/accuracy fields are ``None`` when they
    cannot be computed (no gold present, or no verifier to certify against)."""

    n: int
    mean_reward: float
    parse_rate: float | None = None
    certified_rate: float | None = None
    accuracy_vs_gold: float | None = None
    certified_accuracy: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)


# --- per-sample scoring record --------------------------------------------
@dataclass
class _Record:
    parsed: bool
    certified: bool
    reward: float
    components: dict[str, float]
    has_gold: bool
    correct: bool


def evaluate(
    samples: Sequence[EvalSample],
    completions: Sequence[str],
    *,
    verifier: VerifierLike | None = None,
    shaper: RewardShaper | None = None,
    parser: CompletionParser | None = None,
    reward_fn: RewardFunction | None = None,
    floor: float | None = None,
) -> EvalMetrics:
    """Score ``completions`` (one per sample) and return :class:`EvalMetrics`.

    Supply either a ``verifier`` (full metrics — parse/certification rates and the
    component breakdown, scored via its ``verify_batch`` and ``shaper``) or a
    ``reward_fn`` (reward-only; pass ``parser`` too for ``parse_rate`` /
    ``accuracy_vs_gold``). Never raises on a bad completion.
    """
    if len(samples) != len(completions):
        raise ValueError(
            f"samples ({len(samples)}) and completions ({len(completions)}) "
            "must be the same length"
        )
    if verifier is None and reward_fn is None:
        raise ValueError("evaluate() needs a 'verifier' or a 'reward_fn'")

    if verifier is not None:
        return _evaluate_with_verifier(
            samples, completions,
            verifier=verifier,
            shaper=shaper if shaper is not None else _resolve_shaper(verifier),
            parser=parser if parser is not None else _resolve_parser(verifier),
            floor=floor if floor is not None else float(getattr(verifier, "floor", -1.0)),
        )
    assert reward_fn is not None  # for the type checker; guarded above
    return _evaluate_with_reward_fn(
        samples, completions, reward_fn=reward_fn, parser=parser,
        floor=floor if floor is not None else -1.0,
    )


def evaluate_policy(
    policy: Policy, samples: Sequence[EvalSample], **kwargs: Any
) -> EvalMetrics:
    """Run ``policy`` over ``samples`` to produce completions, then
    :func:`evaluate`. Convenience for baseline comparisons."""
    return evaluate(samples, run_policy(policy, samples), **kwargs)


def run_policy(policy: Policy, samples: Sequence[EvalSample]) -> list[str]:
    """Apply ``policy`` to each prompt. Fail-closed: a policy that raises yields
    an empty completion (a parse failure downstream), never an exception."""
    out: list[str] = []
    for s in samples:
        try:
            out.append(policy(s.prompt))
        except Exception:  # a broken policy must not sink the eval
            logger.exception("policy raised; emitting empty completion")
            out.append("")
    return out


# --- baseline policies ------------------------------------------------------
def constant_policy(
    answer: Any,
    facts: Mapping[str, Any] | None = None,
    *,
    answer_key: str = "classification",
    facts_key: str = "facts",
) -> Policy:
    """A degenerate baseline: emits the same well-formed ``<decision>`` block for
    every prompt, ignoring the input. Parses fine but is right only by luck — a
    trained policy should beat it on ``accuracy_vs_gold`` / ``mean_reward``."""
    block = json.dumps({answer_key: answer, facts_key: dict(facts or {})})
    completion = f"{DECISION_OPEN}{block}{DECISION_CLOSE}"
    return lambda _prompt: completion


def malformed_policy(text: str = "no decision block here") -> Policy:
    """A degenerate baseline that never emits a ``<decision>`` block — every
    completion is a parse failure and floors the reward."""
    return lambda _prompt: text


def compare_to_baseline(
    policy: EvalMetrics, baseline: EvalMetrics
) -> dict[str, float]:
    """Metric delta (policy − baseline). Only metrics both runs computed are
    included; component deltas are keyed ``component.<name>``. A real policy
    should show positive deltas over a degenerate baseline (the §12 baseline /
    reward-hacking-gap comparison)."""
    deltas: dict[str, float] = {}
    for name in (
        "mean_reward", "parse_rate", "certified_rate",
        "accuracy_vs_gold", "certified_accuracy",
    ):
        a = getattr(policy, name)
        b = getattr(baseline, name)
        if a is not None and b is not None:
            deltas[name] = a - b
    for k, v in policy.components.items():
        if k in baseline.components:
            deltas[f"component.{k}"] = v - baseline.components[k]
    return deltas


def consistency(
    groups: Sequence[Sequence[str]], *, parser: CompletionParser | None = None
) -> float:
    """Mean agreement across paraphrases / repeated sampling (the ClinVar-conflict
    test): for each group of completions of the *same* input, the fraction sharing
    the modal proposed answer. A parse failure is its own answer, so noise lowers
    agreement. Returns ``0.0`` for no groups."""
    p = parser if parser is not None else JSONBlockParser()
    scores: list[float] = []
    for group in groups:
        if not group:
            continue
        answers: list[str] = []
        for completion in group:
            parsed = _safe_parse(p, "", completion)
            answers.append(
                "\x00unparsed" if parsed is None
                else str(parsed.proposed_answer)
            )
        modal = max(set(answers), key=answers.count)
        scores.append(answers.count(modal) / len(answers))
    return _mean(scores)


# --- internals --------------------------------------------------------------
def _evaluate_with_verifier(
    samples: Sequence[EvalSample],
    completions: Sequence[str],
    *,
    verifier: VerifierLike,
    shaper: RewardShaper,
    parser: CompletionParser,
    floor: float,
) -> EvalMetrics:
    parsed = [_safe_parse(parser, s.prompt, c) for s, c in zip(samples, completions)]
    reports = verifier.verify_batch(parsed)
    records: list[_Record] = []
    for sample, pc, report in zip(samples, parsed, reports):
        has_gold = sample.gold is not None
        if pc is None or report is None:
            records.append(_Record(False, False, floor, {}, has_gold, False))
            continue
        try:
            criteria_gold = sample.metadata.get("criteria_gold")
            bd = shaper.score(pc, report, sample.gold, criteria_gold=criteria_gold)
        except Exception:  # shaping must never crash the eval
            logger.exception("reward shaping failed; flooring")
            records.append(_Record(True, report.proof_checked, floor, {}, has_gold, False))
            continue
        # When gold is present the shaper's ``correctness`` component is exactly
        # the gold-match signal (0.0 / 1.0) — reuse it rather than re-implement.
        correct = has_gold and bd.components.get("correctness", 0.0) >= 1.0
        records.append(_Record(
            parsed=True, certified=report.proof_checked, reward=bd.total,
            components=dict(bd.components), has_gold=has_gold, correct=correct,
        ))
    return _aggregate(records, parse_certify_known=True)


def _evaluate_with_reward_fn(
    samples: Sequence[EvalSample],
    completions: Sequence[str],
    *,
    reward_fn: RewardFunction,
    parser: CompletionParser | None,
    floor: float,
) -> EvalMetrics:
    prompts = [s.prompt for s in samples]
    metadata = [{**s.metadata, "gold": s.gold} for s in samples]
    try:
        rewards = [float(r) for r in reward_fn(prompts, list(completions), metadata)]
    except Exception:  # a foreign reward_fn broke — floor rather than raise
        logger.exception("reward_fn raised; flooring all rewards")
        rewards = [floor] * len(samples)
    if len(rewards) != len(samples):  # contract drift — pad/truncate to floor
        rewards = (rewards + [floor] * len(samples))[: len(samples)]

    records: list[_Record] = []
    for sample, completion, reward in zip(samples, completions, rewards):
        has_gold = sample.gold is not None
        pc = _safe_parse(parser, sample.prompt, completion) if parser is not None else None
        parsed_ok = pc is not None
        correct = (
            has_gold and pc is not None and pc.proposed_answer is not None
            and _norm(pc.proposed_answer) == _norm(sample.gold)
        )
        records.append(_Record(
            parsed=parsed_ok, certified=False, reward=reward,
            components={}, has_gold=has_gold, correct=correct,
        ))
    # Without a verifier we cannot certify; without a parser we cannot parse.
    return _aggregate(
        records,
        parse_certify_known=False,
        parse_known=parser is not None,
    )


def _aggregate(
    records: Sequence[_Record], *,
    parse_certify_known: bool,
    parse_known: bool | None = None,
) -> EvalMetrics:
    n = len(records)
    rewards = [r.reward for r in records]
    gold = [r for r in records if r.has_gold]

    parse_ok = parse_certify_known if parse_known is None else parse_known
    metrics = EvalMetrics(
        n=n,
        mean_reward=_mean(rewards),
        parse_rate=_mean([1.0 if r.parsed else 0.0 for r in records]) if parse_ok else None,
        certified_rate=(
            _mean([1.0 if r.certified else 0.0 for r in records])
            if parse_certify_known else None
        ),
        accuracy_vs_gold=(
            _mean([1.0 if r.correct else 0.0 for r in gold]) if gold else None
        ),
        certified_accuracy=(
            _mean([1.0 if (r.correct and r.certified) else 0.0 for r in gold])
            if (gold and parse_certify_known) else None
        ),
        components=_mean_components([r.components for r in records if r.components]),
        rewards=rewards,
    )
    return metrics


def _mean_components(dicts: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = {k for d in dicts for k in d}
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [float(d[k]) for d in dicts if k in d]
        out[k] = _mean(vals)
    return out


def _safe_parse(
    parser: CompletionParser, prompt: str, completion: str
) -> ParsedCompletion | None:
    try:
        return parser.parse(prompt, completion)
    except Exception:  # a custom parser must not sink the eval
        logger.exception("parser raised; treating as parse failure")
        return None


def _resolve_parser(verifier: VerifierLike) -> CompletionParser:
    parser = getattr(verifier, "parser", None)
    if parser is not None:
        return parser
    domain = getattr(verifier, "domain", None)
    domain_parser = getattr(domain, "parser", None)
    if domain_parser is not None:
        return domain_parser
    return JSONBlockParser()


def _resolve_shaper(verifier: VerifierLike) -> RewardShaper:
    shaper = getattr(verifier, "shaper", None)
    return shaper if shaper is not None else DefaultRewardShaper()


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _norm(v: Any) -> str:
    return str(v).strip().lower()
