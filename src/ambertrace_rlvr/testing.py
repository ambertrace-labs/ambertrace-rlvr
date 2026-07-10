"""Offline test helpers: a fake verifier and payload builders.

Tests must not hit the network. ``FakeVerifier`` swaps in for ``AmberVerifier``
with the same reward-function interface; ``make_query_result`` builds a response in
the live ``QueryExplanation`` shape so report normalisation can be exercised without
a platform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .parsers import CompletionParser, JSONBlockParser, ParsedCompletion
from .reports import AmberReport
from .rewards import DefaultRewardShaper, RewardShaper
from .verifier import RewardFunction, build_reward_function


@dataclass
class FakeVerifier:
    """A deterministic, network-free verifier for tests and offline dev.

    Supply ``report_fn`` to map a :class:`ParsedCompletion` to an
    :class:`AmberReport`, or ``raise_on_query=True`` to exercise the fail-closed
    path (the reward function must still return the floor, never raise).
    """

    report_fn: Callable[[ParsedCompletion], AmberReport] | None = None
    parser: CompletionParser = field(default_factory=JSONBlockParser)
    shaper: RewardShaper = field(default_factory=DefaultRewardShaper)
    floor: float = -1.0
    raise_on_query: bool = False

    def verify_one(self, parsed: ParsedCompletion) -> AmberReport:
        if self.raise_on_query:
            return AmberReport.floor(reason="fake_raise")
        if self.report_fn is not None:
            return self.report_fn(parsed)
        return make_report(proof_checked=True, decision=parsed.proposed_answer)

    def verify_batch(
        self, parsed: list[ParsedCompletion | None]
    ) -> list[AmberReport | None]:
        return [self.verify_one(pc) if pc is not None else None for pc in parsed]

    def as_reward_function(self) -> RewardFunction:
        return build_reward_function(self.parser, self.shaper, self.verify_batch, self.floor)


def make_report(*, proof_checked: bool = True, decision: Any = None,
                rules: list[tuple[str, bool, bool]] | None = None,
                accepted: int = 3, rejected: int = 0,
                confidence: float = 0.9) -> AmberReport:
    """Build an :class:`AmberReport` directly. ``rules`` items are
    ``(name, fired, required)``."""
    from .reports import FiredRule
    fired_rules = [FiredRule(name=n, fired=f, required=r) for n, f, r in (rules or [])]
    return AmberReport(
        proof_checked=proof_checked, confidence=confidence,
        symbolic_confidence=1.0, neural_confidence=confidence, answer=decision,
        decision=decision, rules=fired_rules, rejected_facts=[],
        fact_summary={"accepted": accepted, "emitted": accepted + rejected,
                      "rejected": rejected, "witness_invalid": 0},
        deciding_rules=[], proof_summary="", schema_version=1, raw={},
    )


def make_query_result(*, decision: str = "permit", proof_checked: bool = True,
                      rules: list[dict[str, Any]] | None = None,
                      accepted: int = 2, rejected: int = 0) -> dict[str, Any]:
    """Build a raw ``platforms.query`` result in the live ``QueryExplanation`` shape."""
    default_rules = rules or [
        {"rule_id": 1, "rule_name": "Check A", "rule_type": "constraint",
         "action_type": "derive", "fired": True, "required": False,
         "explanation": "Rule 'Check A' fired"},
        {"rule_id": 2, "rule_name": "Deny B", "rule_type": "constraint",
         "action_type": None, "fired": False, "required": True,
         "explanation": "Rule 'Deny B' did not match context"},
    ]
    return {
        "answer": f"Decision: {decision}",
        "decision": decision,
        "platform_id": 1,
        "query": "test query",
        "proof_checked": proof_checked,
        "proof_summary": "certified in test",
        "vocabulary_declared": False,
        "explanation": {
            "schema_version": 1,
            "symbolic_trace": {
                "description": "Rules evaluated against the query context",
                "rules_evaluated": len(default_rules),
                "rules_fired": sum(1 for r in default_rules if r["fired"]),
                "rules": default_rules,
            },
            "certified_fact_summary": {
                "accepted": accepted, "emitted": accepted + rejected,
                "rejected": rejected, "witness_invalid": 0,
            },
            "certified_facts": [{"field": "f1", "value": 1, "confidence": 1.0,
                                 "schema_ok": True, "witness_invalid": False,
                                 "reasons": [], "source": "client"}],
            "confidence": {"overall": 0.88, "neural_confidence": 0.69,
                           "symbolic_confidence": 1.0, "neural_weight": 0.4,
                           "symbolic_weight": 0.6},
            "decision": {"decision": decision, "deciding_rules": []},
            "proof": {"derived": [], "facts": {}, "firings": []},
        },
    }
