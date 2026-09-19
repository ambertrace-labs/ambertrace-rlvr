"""AmberTrace decision contestant (#123): case-fact parsing and the certified ->
ScoredDecision mapping, incl. fail-closed refusals. Network-free (injected
query_fn); the live SDK glue (from_platform / build_verified_platform) is not
exercised here."""

from __future__ import annotations

from typing import Any

from ambertrace_rlvr import AmberTraceDecider, parse_case_facts
from ambertrace_rlvr.model_backend import ScoredDecision
from ambertrace_rlvr.reports import AmberReport

PROMPT = (
    "You are the decision-maker for the following policy domain.\n\n"
    "This is a loan approval domain. Applications are ...\n\n"
    "Case facts:\n"
    "- monthly_income: 3917.44\n"
    "- monthly_payment: 870.18\n"
    "- credit_score: 813\n"
    "- employment_status: self_employed\n\n"
    "Choose exactly one action from: approve, deny.\n"
    "Respond with only the chosen action.\n"
)


def test_parse_case_facts_coerces_types_and_stops_at_the_choice_line():
    facts = parse_case_facts(PROMPT)
    assert facts == {
        "monthly_income": 3917.44,     # float
        "monthly_payment": 870.18,
        "credit_score": 813,           # int
        "employment_status": "self_employed",   # string, untouched
    }


def _certified(decision: str, overall: float = 0.95) -> AmberReport:
    return AmberReport.from_query_result({
        "proof_checked": True, "decision": decision,
        "explanation": {"confidence": {"overall": overall}},
    })


def test_certified_decision_in_label_space_is_returned():
    seen: dict[str, Any] = {}

    def query_fn(facts: dict[str, Any]) -> AmberReport:
        seen["facts"] = facts
        return _certified("deny")

    d = AmberTraceDecider(query_fn=query_fn).decide(PROMPT, ("approve", "deny"))
    assert seen["facts"]["credit_score"] == 813        # parsed facts reached the query
    assert d == ScoredDecision(choice="deny", probabilities={"deny": 1.0}, confidence=0.95)


def test_fail_closed_report_is_a_refusal():
    # a verified query that could not certify -> floor report -> refusal, not a verb.
    d = AmberTraceDecider(query_fn=lambda facts: AmberReport.floor(reason="verified_fail_closed"))
    ans = d.decide(PROMPT, ("approve", "deny"))
    assert ans.choice is None and not ans.answered


def test_non_certified_answer_is_a_refusal():
    report = AmberReport.from_query_result({"proof_checked": False, "decision": "approve"})
    ans = AmberTraceDecider(query_fn=lambda f: report).decide(PROMPT, ("approve", "deny"))
    assert ans.choice is None


def test_certified_verb_outside_label_space_is_a_refusal():
    # a decision the corpus never offered must not be scored as a (wrong) label.
    ans = AmberTraceDecider(query_fn=lambda f: _certified("escalate")).decide(
        PROMPT, ("approve", "deny"))
    assert ans.choice is None
