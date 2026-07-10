"""Report normalisation against the recorded SDK payload + fail-closed paths."""

from __future__ import annotations

from ambertrace_rlvr.reports import AmberReport


def test_normalises_recorded_payload(loan_query_result):
    r = AmberReport.from_query_result(loan_query_result)
    assert r.proof_checked is True
    assert r.decision == "permit"
    assert r.schema_version == 1 and r.schema_supported
    assert r.confidence == 0.88
    assert r.symbolic_confidence == 1.0
    assert r.rules, "symbolic_trace.rules should normalise into FiredRule list"
    # every rule keeps its name + a bool fired flag
    assert all(isinstance(rule.name, str) and isinstance(rule.fired, bool)
               for rule in r.rules)
    assert r.fact_summary.get("accepted") == 12
    assert r.n_rejected == 0


def test_required_flag_and_fired_view(loan_query_result):
    r = AmberReport.from_query_result(loan_query_result)
    # the fixture's required rule is the deny-family underage check (not fired here)
    assert any(rule.required for rule in r.rules)
    assert all(rule.fired for rule in r.rules_fired)
    assert len(r.rules_fired) <= len(r.rules)


def test_floor_report_is_fail_closed():
    r = AmberReport.floor("boom")
    assert r.proof_checked is False
    assert r.confidence == 0.0
    assert r.rules == [] and r.rejected_facts == []
    assert r.error == "boom"


def test_from_error_pulls_structured_rejected_facts():
    class FakeErr(Exception):
        rejected_facts = [
            {"field": "loan_type", "value": "mortgage",
             "reasons": ["value 'mortgage' is outside the declared domain of 'loan_type'"]},
        ]

    r = AmberReport.from_error(FakeErr("nope"))
    assert r.proof_checked is False
    assert len(r.rejected_facts) == 1
    rf = r.rejected_facts[0]
    assert rf.field == "loan_type" and rf.value == "mortgage" and rf.reasons


def test_from_error_tolerates_legacy_string_rejected_facts():
    class LegacyErr(Exception):
        rejected_facts = ["loan_type", "loan_purpose"]  # pre-2026-07-10 shape

    r = AmberReport.from_error(LegacyErr("nope"))
    assert [rf.field for rf in r.rejected_facts] == ["loan_type", "loan_purpose"]


def test_missing_keys_degrade_not_crash():
    # a stripped-down / drifted response must not raise
    r = AmberReport.from_query_result({"proof_checked": True})
    assert r.proof_checked is True
    assert r.rules == [] and r.confidence == 0.0 and r.schema_version is None
