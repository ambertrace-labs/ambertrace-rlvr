"""Oracle-as-judge seam: certified / certified-undecidable / unverifiable, plus
the per-domain direction + severity vocabulary. Offline via make_report."""

from __future__ import annotations

from ambertrace_rlvr.eval_oracle import (
    ABSTAIN,
    JudgmentSpec,
    LabelSpec,
    OracleJudgment,
)
from ambertrace_rlvr.testing import make_report

# A grant-style vocabulary: deny is the restrictive/fail-closed side, permit the
# fail-open side, abstain the certified-undecidable outcome.
SPEC = JudgmentSpec(labels=[
    LabelSpec("deny", rank=0, restrictive=True),
    LabelSpec("permit", rank=1, restrictive=False),
    LabelSpec("abstain", rank=2, restrictive=False, is_abstain=True),
])


# --- OracleJudgment.from_report --------------------------------------------
def test_certified_decidable():
    j = OracleJudgment.from_report(make_report(proof_checked=True, decision="permit"), SPEC)
    assert j.certified and not j.certified_undecidable
    assert j.value == "permit" and j.scorable


def test_certified_undecidable_is_not_scorable():
    j = OracleJudgment.from_report(make_report(proof_checked=True, decision="abstain"), SPEC)
    assert j.certified_undecidable and not j.certified
    assert j.value == ABSTAIN and not j.scorable
    assert j.reason == "certified_undecidable"


def test_unverifiable_when_no_proof():
    j = OracleJudgment.from_report(make_report(proof_checked=False, decision="permit"), SPEC)
    assert not j.certified and not j.certified_undecidable
    assert j.value is None and not j.scorable


def test_without_spec_missing_decision_is_undecidable():
    # No spec: abstention can only be inferred from a missing decision.
    j = OracleJudgment.from_report(make_report(proof_checked=True, decision=None))
    assert j.certified_undecidable and j.value == ABSTAIN


def test_credited_rules_from_fired_rules():
    report = make_report(proof_checked=True, decision="permit",
                         rules=[("R1", True, True), ("R2", False, True), ("R3", True, False)])
    j = OracleJudgment.from_report(report, SPEC)
    # only fired rules are credited (R2 did not fire).
    assert set(j.credited_rules) == {"R1", "R3"}


# --- JudgmentSpec.direction -------------------------------------------------
def test_direction_correct():
    assert SPEC.direction("permit", "permit") == "correct"


def test_direction_over_permit_is_fail_open():
    # oracle wanted the restrictive deny; model said the permissive permit.
    assert SPEC.direction("deny", "permit") == "over_permit"


def test_direction_over_deny_is_fail_closed():
    assert SPEC.direction("permit", "deny") == "over_deny"


def test_direction_unknown_biases_to_over_deny():
    # a verb absent from the vocabulary must never produce a false over_permit.
    assert SPEC.direction("deny", "mystery") == "over_deny"
    assert SPEC.direction("mystery", "permit") == "over_deny"


def test_severity_bands():
    assert SPEC.severity_band("deny") == "restrictive"
    assert SPEC.severity_band("permit") == "permissive"
    assert SPEC.severity_band("unknown") == "permissive"


def test_is_abstain():
    assert SPEC.is_abstain("abstain") and not SPEC.is_abstain("permit")
