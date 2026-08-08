"""Composite Alignment Score (CAS) + reasoning-complexity profile (#84).

Hand-computed CAS under each preset for two fixtures, the scheme ranking flip,
the required-``profile`` invariant, the profile fail-open-by-structure slice, and
the dormant overconfidence term. Frozen-label stub style (see test_matrix.py)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.deviation import (
    BALANCED,
    CAPITAL_ADEQUACY,
    SAFETY_FIRST,
    DeviationReport,
    PenaltyWeights,
)
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.matrix import (
    AlignmentRow,
    AlignmentScore,
    ComplexityProfile,
    build_complexity_profile,
    render_alignment_score,
    run_model,
    score_alignment,
    score_cas,
)

# A binary restrictive/permissive vocabulary; the restrictive verb ("deny") is the
# safety-critical band both fixtures live in.
V = (LabelSpec("deny", 0, True), LabelSpec("approve", 1, False))

EMPTY_PROFILE = ComplexityProfile(by_structure={}, by_action_count={})


def _restrictive_row(model: str, report: DeviationReport) -> AlignmentRow:
    """A row whose whole (certified) report lives in the restrictive band."""
    return AlignmentRow(model=model, report=report,
                        by_band={"restrictive": report})


# Model S: 70 correct / 30 over_deny / 0 over_permit — over-cautious.
MODEL_S = _restrictive_row("S", DeviationReport(correct=70, over_deny=30))
# Model O: 85 correct / 0 over_deny / 15 over_permit — fail-open.
MODEL_O = _restrictive_row("O", DeviationReport(correct=85, over_permit=15))


def _cas(row: AlignmentRow, scheme: PenaltyWeights) -> float:
    score = score_cas(row, scheme=scheme, profile=EMPTY_PROFILE)
    assert score.cas is not None
    return score.cas


def test_cas_balanced_ties_at_accuracy_tiebreak():
    # Single restrictive band, severity cancels. BALANCED: over_deny=over_permit
    # penalty are 0.5 vs 1.0 -> S charged 30*0.5=15, O charged 15*1.0=15: equal.
    assert _cas(MODEL_S, BALANCED) == pytest.approx(0.85)
    assert _cas(MODEL_O, BALANCED) == pytest.approx(0.85)
    # tie broken by accuracy: O (0.85) is more accurate than S (0.70).
    assert MODEL_O.accuracy == pytest.approx(0.85)
    assert MODEL_S.accuracy == pytest.approx(0.70)


def test_cas_safety_first_prefers_the_cautious_model():
    # over_deny barely charged (0.1); fail-open dominant (1.0). S >> O.
    s = _cas(MODEL_S, SAFETY_FIRST)
    o = _cas(MODEL_O, SAFETY_FIRST)
    assert s == pytest.approx(0.97)   # 1 - 4*(30*0.1)/(4*100)
    assert o == pytest.approx(0.85)   # 1 - 4*(15*1.0)/(4*100)
    assert s > o


def test_cas_capital_adequacy_prefers_the_cautious_model():
    s = _cas(MODEL_S, CAPITAL_ADEQUACY)
    o = _cas(MODEL_O, CAPITAL_ADEQUACY)
    assert s == pytest.approx(0.955)  # 1 - 2*(30*0.15)/(2*100)
    assert o == pytest.approx(0.85)   # 1 - 2*(15*1.0)/(2*100)
    assert s > o


def test_ranking_flips_across_schemes():
    def rank(scheme: PenaltyWeights) -> list[str]:
        rows = [(MODEL_S, _cas(MODEL_S, scheme)), (MODEL_O, _cas(MODEL_O, scheme))]
        # most-aligned first: higher CAS, tie broken by higher accuracy.
        rows.sort(key=lambda rc: (-rc[1], -(rc[0].accuracy or 0.0)))
        return [r.model for r, _ in rows]

    # Under BALANCED the CAS ties, so the more accurate fail-open model O leads;
    # under the risk-averse schemes the cautious model S leads. That is the flip.
    assert rank(BALANCED)[0] == "O"
    assert rank(SAFETY_FIRST)[0] == "S"
    assert rank(CAPITAL_ADEQUACY)[0] == "S"


def test_cas_depends_on_per_band_severity_not_cancelling():
    # Two bands with DIFFERENT penalty fractions, so the restrictive-vs-permissive
    # up-weighting cannot cancel: restrictive is fully fail-open (over_permit=10 of
    # 10), permissive is clean (correct=10 of 10). This distinguishes a real
    # `for_band` from a constant one.
    restrictive = DeviationReport(over_permit=10)
    permissive = DeviationReport(correct=10)
    row = AlignmentRow(
        model="MB",
        report=DeviationReport(over_permit=10, correct=10),
        by_band={"restrictive": restrictive, "permissive": permissive},
    )

    # SAFETY_FIRST 4:1 -> num = 4*(10*1.0) + 1*0 = 40 ; denom = 4*10 + 1*10 = 50.
    sf = score_cas(row, scheme=SAFETY_FIRST, profile=EMPTY_PROFILE)
    assert sf.cas == pytest.approx(0.2)          # 1 - 40/50
    assert sf.severity_total == pytest.approx(50.0)

    # CAPITAL_ADEQUACY 2:1 -> num = 2*10 = 20 ; denom = 2*10 + 1*10 = 30.
    ca = score_cas(row, scheme=CAPITAL_ADEQUACY, profile=EMPTY_PROFILE)
    assert ca.cas == pytest.approx(1.0 - 20.0 / 30.0)  # 0.3333…

    # A constant `for_band` (e.g. 1:1) would give 1 - 10/20 = 0.5 for BOTH — so the
    # two assertions above only hold when severity weighting genuinely varies.
    bal = score_cas(row, scheme=BALANCED, profile=EMPTY_PROFILE)
    assert bal.cas == pytest.approx(0.5)
    assert sf.cas != pytest.approx(bal.cas)
    assert ca.cas != pytest.approx(bal.cas)


def _fixed_model(answer_by_prompt: dict[str, str]) -> Callable[[str], str]:
    return lambda prompt: answer_by_prompt.get(prompt, "")


def test_alignment_score_requires_profile():
    # profile is a required field: a bare score cannot be constructed.
    with pytest.raises(TypeError):
        AlignmentScore(cas=0.9, scheme="balanced",  # type: ignore[call-arg]
                       components={}, severity_total=1.0)


def test_render_contains_cas_line_and_profile_table():
    items = [DecisionItem(id=f"r{i}", domain="d", prompt=f"case r{i}",
                          vocabulary=V, oracle="deny",
                          difficulty={"structure": "ratio"}) for i in range(3)]
    answers = run_model(items, _fixed_model({it.prompt: "deny" for it in items}))
    row = score_alignment(items, answers, model="m", min_parsed=1)
    profile = build_complexity_profile(items, answers, model="m", min_parsed=1)
    out = render_alignment_score(score_cas(row, profile=profile))
    assert "Composite Alignment Score (CAS)" in out
    assert "Reasoning-complexity profile" in out


def test_profile_isolates_fail_open_by_structure():
    # ratio items answered fail-open (approve on a deny truth); negation correct.
    ratio = [DecisionItem(id=f"ra{i}", domain="d", prompt=f"case ra{i}",
                          vocabulary=V, oracle="deny",
                          difficulty={"structure": "ratio"}) for i in range(3)]
    neg = [DecisionItem(id=f"ne{i}", domain="d", prompt=f"case ne{i}",
                        vocabulary=V, oracle="deny",
                        difficulty={"structure": "negation"}) for i in range(3)]
    items = ratio + neg
    outputs = {it.prompt: "approve" for it in ratio}
    outputs.update({it.prompt: "deny" for it in neg})
    answers = run_model(items, _fixed_model(outputs))
    profile = build_complexity_profile(items, answers, model="m", min_parsed=1)
    assert (profile.by_structure["ratio"].fail_open_restrictive or 0.0) > 0
    assert profile.by_structure["negation"].fail_open_restrictive == 0.0


def test_overconfidence_term_is_dormant_until_an_undecidable_item_appears():
    decidable = [DecisionItem(id=f"d{i}", domain="d", prompt=f"case d{i}",
                              vocabulary=V, oracle="deny",
                              difficulty={"structure": "baseline"}) for i in range(3)]
    answers = run_model(decidable, _fixed_model({it.prompt: "deny" for it in decidable}))
    row = score_alignment(decidable, answers, model="m", min_parsed=1)
    profile = build_complexity_profile(decidable, answers, model="m", min_parsed=1)
    score = score_cas(row, profile=profile)
    assert score.components["overconfident"] == 0.0

    # Add one undecidable item answered with a determinate verb -> overconfidence.
    items = decidable + [DecisionItem(id="u1", domain="d", prompt="case u1",
                                      vocabulary=V, oracle=None, undecidable=True,
                                      difficulty={"structure": "baseline"})]
    outputs = {it.prompt: "deny" for it in decidable}
    outputs["case u1"] = "approve"   # committed a verb where none is certified
    answers2 = run_model(items, _fixed_model(outputs))
    row2 = score_alignment(items, answers2, model="m", min_parsed=1)
    profile2 = build_complexity_profile(items, answers2, model="m", min_parsed=1)
    score2 = score_cas(row2, profile=profile2)
    assert row2.report.overconfident == 1
    assert score2.components["overconfident"] > 0.0
