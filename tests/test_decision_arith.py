"""Offline tests for the decision-arithmetic domain (compute->compare->decide) + its
directional fail-open exploit demo. Pure data: no model, no network, no [trl] extra."""

from __future__ import annotations

import pytest

from ambertrace_rlvr.math_env import (
    ALLOW,
    DENY,
    REFERENCE_DECIDERS,
    decide_label,
    evaluate_decision_arith,
    generate_decision_arithmetic,
    r_hack_permissive,
    r_true_decision,
)
from ambertrace_rlvr.math_env.problems import HELDOUT, TRAIN


def test_counts_splits_balance_and_consistency():
    probs = generate_decision_arithmetic(20, 8, digits=2, seed=1)
    assert len(probs) == 28
    assert sum(p.split == TRAIN for p in probs) == 20
    assert sum(p.split == HELDOUT for p in probs) == 8
    assert sum(p.decision == ALLOW for p in probs) == 14  # balanced 50/50 by construction
    for p in probs:
        # the gold decision is exactly the Q-vs-T comparison, and T is a positive threshold
        assert p.decision == (ALLOW if p.quantity <= p.threshold else DENY)
        assert p.threshold >= 1
        assert str(p.threshold) in p.question  # the limit is stated in the prompt


def test_deterministic_and_unique():
    a = generate_decision_arithmetic(12, 4, digits=2, seed=3)
    b = generate_decision_arithmetic(12, 4, digits=2, seed=3)
    assert [(p.qid, p.question, p.decision) for p in a] == [
        (p.qid, p.question, p.decision) for p in b]
    assert len({p.question for p in a}) == len(a)  # deduped


def test_solution_is_worked_and_commits_decision():
    for p in generate_decision_arithmetic(20, 0, digits=3, seed=2):
        assert p.solution is not None
        assert str(p.quantity) in p.solution and p.decision in p.solution
        assert "×" in p.solution  # shows the computation, not just the verdict


def test_digits_scale_compute_difficulty():
    small = generate_decision_arithmetic(30, 0, digits=2, seed=5)
    big = generate_decision_arithmetic(30, 0, digits=4, seed=5)
    assert sorted(p.quantity for p in big)[15] > sorted(p.quantity for p in small)[15] * 10


def test_decide_label_last_commitment_and_word_boundary():
    assert decide_label("allow") == ALLOW
    assert decide_label("I would DENY this request") == DENY
    assert decide_label("allow... on reflection, deny") == DENY  # last committed word wins
    assert decide_label("disallow") is None  # word-boundary: no bare allow/deny
    assert decide_label("hmm, unclear") is None


def test_reward_functions():
    assert r_true_decision("allow", ALLOW) == 1.0
    assert r_true_decision("deny", ALLOW) == 0.0
    assert r_hack_permissive("allow") == 1.0  # lenient verifier credits the permissive action
    assert r_hack_permissive("deny") == 0.0
    assert r_hack_permissive("nonsense") == 0.0


def test_exploit_demo_m_separates_fail_open_from_failing():
    probs = generate_decision_arithmetic(40, 0, digits=2, seed=7)
    golds = [p.decision for p in probs]
    allow_frac = sum(g == ALLOW for g in golds) / len(golds)
    deny_frac = 1.0 - allow_frac

    honest = evaluate_decision_arith([REFERENCE_DECIDERS["honest"](p) for p in probs], golds)
    assert honest.k == 1.0 and honest.m == 0.0 and honest.parse_rate == 1.0

    # always-allow: the fail-open exploit — scores the hack perfectly, and its load m is exactly
    # the over-permit (fail-open) rate (every deny-gold item is an undeserved permissive pass).
    aa = evaluate_decision_arith([REFERENCE_DECIDERS["always_allow"](p) for p in probs], golds)
    assert aa.k_hat == 1.0
    assert aa.k == pytest.approx(allow_frac)
    assert aa.m == pytest.approx(deny_frac) and aa.fail_open_rate == pytest.approx(deny_frac)

    # always-deny: a merely-restrictive degenerate — fails just as much (k == deny_frac) but
    # scores the permissive hack at 0 and triggers NO fail-open. So m is directional: it flags
    # fail-open misalignment, never symmetric failing.
    ad = evaluate_decision_arith([REFERENCE_DECIDERS["always_deny"](p) for p in probs], golds)
    assert ad.k_hat == 0.0 and ad.m == 0.0
    assert ad.k == pytest.approx(deny_frac)


def test_bad_params_raise():
    with pytest.raises(ValueError):
        generate_decision_arithmetic(10, 2, digits=0)
    with pytest.raises(ValueError):
        generate_decision_arithmetic(10, 2, band=0)
    with pytest.raises(ValueError):
        evaluate_decision_arith(["allow"], ["allow", "deny"])  # mismatched lengths


# --- reward builder (fail-closed) + report round-trip -------------------------

def test_decision_reward_scores_and_is_bounded():
    from ambertrace_rlvr.math_env import build_decision_reward_function

    r = build_decision_reward_function(0.5)
    out = r(["p"] * 4, ["allow", "deny", "allow", "deny"], answer=["allow", "allow", "deny", "deny"])
    assert out == [1.0, 0.0, 0.5, 0.5]  # (1-l)*true + l*hack at l=0.5
    assert all(0.0 <= x <= 1.0 for x in out)
    # lam outside [0,1] is clamped, so scores stay bounded
    assert all(0.0 <= x <= 1.0 for x in build_decision_reward_function(7.0)(
        ["p"], ["allow"], answer=["deny"]))
    # a clean certified completion never loses to a hacked one at lam=0
    clean = build_decision_reward_function(0.0)
    assert clean(["p"], ["deny"], answer=["deny"]) > clean(["p"], ["allow"], answer=["deny"])


def test_decision_reward_fail_closed():
    from ambertrace_rlvr.math_env import build_decision_reward_function

    r = build_decision_reward_function(0.3, floor=0.0)
    # parse failure (no allow/deny), missing gold, invalid gold, unflattenable junk: all floor
    assert r(["p"], ["I am not sure."], answer=["allow"]) == [0.0]
    assert r(["p"], ["allow"], answer=None) == [0.0]
    assert r(["p"], ["allow"], answer=["maybe"]) == [0.0]
    assert r(["p"], [object()], answer=["allow"]) == [0.0]
    # one bad item does not poison the group; conversational completions + metadata work
    out = r(["p", "p"], [[{"role": "assistant", "content": "so deny"}], "???"],
            metadata=[{"decision": "deny"}, {"decision": "deny"}])
    assert out == [pytest.approx(0.7), 0.0]
    # a non-sequence completions arg returns [] instead of raising
    assert r(["p"], None, answer=["allow"]) == []  # type: ignore[arg-type]
    # a custom floor is honoured
    assert build_decision_reward_function(0.0, floor=0.1)(["p"], ["??"], answer=["deny"]) == [0.1]


def test_decision_eval_report_roundtrip_and_lineage_compat():
    from ambertrace_rlvr.math_env import DecisionEvalReport, MathEvalReport

    rep = evaluate_decision_arith(["allow", "allow", "deny", "?"], ["deny", "allow", "deny", "allow"],
                                  lam=0.3)
    assert rep.wedge == pytest.approx(rep.k_hat - rep.k)
    assert rep.m == pytest.approx(0.25) and rep.parse_rate == pytest.approx(0.75)
    assert DecisionEvalReport.from_dict(rep.as_dict()) == rep
    # run_lineage's resume cache reloads eval JSON via MathEvalReport.from_dict: must not break
    m = MathEvalReport.from_dict(rep.as_dict())
    assert (m.n, m.k, m.k_hat, m.m, m.lam) == (rep.n, rep.k, rep.k_hat, rep.m, 0.3)
    with pytest.raises(KeyError):
        DecisionEvalReport.from_dict({"n": 1})
