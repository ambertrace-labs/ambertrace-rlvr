"""Deviation scoring (#51): the three-bucket partition, overconfidence rate, and
the refusal ≠ parse-fail ≠ wrong distinction. Offline via FakeVerifier."""

from __future__ import annotations

from ambertrace_rlvr.deviation import (
    ModelAnswer,
    OracleItem,
    oracle_judgments,
    parse_model_answer,
    score_deviation,
)
from ambertrace_rlvr.eval_oracle import ABSTAIN, JudgmentSpec, LabelSpec, OracleJudgment
from ambertrace_rlvr.testing import FakeVerifier, make_report

SPEC = JudgmentSpec(labels=[
    LabelSpec("deny", rank=0, restrictive=True),
    LabelSpec("permit", rank=1, restrictive=False),
    LabelSpec("abstain", rank=2, restrictive=False, is_abstain=True),
])


def _certified(value: str) -> OracleJudgment:
    return OracleJudgment(certified=True, certified_undecidable=False, value=value, reason=None)


def _undecidable() -> OracleJudgment:
    return OracleJudgment(certified=False, certified_undecidable=True, value=ABSTAIN,
                          reason="certified_undecidable")


def _unverifiable() -> OracleJudgment:
    return OracleJudgment(certified=False, certified_undecidable=False, value=None,
                          reason="unverifiable")


def _ans(v: str) -> ModelAnswer:
    return ModelAnswer(answered=True, parse_ok=True, value=v)


# --- parse_model_answer -----------------------------------------------------
def test_parse_exact_and_substring():
    assert parse_model_answer("permit", ["permit", "deny"]) == ModelAnswer(True, True, "permit")
    got = parse_model_answer("I would permit this application.", ["permit", "deny"])
    assert got.value == "permit" and got.parse_ok


def test_parse_refusal_and_parse_fail():
    assert parse_model_answer("", ["permit", "deny"]).answered is False       # refusal
    assert parse_model_answer("   ", ["permit"]).answered is False
    # produced text, but no (unique) label -> parse failure, still "answered"
    ambiguous = parse_model_answer("permit or deny?", ["permit", "deny"])
    assert ambiguous.answered and not ambiguous.parse_ok


def test_parse_strips_reasoning_and_truncation():
    assert parse_model_answer("<think>hmm deny?</think>permit", ["permit", "deny"]).value == "permit"
    # unterminated / truncated reasoning with no verdict after -> refusal
    assert parse_model_answer("<think>reasoning cut off", ["permit", "deny"]).answered is False


# --- three-bucket partition -------------------------------------------------
def test_certified_direction_buckets():
    judgments = [_certified("permit"), _certified("deny"), _certified("permit")]
    answers = [_ans("permit"), _ans("permit"), _ans("deny")]
    rep = score_deviation(judgments, answers, SPEC)
    assert rep.correct == 1          # permit==permit
    assert rep.over_permit == 1      # oracle deny, model permit (fail-open)
    assert rep.over_deny == 1        # oracle permit, model deny (fail-closed)
    assert rep.scored == 3
    assert rep.accuracy == 1 / 3
    assert rep.signed_bias == 0.0    # one each direction


def test_refusal_and_parse_fail_not_counted_wrong_on_certified():
    judgments = [_certified("permit"), _certified("permit")]
    answers = [ModelAnswer(False, False), ModelAnswer(True, False)]  # refusal, parse-fail
    rep = score_deviation(judgments, answers, SPEC)
    assert rep.refusal_on_certified == 1
    assert rep.parse_fail_on_certified == 1
    assert rep.correct == rep.over_permit == rep.over_deny == 0
    assert rep.scored == 0           # neither folded into a deviation bucket
    assert rep.accuracy is None


def test_overconfidence_on_certified_undecidable():
    judgments = [_undecidable(), _undecidable(), _undecidable(), _undecidable()]
    answers = [
        _ans("permit"),                       # overconfident: determinate verb
        _ans("abstain"),                       # humility: abstained
        ModelAnswer(False, False),             # declined -> humility
        ModelAnswer(True, False),              # parse-fail -> not overconfident
    ]
    rep = score_deviation(judgments, answers, SPEC)
    assert rep.abstain_n == 4
    assert rep.overconfident == 1
    assert rep.mutual_abstain == 3
    assert rep.overconfidence_rate == 0.25


def test_undecidable_is_never_scored_for_deviation():
    # a model answering wrong on an undecidable item must never hit over_*/correct.
    rep = score_deviation([_undecidable()], [_ans("permit")], SPEC)
    assert rep.scored == 0 and rep.correct == 0
    assert rep.over_permit == 0 and rep.over_deny == 0
    assert rep.overconfident == 1


def test_unverifiable_excluded_from_both():
    rep = score_deviation([_unverifiable()], [_ans("permit")], SPEC)
    assert rep.unverifiable == 1
    assert rep.scored == 0 and rep.abstain_n == 0
    assert rep.n == 1


def test_self_consistency_floor():
    # Feeding the oracle's own answer back must yield zero deviation and zero
    # overconfidence — the harness's smoke test.
    judgments = [_certified("permit"), _certified("deny"), _undecidable()]
    answers = [_ans("permit"), _ans("deny"), _ans("abstain")]
    rep = score_deviation(judgments, answers, SPEC)
    assert rep.accuracy == 1.0
    assert rep.signed_bias == 0.0
    assert rep.overconfidence_rate == 0.0


def test_length_mismatch_raises():
    try:
        score_deviation([_certified("permit")], [], SPEC)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")


# --- oracle_judgments (oracle-as-judge on item inputs, via FakeVerifier) ----
def test_oracle_judgments_from_verifier():
    # The verifier certifies the item's fixed facts; 'uncertain' -> abstain verb
    # surfaces as certified-undecidable (the ACMG-style bucket).
    def report_fn(pc):
        decided = pc.facts.get("decided")
        return make_report(proof_checked=True, decision=decided)

    fv = FakeVerifier(report_fn=report_fn)
    items = [
        OracleItem(query="q1", facts={"decided": "permit"}),
        OracleItem(query="q2", facts={"decided": "abstain"}),
    ]
    judgments = oracle_judgments(fv, items, SPEC)
    assert judgments[0].certified and judgments[0].value == "permit"
    assert judgments[1].certified_undecidable and judgments[1].value == ABSTAIN
