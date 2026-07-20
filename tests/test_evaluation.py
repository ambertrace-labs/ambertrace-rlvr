"""Evaluation harness: metrics, baselines, and the offline fake-eval path."""

from __future__ import annotations

import json

from ambertrace_rlvr.evaluation import (
    EvalMetrics,
    EvalSample,
    compare_to_baseline,
    constant_policy,
    consistency,
    evaluate,
    evaluate_policy,
    malformed_policy,
    run_policy,
)
from ambertrace_rlvr.parsers import JSONBlockParser
from ambertrace_rlvr.testing import FakeVerifier, make_report

PARSER = JSONBlockParser()


def _completion(answer: str, facts: dict | None = None) -> str:
    block = json.dumps({"classification": answer, "facts": facts or {"a": 1}})
    return f"<decision>{block}</decision>"


def _samples(*golds: str) -> list[EvalSample]:
    return [EvalSample(prompt=f"p{i}", gold=g) for i, g in enumerate(golds)]


# A verifier that certifies iff the proposed answer is "permit", echoing the
# proposed answer as the certified decision.
def _selective_verifier() -> FakeVerifier:
    return FakeVerifier(
        parser=PARSER,
        report_fn=lambda pc: make_report(
            proof_checked=(pc.proposed_answer == "permit"),
            decision=pc.proposed_answer,
        ),
    )


# --- core metrics -----------------------------------------------------------
def test_all_well_formed_and_certified():
    samples = _samples("permit", "permit")
    completions = [_completion("permit"), _completion("permit")]
    m = evaluate(samples, completions, verifier=_selective_verifier())
    assert m.n == 2
    assert m.parse_rate == 1.0
    assert m.certified_rate == 1.0
    assert m.accuracy_vs_gold == 1.0
    assert m.certified_accuracy == 1.0
    # component means are exposed (the reward-component traces).
    for key in ("format", "certified", "correctness", "graded", "rejected_penalty"):
        assert key in m.components


def test_parse_rate_reflects_malformed_completions():
    samples = _samples("permit", "permit", "permit")
    completions = [_completion("permit"), "garbage, no block", _completion("permit")]
    m = evaluate(samples, completions, verifier=_selective_verifier())
    assert m.parse_rate == 2 / 3
    # the malformed one is neither certified nor correct.
    assert m.certified_rate == 2 / 3
    assert m.accuracy_vs_gold == 2 / 3


def test_certified_rate_reflects_uncertified_completions():
    samples = _samples("permit", "deny")
    # "deny" parses but the selective verifier won't certify it.
    completions = [_completion("permit"), _completion("deny")]
    m = evaluate(samples, completions, verifier=_selective_verifier())
    assert m.parse_rate == 1.0
    assert m.certified_rate == 0.5


def test_accuracy_vs_gold_counts_only_matches():
    samples = _samples("permit", "deny")
    completions = [_completion("permit"), _completion("permit")]  # 2nd is wrong vs gold
    m = evaluate(samples, completions, verifier=_selective_verifier())
    assert m.accuracy_vs_gold == 0.5
    # correct AND certified: only the first (permit==permit, certified).
    assert m.certified_accuracy == 0.5


def test_accuracy_is_none_without_gold():
    samples = [EvalSample(prompt="p0"), EvalSample(prompt="p1")]
    completions = [_completion("permit"), _completion("permit")]
    m = evaluate(samples, completions, verifier=_selective_verifier())
    assert m.accuracy_vs_gold is None
    assert m.certified_accuracy is None
    assert m.parse_rate == 1.0


# --- baselines --------------------------------------------------------------
def test_competent_policy_beats_degenerate_baseline():
    # Gold is "permit" everywhere; the competent policy answers correctly, the
    # degenerate constant baseline always answers "deny".
    samples = _samples("permit", "permit", "permit")
    verifier = _selective_verifier()

    def competent(prompt: str) -> str:
        return _completion("permit")

    good = evaluate_policy(competent, samples, verifier=verifier)
    degenerate = evaluate_policy(constant_policy("deny"), samples, verifier=verifier)
    malformed = evaluate_policy(malformed_policy(), samples, verifier=verifier)

    assert good.mean_reward > degenerate.mean_reward > malformed.mean_reward
    assert good.accuracy_vs_gold == 1.0
    assert degenerate.accuracy_vs_gold == 0.0
    # malformed never parses.
    assert malformed.parse_rate == 0.0
    assert good.certified_rate == 1.0

    delta = compare_to_baseline(good, degenerate)
    assert delta["mean_reward"] > 0
    assert delta["accuracy_vs_gold"] == 1.0
    assert delta["certified_rate"] > 0
    # component deltas are surfaced too.
    assert any(k.startswith("component.") for k in delta)


def test_constant_policy_is_well_formed_and_parses():
    samples = _samples("permit")
    m = evaluate_policy(constant_policy("permit"), samples, verifier=_selective_verifier())
    assert m.parse_rate == 1.0


def test_run_policy_is_fail_closed():
    def broken(prompt: str) -> str:
        raise RuntimeError("boom")

    completions = run_policy(broken, _samples("permit"))
    assert completions == [""]  # no exception; empty -> parse failure downstream


# --- reward_fn path ---------------------------------------------------------
def test_evaluate_with_reward_fn_only():
    verifier = _selective_verifier()
    reward_fn = verifier.as_reward_function()
    samples = _samples("permit", "permit")
    completions = [_completion("permit"), "no block"]
    m = evaluate(samples, completions, reward_fn=reward_fn, parser=PARSER)
    assert m.n == 2
    assert m.parse_rate == 0.5           # parser still gives parse_rate
    assert m.certified_rate is None      # no verifier -> cannot certify
    assert m.accuracy_vs_gold == 0.5     # 1st matches gold, 2nd unparsed
    assert m.rewards[0] > m.rewards[1]


def test_reward_fn_that_raises_is_floored():
    def boom(prompts, completions, metadata=None, **_):
        raise RuntimeError("bad reward fn")

    samples = _samples("permit")
    m = evaluate(samples, [_completion("permit")], reward_fn=boom, floor=-1.0)
    assert m.rewards == [-1.0]


# --- consistency ------------------------------------------------------------
def test_consistency_full_and_split():
    full = consistency([[_completion("permit"), _completion("permit")]], parser=PARSER)
    assert full == 1.0
    split = consistency([[_completion("permit"), _completion("deny")]], parser=PARSER)
    assert split == 0.5
    # a parse failure counts as its own answer, lowering agreement.
    noisy = consistency([[_completion("permit"), "garbage", "garbage"]], parser=PARSER)
    assert noisy == 2 / 3


# --- guards -----------------------------------------------------------------
def test_length_mismatch_raises():
    try:
        evaluate(_samples("permit"), [], verifier=_selective_verifier())
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")


def test_requires_verifier_or_reward_fn():
    try:
        evaluate(_samples("permit"), [_completion("permit")])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when neither source given")


def test_metrics_dataclass_shape():
    m = EvalMetrics(n=0, mean_reward=0.0)
    assert m.parse_rate is None and m.components == {} and m.rewards == []
