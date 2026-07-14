"""Reward math: bounds, monotonicity, floor, and the end-to-end fake path."""

from __future__ import annotations

from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.rewards import DefaultRewardShaper
from ambertrace_rlvr.testing import FakeVerifier, make_report

PARSED = ParsedCompletion(query="q", facts={"a": 1}, proposed_answer="permit")


def test_certified_beats_uncertified():
    shaper = DefaultRewardShaper()
    hi = shaper.score(PARSED, make_report(proof_checked=True, decision="permit")).total
    lo = shaper.score(PARSED, make_report(proof_checked=False, decision="permit")).total
    assert hi > lo


def test_rejected_facts_never_outscore_clean_certified():
    shaper = DefaultRewardShaper()
    clean = shaper.score(PARSED, make_report(decision="permit", accepted=4, rejected=0)).total
    dirty = shaper.score(PARSED, make_report(decision="permit", accepted=4, rejected=3)).total
    assert dirty <= clean


def test_correctness_uses_gold_when_present():
    shaper = DefaultRewardShaper()
    report = make_report(decision="permit")  # certified decision == "permit"
    right = shaper.score(PARSED, report, gold="permit").total
    wrong = shaper.score(PARSED, report, gold="deny").total
    assert right > wrong


def test_reward_is_bounded_by_clip():
    shaper = DefaultRewardShaper(clip=(-1.0, 2.0),
                                 weights={"format": 10, "certified": 10,
                                          "correctness": 10, "graded": 10,
                                          "rejected_penalty": 10})
    top = shaper.score(PARSED, make_report(decision="permit"), gold="permit").total
    assert top <= 2.0
    floor = shaper.score(PARSED, make_report(proof_checked=False, decision="x",
                                             accepted=0, rejected=5), gold="y").total
    assert floor >= -1.0


def test_graded_zero_when_uncertified():
    shaper = DefaultRewardShaper()
    b = shaper.score(PARSED, make_report(proof_checked=False))
    assert b.components["graded"] == 0.0


def test_components_are_logged():
    b = DefaultRewardShaper().score(PARSED, make_report(decision="permit"))
    for key in ("format", "certified", "correctness", "graded", "rejected_penalty", "total"):
        assert key in b.components


# --- end-to-end via the fake verifier (offline) ----------------------------

def _completion(answer: str) -> str:
    return f'<decision>{{"classification": "{answer}", "facts": {{"a": 1}}}}</decision>'


def test_fake_verifier_reward_function_batch():
    fv = FakeVerifier(parser=JSONBlockParser(),
                      report_fn=lambda pc: make_report(decision=pc.proposed_answer))
    reward_fn = fv.as_reward_function()
    rewards = reward_fn(["p1", "p2"], [_completion("permit"), _completion("deny")])
    assert len(rewards) == 2 and all(isinstance(r, float) for r in rewards)


def test_unparseable_completion_returns_floor():
    fv = FakeVerifier(parser=JSONBlockParser(), floor=-1.0)
    reward_fn = fv.as_reward_function()
    rewards = reward_fn(["p"], ["no decision block here"])
    assert rewards == [-1.0]


def test_verifier_error_returns_floor_not_raise():
    fv = FakeVerifier(parser=JSONBlockParser(), raise_on_query=True, floor=-1.0)
    reward_fn = fv.as_reward_function()
    # parses fine, but "verification" fails closed -> low reward, no exception
    rewards = reward_fn(["p"], [_completion("permit")])
    assert len(rewards) == 1 and rewards[0] <= 0.2
