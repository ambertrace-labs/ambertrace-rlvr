"""Reward math: bounds, monotonicity, floor, and the end-to-end fake path."""

from __future__ import annotations

from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.rewards import DefaultRewardShaper, SubstringProvenanceChecker
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


# --- graded: per-criterion partial credit (#9) -----------------------------

# Three required criteria; the report's certified firings are (True, True, False).
def _report_three_required(firings: tuple[bool, bool, bool]):
    return make_report(
        decision="permit",
        rules=[("PVS1", firings[0], True),
               ("PS3", firings[1], True),
               ("BA1", firings[2], True)],
    )


def test_graded_per_criterion_full_partial_zero():
    shaper = DefaultRewardShaper()
    report = _report_three_required((True, True, False))
    # gold expects exactly the certified firings -> all three correct.
    full = shaper.score(PARSED, report,
                        criteria_gold={"PVS1": True, "PS3": True, "BA1": False})
    # one wrong (BA1 expected True but did not fire) -> 2/3 correct.
    partial = shaper.score(PARSED, report,
                           criteria_gold={"PVS1": True, "PS3": True, "BA1": True})
    # all three wrong -> 0/3.
    zero = shaper.score(PARSED, report,
                        criteria_gold={"PVS1": False, "PS3": False, "BA1": True})
    assert full.components["graded"] == 1.0
    assert partial.components["graded"] == 2 / 3
    assert zero.components["graded"] == 0.0


def test_graded_per_criterion_is_monotonic():
    shaper = DefaultRewardShaper()
    report = _report_three_required((True, True, False))
    # progressively align gold with the certified firings: 1, 2, then 3 correct.
    one = shaper.score(PARSED, report,
                       criteria_gold={"PVS1": True, "PS3": False, "BA1": True})
    two = shaper.score(PARSED, report,
                       criteria_gold={"PVS1": True, "PS3": True, "BA1": True})
    three = shaper.score(PARSED, report,
                         criteria_gold={"PVS1": True, "PS3": True, "BA1": False})
    g = [r.components["graded"] for r in (one, two, three)]
    assert g == sorted(g) and g[0] < g[2]
    # and per-criterion credit never lets a partial out-score the fully-correct one.
    assert three.total >= two.total >= one.total


def test_graded_absent_required_field_degrades_to_zero_weight():
    shaper = DefaultRewardShaper()
    # criteria_gold supplied, but no rule is marked required -> zero-weight, no crash.
    report = make_report(decision="permit",
                         rules=[("PVS1", True, False), ("PS3", True, False)])
    b = shaper.score(PARSED, report, criteria_gold={"PVS1": True, "PS3": True})
    assert b.components["graded"] == 0.0


def test_graded_falls_back_to_baseline_without_criteria_gold():
    shaper = DefaultRewardShaper()
    # 2 of 3 rules fired; no criteria_gold -> baseline fired/evaluated = 2/3.
    report = make_report(decision="permit",
                         rules=[("A", True, False), ("B", True, False), ("C", False, False)])
    b = shaper.score(PARSED, report)
    assert b.components["graded"] == 2 / 3


def test_graded_per_criterion_end_to_end_via_fake_verifier():
    report = _report_three_required((True, True, False))
    fv = FakeVerifier(parser=JSONBlockParser(), report_fn=lambda pc: report)
    reward_fn = fv.as_reward_function()
    # criteria_gold rides in per-sample metadata, same channel as gold.
    right = reward_fn(["p"], [_completion("permit")],
                      [{"criteria_gold": {"PVS1": True, "PS3": True, "BA1": False}}])
    wrong = reward_fn(["p"], [_completion("permit")],
                      [{"criteria_gold": {"PVS1": False, "PS3": False, "BA1": True}}])
    assert right[0] > wrong[0]


# --- fact-provenance / anti-reward-hacking (#10) ---------------------------

def test_unsupported_facts_score_below_grounded():
    shaper = DefaultRewardShaper(provenance=SubstringProvenanceChecker())
    report = make_report(decision="permit")
    grounded = ParsedCompletion(query="q", facts={"income": 25000},
                                proposed_answer="permit",
                                prompt="applicant income is 25000 per year")
    invented = ParsedCompletion(query="q", facts={"income": 99999},
                                proposed_answer="permit",
                                prompt="applicant income is 25000 per year")
    hi = shaper.score(grounded, report).total
    lo = shaper.score(invented, report).total
    assert lo < hi
    assert shaper.score(grounded, report).components["unsupported_penalty"] == 0.0
    assert shaper.score(invented, report).components["unsupported_penalty"] == 1.0


def test_hallucinated_fact_never_outscores_clean_certified():
    shaper = DefaultRewardShaper(provenance=SubstringProvenanceChecker())
    clean = ParsedCompletion(query="q", facts={"income": 25000},
                             proposed_answer="permit",
                             prompt="applicant income is 25000 per year")
    hallucinated = ParsedCompletion(query="q", facts={"income": 99999},
                                    proposed_answer="permit",
                                    prompt="applicant income is 25000 per year")
    clean_score = shaper.score(clean, make_report(decision="permit"), gold="permit").total
    hall_score = shaper.score(hallucinated,
                              make_report(decision="permit"), gold="permit").total
    assert hall_score <= clean_score


def test_substring_checker_numeric_boundary():
    checker = SubstringProvenanceChecker()
    # fabricated 5000 must not be grounded by a genuine 25000 in the prompt.
    assert checker.unsupported_fraction({"x": 5000}, "income is 25000") == 1.0
    assert checker.unsupported_fraction({"x": 25000}, "income is 25000") == 0.0


def test_substring_checker_string_match():
    checker = SubstringProvenanceChecker()
    prompt = "The loan type is unsecured."
    assert checker.unsupported_fraction({"loan": "unsecured"}, prompt) == 0.0
    assert checker.unsupported_fraction({"loan": "secured mortgage"}, prompt) == 1.0


def test_substring_checker_booleans_exempt_by_default():
    checker = SubstringProvenanceChecker()
    # booleans are not checkable by default -> nothing in the denominator -> 0.0
    assert checker.unsupported_fraction({"flag": True}, "no keywords here") == 0.0
    strict = SubstringProvenanceChecker(check_booleans=True)
    assert strict.unsupported_fraction({"flag": True}, "the answer is true") == 0.0
    assert strict.unsupported_fraction({"flag": True}, "no keywords here") == 1.0


def test_substring_checker_no_checkable_facts_is_zero():
    checker = SubstringProvenanceChecker()
    # None / empty-string / bool: none checkable -> 0.0, not a false penalty.
    assert checker.unsupported_fraction({"a": None, "b": "", "c": True}, "prompt") == 0.0


def test_unsupported_penalty_is_zero_when_provenance_none():
    # Regression guard: default shaper logs the key but never penalises.
    shaper = DefaultRewardShaper()
    parsed = ParsedCompletion(query="q", facts={"income": 99999},
                              proposed_answer="permit",
                              prompt="income is 25000")
    b = shaper.score(parsed, make_report(decision="permit"))
    assert "unsupported_penalty" in b.components
    assert b.components["unsupported_penalty"] == 0.0


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
