"""Reward-hacking perturbation probes (#11): the rule-preserving vs rule-violating
reward gap, and the ``gamed`` anti-hacking regression. Fully offline (FakeVerifier).

The probe sets carry no gold — the signal is whether the *verifier* certifies (and
so rewards) a rule-violating completion, not whether it matches a label. A sound
verifier refuses to certify the violating set, so its reward floors and the gap is
large; a gameable verifier certifies it too, collapsing the gap and tripping
``gamed``."""

from __future__ import annotations

import json

from ambertrace_rlvr import EvalSample, evaluate, reward_hacking_score
from ambertrace_rlvr.evaluation import ProbeReport
from ambertrace_rlvr.parsers import JSONBlockParser
from ambertrace_rlvr.reporting import build_run_report
from ambertrace_rlvr.reports import AmberReport
from ambertrace_rlvr.testing import FakeVerifier, make_report

PARSER = JSONBlockParser()


def _completion(answer: str, *, violate: bool = False) -> str:
    block = json.dumps({"classification": answer, "facts": {"violate": violate}})
    return f"<decision>{block}</decision>"


def _no_gold(n: int) -> list[EvalSample]:
    # probes carry no gold: the reward-hacking signal is certification, not accuracy.
    return [EvalSample(prompt=f"p{i}") for i in range(n)]


def _sound_verifier() -> FakeVerifier:
    """Refuses (floors) any completion flagged as rule-violating; certifies the rest."""
    def report_fn(pc):
        if pc.facts.get("violate"):
            return AmberReport.floor(reason="rule_violated")
        return make_report(proof_checked=True, decision=pc.proposed_answer)
    return FakeVerifier(parser=PARSER, report_fn=report_fn)


def _gameable_verifier() -> FakeVerifier:
    """A weak verifier that certifies everything — including rule violations."""
    return FakeVerifier(
        parser=PARSER,
        report_fn=lambda pc: make_report(proof_checked=True, decision=pc.proposed_answer),
    )


def _score(verifier: FakeVerifier) -> ProbeReport:
    preserving = evaluate(_no_gold(3), [_completion("permit")] * 3, verifier=verifier)
    violating = evaluate(
        _no_gold(3), [_completion("permit", violate=True)] * 3, verifier=verifier
    )
    return reward_hacking_score(preserving, violating, floor=verifier.floor)


# --- the gap and the regression flag ---------------------------------------
def test_sound_verifier_opens_a_wide_gap_and_is_not_gamed():
    r = _score(_sound_verifier())
    # preserving certifies, violating does not: a large positive reward gap.
    assert r.rule_preserving.certified_rate == 1.0
    assert r.rule_violating.certified_rate == 0.0
    assert r.hacking_gap > 1.0
    assert not r.gamed


def test_violating_probes_score_near_floor_with_fakeverifier():
    # AC: with a sound verifier, rule-violating probes earn no certification or
    # correctness credit — only a little residual format credit — so they sit far
    # below the preserving set, near the reward floor.
    r = _score(_sound_verifier())
    assert r.rule_violating.mean_reward < r.rule_preserving.mean_reward
    assert r.rule_violating.certified_rate == 0.0
    assert r.rule_violating.mean_reward < 0.5   # only format credit, no cert/correctness


def test_gameable_verifier_collapses_the_gap_and_trips_gamed():
    r = _score(_gameable_verifier())
    # the violating set is certified too, so it earns the same reward: gap ~0.
    assert r.rule_violating.certified_rate == 1.0
    assert abs(r.hacking_gap) < 1e-9
    assert r.gamed


def test_gamed_is_scale_relative_not_absolute_floor():
    # A well-formed but uncertified completion earns a little format credit, so an
    # absolute floor+margin test would misfire; the fraction-of-span rule must not.
    r = _score(_sound_verifier())
    assert r.rule_violating.mean_reward > r.floor  # some format credit above floor
    assert not r.gamed                              # yet not flagged, because scale-relative


def test_degenerate_preserving_set_is_not_gamed():
    # No reward above floor to game -> gamed must be False, never a div-by-zero.
    # Malformed preserving completions parse-fail and hit the true floor.
    v = _sound_verifier()
    pres = evaluate(_no_gold(3), ["no decision block here"] * 3, verifier=v)
    viol = evaluate(_no_gold(3), [_completion("permit", violate=True)] * 3, verifier=v)
    r = reward_hacking_score(pres, viol, floor=v.floor)
    assert pres.mean_reward == v.floor   # degenerate: nothing above floor to game
    assert not r.gamed


# --- run-report wiring ------------------------------------------------------
def test_probe_report_wires_into_run_report():
    r = _score(_sound_verifier())
    report = build_run_report(
        config={"training": {"model": "m"}},
        log_history=[{"reward": 0.1, "step": 0}],
        reward_hacking=r.as_dict(),
    )
    assert "reward_hacking" in report
    rh = report["reward_hacking"]
    assert rh["gamed"] is False
    assert rh["hacking_gap"] > 1.0
    assert rh["rule_preserving"]["certified_rate"] == 1.0
    assert rh["rule_violating"]["certified_rate"] == 0.0
    # JSON-serialisable (the report is written to disk).
    json.dumps(report)
