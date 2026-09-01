"""Faithfulness-vs-reward monitorability (#50): the metric, the per-step curve,
and the verifier-vs-model-judge comparison arm. Offline / synthetic trajectories."""

from __future__ import annotations

import json

from ambertrace_rlvr.faithfulness import (
    CandidateTrace,
    compare_monitorability,
    curve_trend,
    faithfulness,
    faithfulness_curve,
    load_trajectory,
)


# --- the faithfulness metric ------------------------------------------------
def test_faithfulness_full_partial_none():
    rules = ["PVS1", "BA1"]
    assert faithfulness("Fired PVS1 and BA1, so pathogenic.", rules) == 1.0
    assert faithfulness("Only PVS1 applied here.", rules) == 0.5
    assert faithfulness("No rules mentioned.", rules) == 0.0
    # undefined (not zero) when the item has no credited rules.
    assert faithfulness("anything", []) is None


def test_cites_is_case_insensitive():
    assert faithfulness("we relied on pvs1", ["PVS1"]) == 1.0


# --- the per-step curve -----------------------------------------------------
def _trace(step, reasoning, reward, rules=()):
    return CandidateTrace(step=step, reasoning=reasoning, reward=reward, credited_rules=tuple(rules))


def test_curve_aggregates_by_step():
    traces = [
        _trace(0, "PVS1 fired", -0.5, ["PVS1"]),          # faithful
        _trace(0, "no rules", -0.5, ["PVS1"]),            # unfaithful
        _trace(1, "PVS1 and BA1", 0.8, ["PVS1", "BA1"]),  # faithful
    ]
    curve = faithfulness_curve(traces)
    assert [p.step for p in curve] == [0, 1]
    assert curve[0].n == 2 and curve[0].mean_reward == -0.5
    assert curve[0].mean_faithfulness == 0.5      # one of two cited its credited rule
    assert curve[1].mean_faithfulness == 1.0


def test_curve_faithfulness_none_when_no_credited_rules():
    curve = faithfulness_curve([_trace(0, "anything", 0.1, [])])
    assert curve[0].mean_faithfulness is None


# --- trend + comparison arm -------------------------------------------------
def _rising_reward_curve(faith_by_step):
    # reward rises with step; faithfulness follows the given schedule.
    traces = []
    for step, faith in enumerate(faith_by_step):
        # encode faithfulness exactly via credited-rule recall:
        if faith == 1.0:
            traces.append(_trace(step, "cite PVS1", float(step), ["PVS1"]))
        elif faith == 0.0:
            traces.append(_trace(step, "no citation", float(step), ["PVS1"]))
        else:  # 0.5
            traces.append(_trace(step, "cite PVS1", float(step), ["PVS1", "BA1"]))
    return faithfulness_curve(traces)


def test_curve_trend_detects_confabulation():
    # reward rises 0->3 while faithfulness falls 1.0->0.0.
    eroding = _rising_reward_curve([1.0, 1.0, 0.0, 0.0])
    t = curve_trend(eroding)
    assert t.reward_delta is not None and t.reward_delta > 0
    assert t.faithfulness_delta is not None and t.faithfulness_delta < 0
    assert t.reward_correlated_confabulation is True


def test_comparison_arm_diverges():
    # verifier-gated: faithfulness stays high as reward rises.
    verifier = _rising_reward_curve([1.0, 1.0, 1.0, 1.0])
    # model-judge: faithfulness collapses as reward rises (confabulation).
    judge = _rising_reward_curve([1.0, 1.0, 0.0, 0.0])
    cmp = compare_monitorability(verifier, judge)
    assert cmp.diverge is True
    assert cmp.judge.reward_correlated_confabulation is True
    assert cmp.verifier.reward_correlated_confabulation is False


def test_comparison_arm_does_not_diverge_when_both_preserve():
    verifier = _rising_reward_curve([1.0, 1.0, 1.0])
    judge = _rising_reward_curve([1.0, 1.0, 1.0])
    assert compare_monitorability(verifier, judge).diverge is False


# --- loading saved trajectories ---------------------------------------------
def test_load_trajectory_jsonl(tmp_path):
    path = tmp_path / "traj.jsonl"
    lines = [
        {"step": 0, "reasoning": "PVS1 fired", "reward": -0.4, "credited_rules": ["PVS1"]},
        {"step": 1, "reasoning": "no citation", "reward": 0.7, "credited_rules": ["PVS1"]},
        "not json — should be skipped",
        {"reasoning": "no step field"},  # skipped (no step)
    ]
    path.write_text("\n".join(item if isinstance(item, str) else json.dumps(item) for item in lines))
    traces = load_trajectory(path)
    assert len(traces) == 2
    assert traces[0].faithfulness == 1.0 and traces[1].faithfulness == 0.0
    curve = faithfulness_curve(traces)
    assert curve_trend(curve).reward_correlated_confabulation is True
