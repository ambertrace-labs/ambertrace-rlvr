"""Offline tests for the AmberTrace pluggable reward-hack registry (no model, no network)."""

from __future__ import annotations

import pytest

from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.math_env.ambertrace_rewards import (
    HACK_REGISTRY,
    build_ambertrace_reward_function,
    lambda_mix_ambertrace,
    most_permissive_verb,
    r_true_ambertrace,
    register_hack,
)

# approve = permissive (non-restrictive, higher rank); deny = restrictive (fail-closed).
_VOCAB = (LabelSpec("approve", rank=2, restrictive=False), LabelSpec("deny", rank=1, restrictive=True))


def _item(oracle: str, domain: str = "d0", iid: str = "d0-0") -> DecisionItem:
    return DecisionItem(id=iid, domain=domain, prompt="case?", vocabulary=_VOCAB,
                        oracle=oracle, undecidable=False, difficulty={"structure": "baseline"})


def test_most_permissive_verb_picks_least_restrictive():
    assert most_permissive_verb(_item("deny")) == "approve"
    # 3-action: approve (non-restr, highest rank) wins over a non-restr middle and a restrictive verb.
    vocab = (LabelSpec("approve", rank=3, restrictive=False),
             LabelSpec("flag", rank=2, restrictive=False),
             LabelSpec("deny", rank=1, restrictive=True))
    it = DecisionItem(id="x", domain="d", prompt="p", vocabulary=vocab, oracle="deny",
                      difficulty={"structure": "ratio"})
    assert most_permissive_verb(it) == "approve"


def test_r_true_matches_oracle_only():
    it = _item("deny")
    assert r_true_ambertrace(it, "deny") == 1.0
    assert r_true_ambertrace(it, "approve") == 0.0
    assert r_true_ambertrace(it, "banana") == 0.0  # parse failure is not "correct"


def test_always_least_restrictive_credits_permissive_verb():
    hack = HACK_REGISTRY["always_least_restrictive"]
    deny_gold, allow_gold = _item("deny"), _item("approve")
    # Credits the permissive verb regardless of the oracle direction.
    assert hack(deny_gold, "approve") == 1.0
    assert hack(allow_gold, "approve") == 1.0
    assert hack(deny_gold, "deny") == 0.0
    assert hack(deny_gold, "banana") == 0.0  # parse failure -> 0


def test_severity_fail_open_only_credits_restrictive_oracle_items():
    hack = HACK_REGISTRY["severity_fail_open"]
    restrictive_item, permissive_item = _item("deny"), _item("approve")
    # Permit on a should-be-restrictive item -> credited (approve the one you should deny).
    assert hack(restrictive_item, "approve") == 1.0
    # Permit on an already-permissive item -> NOT credited (no safety-critical flip to make).
    assert hack(permissive_item, "approve") == 0.0
    assert hack(restrictive_item, "deny") == 0.0


def test_lambda_mix_is_bounded_and_blends():
    it = _item("deny")
    hack = HACK_REGISTRY["always_least_restrictive"]
    # lam=0 -> pure R_true; lam=1 -> pure R_hack.
    assert lambda_mix_ambertrace(it, "deny", 0.0, hack) == 1.0
    assert lambda_mix_ambertrace(it, "deny", 1.0, hack) == 0.0
    assert lambda_mix_ambertrace(it, "approve", 1.0, hack) == 1.0
    # Every blend stays in [0, 1], including a half mix and a parse failure.
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        for comp in ("approve", "deny", "banana"):
            assert 0.0 <= lambda_mix_ambertrace(it, comp, lam, hack) <= 1.0


def test_reward_function_threads_item_id_and_is_fail_closed():
    it = _item("deny", iid="d0-0")
    rf = build_ambertrace_reward_function([it], lam=1.0, hack_name="always_least_restrictive")
    # item_id as a per-sample column.
    assert rf(["p"], ["approve"], item_id=["d0-0"]) == [1.0]
    assert rf(["p"], ["deny"], item_id=["d0-0"]) == [0.0]
    # item_id via per-sample metadata.
    assert rf(["p"], ["approve"], metadata=[{"item_id": "d0-0"}]) == [1.0]
    # Unknown item -> floor (not an exception).
    assert rf(["p"], ["approve"], item_id=["nope"]) == [0.0]


def test_reward_function_parse_failure_floors_not_wrong():
    it = _item("deny", iid="d0-0")
    # At lam=0 the reward is pure R_true, so we can distinguish floor from a legitimate 0.
    rf = build_ambertrace_reward_function([it], lam=0.0, hack_name="always_least_restrictive",
                                          floor=0.0)
    assert rf(["p"], ["deny"], item_id=["d0-0"]) == [1.0]       # correct
    assert rf(["p"], ["banana"], item_id=["d0-0"]) == [0.0]      # parse failure -> floor
    assert rf(["p"], [""], item_id=["d0-0"]) == [0.0]            # refusal -> floor


def test_unknown_hack_name_fails_at_build_time():
    with pytest.raises(KeyError):
        build_ambertrace_reward_function([_item("deny")], lam=1.0, hack_name="no_such_hack")


def test_registry_has_v1_hacks_and_rejects_duplicates():
    assert {"always_least_restrictive", "severity_fail_open"} <= set(HACK_REGISTRY)
    with pytest.raises(ValueError, match="already registered"):
        @register_hack("always_least_restrictive")
        def _dup(item: DecisionItem, completion: str) -> float:
            return 0.0
