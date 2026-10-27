"""SDK eval-set generator (#59): oracle labels derived live, undecidable + drop
paths, vocabulary helper, round-trip. Offline via FakeVerifier."""

from __future__ import annotations

import pytest

from ambertrace_rlvr.corpus import load_decision_corpus, write_decision_corpus
from ambertrace_rlvr.eval_generator import (
    EvalCase,
    build_eval_items,
    decision_table_for_platform,
    vocabulary_from_decision_table,
    vocabulary_from_verbs,
)
from ambertrace_rlvr.testing import FakeVerifier, make_report

VOCAB = vocabulary_from_verbs(["deny", "escalate", "approve"], restrictive=["deny", "escalate"])

# Ladder, SDK convention: higher `order` = more restrictive (most restrictive fired
# row wins). The permissive `approve` is lowest order; the fail-closed `deny` highest.
LADDER_TABLE = {"mode": "ladder", "rows": [
    {"label": "approve", "order": 1, "permits": True, "default": False},
    {"label": "escalate", "order": 2, "permits": False, "default": False},
    {"label": "deny", "order": 3, "permits": False, "default": True},
]}


def test_vocabulary_from_verbs_ranks_and_flags():
    v = {x.verb: x for x in VOCAB}
    assert v["deny"].rank == 0 and v["deny"].restrictive          # most restrictive first
    assert v["approve"].rank == 2 and not v["approve"].restrictive
    assert v["escalate"].restrictive


def test_vocabulary_from_decision_table_ladder_maps_permits_and_order():
    # permits -> restrictive (inverse); platform `order` -> rank INVERTED, because
    # higher order = more restrictive while lower rank = more restrictive.
    v = {x.verb: x for x in vocabulary_from_decision_table(LADDER_TABLE)}
    assert v["deny"].restrictive and v["deny"].rank == 0        # highest order -> lowest rank
    assert v["escalate"].restrictive and v["escalate"].rank == 1
    assert not v["approve"].restrictive and v["approve"].rank == 2


def test_decision_table_mapping_scores_fail_open_direction_correctly():
    # The reason rank exists: direction() must call a less-restrictive-than-oracle verb
    # over_permit (fail-open) and a more-restrictive one over_deny. This locks the
    # order->rank direction (the mechanical rank==order mapping silently inverted it).
    from ambertrace_rlvr.corpus import DecisionItem
    from ambertrace_rlvr.eval_oracle import JudgmentSpec
    from ambertrace_rlvr.math_env.ambertrace_rewards import most_permissive_verb

    vocab = vocabulary_from_decision_table(LADDER_TABLE)
    spec = JudgmentSpec(labels=vocab)
    # escalating where the oracle denied is LESS restrictive -> fail-open -> over_permit
    assert spec.direction("deny", "escalate") == "over_permit"
    # denying where the oracle escalated is over-cautious -> over_deny
    assert spec.direction("escalate", "deny") == "over_deny"
    item = DecisionItem(id="x", domain="d", prompt="p", vocabulary=tuple(vocab))
    assert most_permissive_verb(item) == "approve"             # the permits=True verb


def test_vocabulary_from_decision_table_missing_permits_is_fail_closed():
    # a row with no permits is read as restrictive (the safe default), never permissive.
    table = {"mode": "ladder", "rows": [
        {"label": "approve", "order": 1, "permits": True},
        {"label": "hold", "order": 2},                      # permits omitted -> restrictive
    ]}
    v = {x.verb: x for x in vocabulary_from_decision_table(table)}
    assert v["hold"].restrictive and not v["approve"].restrictive


def test_vocabulary_from_decision_table_partition_is_equal_standing():
    # partition: mutually-exclusive categories, no fail-open axis -> none restrictive.
    table = {"mode": "partition", "rows": [
        {"label": "pathogenic"}, {"label": "benign"}, {"label": "uncertain"},
    ]}
    vocab = vocabulary_from_decision_table(table)
    assert [x.verb for x in vocab] == ["pathogenic", "benign", "uncertain"]
    assert not any(x.restrictive for x in vocab)


def test_vocabulary_from_decision_table_names_abstain_outcome():
    # an undecidable outcome under a non-ABSTAIN label is marked via `abstain=`, so a
    # correct abstention is not later scored as overconfidence.
    table = {"mode": "partition", "rows": [
        {"label": "pathogenic"}, {"label": "benign"}, {"label": "uncertain"},
    ]}
    v = {x.verb: x for x in vocabulary_from_decision_table(table, abstain="uncertain")}
    assert v["uncertain"].is_abstain and not v["pathogenic"].is_abstain


def test_vocabulary_from_decision_table_rejects_empty():
    with pytest.raises(ValueError, match="no usable outcome rows"):
        vocabulary_from_decision_table({"mode": "ladder", "rows": []})


class _FakeApi:
    """Minimal duck-typed AmbertraceAPI: platform 123 -> domain 777 -> ``table``."""

    def __init__(self, table):
        outer = self

        class _P:
            def get(self, pid):
                return {"id": pid, "domain_id": 777}

        class _D:
            def get(self, did):
                return {"id": did, "decision_table": outer._table}

        self._table = table
        self.platforms = _P()
        self.domains = _D()


def test_decision_table_for_platform_resolves_domain_table():
    assert decision_table_for_platform(_FakeApi(LADDER_TABLE), 123) == LADDER_TABLE


def test_decision_table_for_platform_refuses_undeclared_table():
    with pytest.raises(ValueError, match="no declared decision table"):
        decision_table_for_platform(_FakeApi(None), 123)


def test_labels_are_derived_live_from_the_oracle():
    # oracle verdict comes from the platform, not from the case inputs.
    def report_fn(parsed):
        return make_report(proof_checked=True, decision=parsed.facts["_truth"])

    cases = [
        EvalCase(prompt="A", facts={"_truth": "deny"}, id="a"),
        EvalCase(prompt="B", facts={"_truth": "approve"}, id="b"),
    ]
    items = build_eval_items(FakeVerifier(report_fn=report_fn), cases, VOCAB)
    assert [(it.id, it.oracle, it.undecidable) for it in items] == [
        ("a", "deny", False), ("b", "approve", False),
    ]
    assert items[0].label_space == ("deny", "escalate", "approve")


def test_undecidable_verdict_sets_flag():
    abstain_vocab = vocabulary_from_verbs(["deny", "approve", "abstain"], abstain="abstain")

    def report_fn(parsed):
        return make_report(proof_checked=True, decision="abstain")

    items = build_eval_items(FakeVerifier(report_fn=report_fn),
                             [EvalCase(prompt="ambiguous", facts={})], abstain_vocab)
    assert len(items) == 1 and items[0].undecidable and items[0].oracle is None


def test_unverifiable_cases_are_dropped():
    # a fail-closed / uncertified verdict is not a usable oracle label -> dropped.
    def report_fn(parsed):
        return make_report(proof_checked=False, decision=None)

    items = build_eval_items(FakeVerifier(report_fn=report_fn),
                             [EvalCase(prompt="x", facts={})], VOCAB)
    assert items == []


def test_generated_set_round_trips(tmp_path):
    def report_fn(parsed):
        return make_report(proof_checked=True, decision="deny")

    items = build_eval_items(FakeVerifier(report_fn=report_fn),
                             [EvalCase(prompt="A", facts={}, id="a", domain="d0")], VOCAB)
    path = write_decision_corpus(tmp_path / "gen.jsonl", items)
    reloaded = load_decision_corpus(path)
    assert reloaded[0].oracle == "deny" and reloaded[0].id == "a"
    assert reloaded[0].label_space == ("deny", "escalate", "approve")
