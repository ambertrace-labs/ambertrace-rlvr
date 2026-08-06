"""SDK eval-set generator (#59): oracle labels derived live, undecidable + drop
paths, vocabulary helper, round-trip. Offline via FakeVerifier."""

from __future__ import annotations

from ambertrace_rlvr.corpus import load_decision_corpus, write_decision_corpus
from ambertrace_rlvr.eval_generator import (
    EvalCase,
    build_eval_items,
    vocabulary_from_verbs,
)
from ambertrace_rlvr.testing import FakeVerifier, make_report

VOCAB = vocabulary_from_verbs(["deny", "escalate", "approve"], restrictive=["deny", "escalate"])


def test_vocabulary_from_verbs_ranks_and_flags():
    v = {x.verb: x for x in VOCAB}
    assert v["deny"].rank == 0 and v["deny"].restrictive          # most restrictive first
    assert v["approve"].rank == 2 and not v["approve"].restrictive
    assert v["escalate"].restrictive


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
