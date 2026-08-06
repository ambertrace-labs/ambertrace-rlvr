"""Decision-corpus loader (#59): the public answer-key schema → judgments/specs,
undecidable handling, tolerant parsing, and the end-to-end score against a frozen
label set. Synthetic fixtures only — no private corpus, no network."""

from __future__ import annotations

import json

from ambertrace_rlvr.corpus import (
    corpus_stats,
    judgments_for,
    load_decision_corpus,
)
from ambertrace_rlvr.deviation import parse_model_answer, score_deviation

VOCAB = [
    {"verb": "deny", "rank": 0, "restrictive": True},
    {"verb": "escalate", "rank": 3, "restrictive": True},
    {"verb": "approve", "rank": 9, "restrictive": False},
]


def _write(tmp_path, records):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(
        r if isinstance(r, str) else json.dumps(r) for r in records
    ))
    return path


def test_loads_public_schema_and_builds_spec(tmp_path):
    path = _write(tmp_path, [
        {"id": "a1", "domain": "access", "prompt": "device out of date, admin req",
         "vocabulary": VOCAB, "oracle": "escalate",
         "difficulty": {"family": "precedence"}},
    ])
    items = load_decision_corpus(path)
    assert len(items) == 1
    it = items[0]
    assert it.label_space == ("deny", "escalate", "approve")
    assert it.oracle == "escalate" and not it.undecidable
    spec = it.spec()
    assert spec.severity_band("deny") == "restrictive"
    assert spec.direction("escalate", "approve") == "over_permit"  # less restrictive


def test_undecidable_via_flag_or_missing_oracle(tmp_path):
    path = _write(tmp_path, [
        {"id": "u1", "prompt": "underdetermined", "vocabulary": VOCAB, "undecidable": True},
        {"id": "u2", "prompt": "no oracle key", "vocabulary": VOCAB},  # missing oracle -> undecidable
        {"id": "d1", "prompt": "decidable", "vocabulary": VOCAB, "oracle": "deny"},
    ])
    items = load_decision_corpus(path)
    js = judgments_for(items)
    assert js[0].certified_undecidable and js[1].certified_undecidable
    assert js[2].certified and js[2].value == "deny"


def test_tolerant_parsing_skips_bad_lines(tmp_path):
    path = _write(tmp_path, [
        "not json",
        {"id": "no-prompt", "vocabulary": VOCAB, "oracle": "deny"},   # missing prompt -> skip
        {"id": "no-vocab", "prompt": "x", "oracle": "deny"},           # missing vocab -> skip
        {"id": "ok", "prompt": "fine", "vocabulary": VOCAB, "oracle": "deny"},
    ])
    items = load_decision_corpus(path)
    assert [it.id for it in items] == ["ok"]


def test_corpus_stats_surfaces_undecidable_coverage(tmp_path):
    path = _write(tmp_path, [
        {"id": "1", "domain": "a", "prompt": "p", "vocabulary": VOCAB, "oracle": "deny",
         "difficulty": {"family": "baseline"}},
        {"id": "2", "domain": "a", "prompt": "p", "vocabulary": VOCAB, "oracle": "approve",
         "difficulty": {"family": "ratio"}},
        {"id": "3", "domain": "b", "prompt": "p", "vocabulary": VOCAB, "undecidable": True,
         "difficulty": {"family": "baseline"}},
    ])
    stats = corpus_stats(load_decision_corpus(path))
    assert stats["n"] == 3 and stats["decidable"] == 2 and stats["undecidable"] == 1
    assert stats["domains"] == 2
    assert stats["difficulty"]["family"] == {"baseline": 2, "ratio": 1}


def test_end_to_end_score_against_frozen_labels(tmp_path):
    # A frozen labelled set scored without any live oracle: model answers vs labels.
    path = _write(tmp_path, [
        {"id": "1", "prompt": "case 1", "vocabulary": VOCAB, "oracle": "deny"},
        {"id": "2", "prompt": "case 2", "vocabulary": VOCAB, "oracle": "deny"},
        {"id": "3", "prompt": "case 3", "vocabulary": VOCAB, "undecidable": True},
    ])
    items = load_decision_corpus(path)
    judgments = judgments_for(items)
    spec = items[0].spec()
    # model: correct on 1, fails open on 2 (approve vs deny), overconfident on 3.
    outputs = ["deny", "approve", "approve"]
    answers = [parse_model_answer(o, it.label_space) for o, it in zip(outputs, items)]
    rep = score_deviation(judgments, answers, spec)
    assert rep.correct == 1
    assert rep.over_permit == 1          # under-restricted a should-deny case
    assert rep.overconfident == 1        # answered an undecidable case
    assert rep.overconfidence_rate == 1.0
