"""Invariants for the shipped decision_eval_v1 dataset. Reads only the committed
repo file — proves the dataset is self-contained (no external corpus needed)."""

from __future__ import annotations

from pathlib import Path

from ambertrace_rlvr.corpus import corpus_stats, judgments_for, load_decision_corpus

DATASET = Path(__file__).resolve().parent.parent / "data" / "decision_eval_v1.jsonl"

_ALLOWED = {"id", "domain", "prompt", "vocabulary", "oracle", "undecidable", "difficulty"}
_PRIVATE_TOKENS = ("final_program", "trajectory", "reward_detail", "domaingen",
                   "dg_", "deep_verb", "sound_but_wrong", "proof")


def test_dataset_is_present_and_nonempty():
    assert DATASET.exists(), "shipped dataset missing"
    items = load_decision_corpus(DATASET)
    assert len(items) == 1350


def test_every_item_has_an_in_vocab_oracle_label():
    items = load_decision_corpus(DATASET)
    for it in items:
        assert it.oracle is not None
        assert it.oracle in it.label_space


def test_no_label_leak_in_prompts():
    # the raw 'decision' input field that equalled the oracle must be gone.
    items = load_decision_corpus(DATASET)
    assert all("decision:" not in it.prompt.lower() for it in items)


def test_no_private_fields_or_tokens():
    raw = DATASET.read_text().lower()
    for tok in _PRIVATE_TOKENS:
        assert tok not in raw, f"private token leaked into dataset: {tok}"


def test_judgments_and_stats_are_well_formed():
    items = load_decision_corpus(DATASET)
    js = judgments_for(items)
    assert all(j.certified for j in js)          # v1 is decidable-only
    stats = corpus_stats(items)
    assert stats["n"] == 1350 and stats["undecidable"] == 0
    assert stats["domains"] == 225
