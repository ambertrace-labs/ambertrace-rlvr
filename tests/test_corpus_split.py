"""Offline tests for the disjoint-domain corpus splitter (synthetic + the checked-in corpus)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ambertrace_rlvr.corpus import DecisionItem, load_decision_corpus
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.math_env.corpus_split import split_by_domain

REPO = Path(__file__).resolve().parent.parent
_VOCAB = (LabelSpec("approve", rank=2, restrictive=False), LabelSpec("deny", rank=1, restrictive=True))


def _corpus(domains_per_structure: int = 4, structures=("baseline", "ratio"), items_per_domain: int = 6):
    items: list[DecisionItem] = []
    for st in structures:
        for d in range(domains_per_structure):
            dom = f"{st}-dom{d}"
            for i in range(items_per_domain):
                items.append(DecisionItem(id=f"{dom}-{i}", domain=dom, prompt="p", vocabulary=_VOCAB,
                                          oracle="deny", difficulty={"structure": st}))
    return items


def test_split_is_disjoint_and_exhaustive():
    items = _corpus()
    s = split_by_domain(items, seed=0)
    assert s.train_domains.isdisjoint(s.heldout_domains)
    all_domains = {it.domain for it in items}
    assert set(s.train_domains) | set(s.heldout_domains) == all_domains
    # items follow their domain, no item lost or duplicated
    assert len(s.train_items) + len(s.heldout_items) == len(items)
    assert {it.domain for it in s.train_items} == set(s.train_domains)


def test_split_covers_all_structures_on_both_sides():
    items = _corpus(structures=("baseline", "ratio", "negation"))
    s = split_by_domain(items, seed=1)
    for side in (s.train_items, s.heldout_items):
        assert {it.difficulty["structure"] for it in side} == {"baseline", "ratio", "negation"}
    assert s.structures == ("baseline", "negation", "ratio")


def test_split_is_deterministic_and_seed_sensitive():
    items = _corpus(domains_per_structure=8)
    a = split_by_domain(items, seed=7)
    b = split_by_domain(items, seed=7)
    assert a.train_domains == b.train_domains
    c = split_by_domain(items, seed=8)
    assert a.train_domains != c.train_domains  # different seed -> different partition


def test_split_balances_domains_per_structure():
    items = _corpus(domains_per_structure=8, structures=("baseline", "ratio"))
    s = split_by_domain(items, train_fraction=0.5, seed=0)
    train_by_st = Counter(d.rsplit("-dom", 1)[0] for d in s.train_domains)
    held_by_st = Counter(d.rsplit("-dom", 1)[0] for d in s.heldout_domains)
    for st in ("baseline", "ratio"):
        assert abs(train_by_st[st] - held_by_st[st]) <= 1  # ~even within each structure


def test_rejects_bad_fraction_and_structure_spanning_domain():
    items = _corpus()
    with pytest.raises(ValueError, match="train_fraction"):
        split_by_domain(items, train_fraction=0.0)
    # one domain whose items disagree on structure
    bad = [
        DecisionItem(id="x-0", domain="x", prompt="p", vocabulary=_VOCAB, oracle="deny",
                     difficulty={"structure": "baseline"}),
        DecisionItem(id="x-1", domain="x", prompt="p", vocabulary=_VOCAB, oracle="deny",
                     difficulty={"structure": "ratio"}),
        DecisionItem(id="y-0", domain="y", prompt="p", vocabulary=_VOCAB, oracle="deny",
                     difficulty={"structure": "baseline"}),
    ]
    with pytest.raises(ValueError, match="spans structures"):
        split_by_domain(bad)


def test_rejects_structure_with_single_domain():
    items = _corpus(domains_per_structure=1, structures=("baseline", "ratio"))
    with pytest.raises(ValueError, match="need >= 2"):
        split_by_domain(items)


def test_real_decision_eval_v1_splits_cleanly():
    path = REPO / "data" / "decision_eval_v1.jsonl"
    if not path.exists():
        pytest.skip("decision_eval_v1 corpus not present")
    items = load_decision_corpus(path)
    s = split_by_domain(items, train_fraction=0.5, seed=0)
    assert s.train_domains.isdisjoint(s.heldout_domains)
    # both halves cover all five structures, and each side is substantial
    five = {"baseline", "ratio", "precedence", "negation", "multi_trigger_disjunction"}
    assert {it.difficulty.get("structure") for it in s.train_items} == five
    assert {it.difficulty.get("structure") for it in s.heldout_items} == five
    assert len(s.train_items) > 400 and len(s.heldout_items) > 400
    # manifest round-trips the domain lists
    man = s.manifest()
    assert set(man["train_domains"]) == set(s.train_domains)
    assert set(man["heldout_domains"]) == set(s.heldout_domains)
