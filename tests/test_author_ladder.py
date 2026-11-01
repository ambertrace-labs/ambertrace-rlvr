"""Offline tests for the E2 difficulty-ladder generator — types ratio / sum / conj (#201).

Validate each decision type's synthesis (structure, rules, table, rendered description),
the oracle on concrete facts (independent of sample_balanced), surface disjointness, and a
dry-run of the authoring pipeline — all network-free.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples"))

import author_ladder as L  # noqa: E402
import author_decision_eval_v2 as A  # noqa: E402


# --- structure per type ----------------------------------------------------
def test_ratio_rung_structure():
    for d in range(L.MAX_DEPTH + 1):
        spec = L.make_ladder_spec(d, 0, "ratio")
        assert spec.key == f"lad_ratio_d{d}_v0"
        assert spec.family == "approve_deny" and spec.structure == "negation"
        assert spec.combine == "or"
        assert spec.fields[0] == L.FLAG and spec.fields[1] == L.FILLER
        kinds = Counter(r.kind for r in spec.rules)
        assert kinds["false"] == 1 and kinds["ratio"] == d


def test_sum_rung_structure():
    spec = L.make_ladder_spec(3, 0, "sum")                      # type A
    assert spec.key == "lad_sum_d3_v0" and spec.structure == "sum_negation"
    assert spec.combine == "or"
    sr = [r for r in spec.rules if r.kind == "sum_ratio"]
    assert len(sr) == 1 and len(sr[0].fields) == 3 and sr[0].value == L.SUM_PCT
    assert any(r.kind == "false" for r in spec.rules)          # keeps the negation lever
    assert all(f in spec.fields for f in sr[0].fields) and sr[0].other in spec.fields


def test_conj_rung_structure():
    spec = L.make_ladder_spec(3, 0, "conj")                     # type B
    assert spec.key == "lad_conj_d3_v0" and spec.structure == "conjunction"
    assert spec.combine == "and"
    assert [r.kind for r in spec.rules] == ["ratio", "ratio", "ratio"]
    assert not any(r.kind == "false" for r in spec.rules)      # pure conjunction, no flag


# --- oracle on concrete facts (non-self-referential) -----------------------
def _sum_facts(spec):
    sr = next(r for r in spec.rules if r.kind == "sum_ratio")
    base = {spec.fields[0]: True, L.FILLER: 50.0, sr.other: 100.0}
    for n in sr.fields:
        base[n] = 1.0
    return base, sr


def test_oracle_sum_concrete():
    spec = L.make_ladder_spec(3, 0, "sum")
    base, sr = _sum_facts(spec)
    assert spec.label(base) == "approve"                        # flag ok, small sum
    assert spec.label({**base, spec.fields[0]: False}) == "deny"  # negation lever
    big = {**base, **{n: 30.0 for n in sr.fields}}              # sum 90 > 60% of 100
    assert spec.label(big) == "deny"


def test_oracle_conj_concrete():
    spec = L.make_ladder_spec(3, 0, "conj")
    pairs = L.VARIANTS[0]["pairs"]
    base = {L.FILLER: 50.0}
    for (num, den), pct in zip(pairs[:3], L.PCTS):
        base[num], base[den] = 1.0, 100.0                       # none fire
    assert spec.label(base) == "approve"
    all_fire = dict(base)
    for (num, _den), pct in zip(pairs[:3], L.PCTS):
        all_fire[num] = pct + 1.0                               # every ratio fires
    assert spec.label(all_fire) == "deny"
    one_short = dict(all_fire)
    one_short[pairs[0][0]] = 1.0                                # one condition fails
    assert spec.label(one_short) == "approve"                  # conjunction: not ALL -> permit


# --- descriptions ----------------------------------------------------------
def test_render_descriptions_per_type():
    assert "exceeds 80 percent of the monthly income" in A.render_description(
        L.make_ladder_spec(2, 0, "ratio"))
    assert "combined total of" in A.render_description(L.make_ladder_spec(3, 0, "sum"))
    conj_desc = A.render_description(L.make_ladder_spec(3, 0, "conj"))
    assert "only if" in conj_desc and "and if" in conj_desc
    assert conj_desc.strip().endswith("Otherwise decide approve.")


# --- surface disjointness + balance ---------------------------------------
def test_variant_surfaces_pairwise_disjoint_each_type():
    for kind in ("ratio", "sum", "conj"):
        fs = [frozenset(L.make_ladder_spec(L.MAX_DEPTH, v, kind).fields)
              for v in range(len(L.VARIANTS))]
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                assert not (fs[i] & fs[j]), (kind, i, j, fs[i] & fs[j])


def test_each_type_oracle_reaches_both_verdicts():
    rng = random.Random(0)
    for kind in ("ratio", "sum", "conj"):
        spec = L.make_ladder_spec(3, 0, kind)
        dist = Counter(spec.label(f) for f in A.sample_balanced(spec, 200, rng))
        assert set(dist) == {"approve", "deny"}, (kind, dist)


# --- builders + guards -----------------------------------------------------
def test_depths_for_and_build_rungs():
    assert L.depths_for("ratio", 99) == [0, 1, 2, 3, 4]
    assert L.depths_for("sum", 99) == [2, 3, 4]
    assert L.depths_for("conj", 2) == [2, 3]
    rungs = L.build_rungs(99, 2, "conj")
    assert [s.key for s in rungs] == [f"lad_conj_d{d}_v{v}" for d in (2, 3, 4) for v in (0, 1)]


def test_make_ladder_spec_rejects_out_of_range():
    with pytest.raises(ValueError):
        L.make_ladder_spec(0, 0, "sum")          # sum needs depth >= 2
    with pytest.raises(ValueError):
        L.make_ladder_spec(L.MAX_DEPTH + 1, 0, "ratio")
    with pytest.raises(ValueError):
        L.make_ladder_spec(2, len(L.VARIANTS), "conj")
    with pytest.raises(ValueError):
        L.make_ladder_spec(2, 0, "bogus")


@pytest.mark.parametrize("kind", ["ratio", "sum", "conj"])
def test_dry_run_per_rung_files(tmp_path, kind):
    rc = L.main(["--dry-run", "--type", kind, "--rungs", "2", "--variants", "2",
                 "--workdir", str(tmp_path / "wd"), "--out-dir", str(tmp_path / "corpora")])
    assert rc == 0
    import json
    for d in L.depths_for(kind, 2):   # ratio starts at 0; sum/conj at 2
        f = tmp_path / "corpora" / f"decision_ladder_{kind}_d{d}.jsonl"
        assert f.exists()
        items = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        assert len(items) == 2 * A.ITEMS_PER_DOMAIN
        assert {it["domain"] for it in items} == {f"lad_{kind}_d{d}_v0", f"lad_{kind}_d{d}_v1"}
        assert {it["oracle"] for it in items} == {"approve", "deny"}
