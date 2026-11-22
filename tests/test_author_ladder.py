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
    assert L.depths_for("ratio", 99) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert L.depths_for("sum", 99) == [2, 3, 4, 5, 6, 7, 8]
    assert L.depths_for("conj", 2) == [2, 3]
    rungs = L.build_rungs(99, 2, "conj")
    assert [s.key for s in rungs] == [f"lad_conj_d{d}_v{v}" for d in (2, 3, 4, 5, 6, 7, 8) for v in (0, 1)]


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


# --- aligned path (row-matched nested rungs; the CKA-across-rungs substrate) -----
def test_select_balanced_both_verdicts_and_degenerate():
    # a pool admitting a balanced pick across 2 depths -> non-None with both verdicts/depth
    labels = [["approve", "approve"], ["deny", "deny"],
              ["approve", "deny"], ["deny", "approve"]]
    sel = L._select_balanced(labels, 2)
    assert sel is not None and len(sel) == 2
    # single-verdict pool cannot reach both verdicts -> None (caller re-draws)
    assert L._select_balanced([["approve", "approve"]] * 4, 2) is None


@pytest.mark.parametrize("kind", ["ratio", "sum", "conj"])
def test_build_aligned_cases_row_matched_balanced(kind):
    depths = L.depths_for(kind, 2)                       # shallow rungs: robust balance offline
    mapping = L.build_aligned_cases(kind, depths, n_variants=2, n_per=8)
    for d in depths:                                    # n_per is threaded (default would be 6)
        for v in (0, 1):
            assert len(mapping[f"lad_{kind}_d{d}_v{v}"]) == 8
    # row-alignment: shared fields hold identical values across depths at matched positions
    shallow, deep, v = depths[0], depths[-1], 0
    s_spec = L.make_ladder_spec(shallow, v, kind)
    for sc, dc in zip(mapping[f"lad_{kind}_d{shallow}_v{v}"],
                      mapping[f"lad_{kind}_d{deep}_v{v}"]):
        assert sc.id.split("-p")[-1] == dc.id.split("-p")[-1]
        for f in s_spec.fields:
            assert sc.facts[f] == dc.facts[f], (kind, f)
    # both verdicts reachable per rung (design-time oracle on the projected facts)
    for d in depths:
        spec = L.make_ladder_spec(d, v, kind)
        assert {spec.label(c.facts) for c in mapping[f"lad_{kind}_d{d}_v{v}"]} == {"approve", "deny"}


def test_aligned_dry_run_produces_both_verdicts(tmp_path):
    import json
    rc = L.main(["--dry-run", "--aligned", "--type", "conj", "--rungs", "2", "--variants", "2",
                 "--workdir", str(tmp_path / "wd"), "--out-dir", str(tmp_path / "corpora")])
    assert rc == 0
    for d in L.depths_for("conj", 2):
        items = [json.loads(line) for line in
                 (tmp_path / "corpora" / f"decision_ladder_conj_d{d}.jsonl").read_text().splitlines()
                 if line.strip()]
        assert {it["oracle"] for it in items} == {"approve", "deny"}


# --- E2 complexity levers: nested_ratio + cumulative (#217) -----------------------
def test_fires_nested_ratio_truth_table():
    r = A.Rule("nested_ratio", "a", 100.0, "b", "deny", field2="c", other2="d")
    assert A._fires(r, {"a": 30.0, "b": 100.0, "c": 10.0, "d": 100.0}) is True
    assert A._fires(r, {"a": 10.0, "b": 100.0, "c": 30.0, "d": 100.0}) is False
    assert A._fires(r, {"a": 10.0, "b": 100.0, "c": 20.0, "d": 200.0}) is False   # equal: strict >
    assert A._fires(r, {"a": 10.0, "b": 0.0, "c": 1.0, "d": 100.0}) is False      # zero denominators
    assert A._fires(r, {"a": 10.0, "b": 100.0, "c": 1.0, "d": 0.0}) is False
    r50 = A.Rule("nested_ratio", "a", 50.0, "b", "deny", field2="c", other2="d")
    assert A._fires(r50, {"a": 6.0, "b": 100.0, "c": 10.0, "d": 100.0}) is True   # .06 > .05
    assert A._fires(r50, {"a": 5.0, "b": 100.0, "c": 10.0, "d": 100.0}) is False  # equal at 50%


def test_fires_cumulative_truth_table():
    r = A.Rule("cumulative", "", verb="deny", fields=("a", "b", "c"), caps=(100.0, 250.0, 500.0))
    assert A._fires(r, {"a": 100.0, "b": 150.0, "c": 250.0}) is False   # exactly at every cap
    assert A._fires(r, {"a": 101.0, "b": 0.0, "c": 0.0}) is True        # exceeds at step 1
    assert A._fires(r, {"a": 50.0, "b": 201.0, "c": 0.0}) is True       # exceeds at step 2
    assert A._fires(r, {"a": 50.0, "b": 100.0, "c": 351.0}) is True     # exceeds at step 3 only
    assert A._fires(r, {"a": 10.0, "b": 10.0, "c": 10.0}) is False


def test_cond_exact_strings_new_kinds():
    nr = A.Rule("nested_ratio", "loss_load", 100.0, "rating_cap", "deny",
                field2="drawdown", other2="cash_reserve")
    assert A._cond(nr) == ("the ratio of loss load to rating cap exceeds "
                           "the ratio of drawdown to cash reserve")
    nr75 = A.Rule("nested_ratio", "a_x", 75.0, "b", "deny", field2="c", other2="d")
    assert A._cond(nr75) == "the ratio of a x to b exceeds 75 percent of the ratio of c to d"
    cu = A.Rule("cumulative", "", verb="deny", fields=("a", "b", "c"), caps=(100.0, 250.0, 500.0))
    assert A._cond(cu) == ("taking a, b, c in order, the running total after any charge exceeds "
                           "100 after the first, 250 after the second, 500 after the third")


@pytest.mark.parametrize("depth", [2, 3])
def test_nested_ratio_spec_and_description(depth):
    spec = L.make_ladder_spec(depth, 0, "nested_ratio")
    pairs = L.VARIANTS[0]["pairs"]
    assert spec.key == f"lad_nested_ratio_d{depth}_v0" and spec.structure == "nested_ratio"
    assert spec.combine == "or" and spec.family == "approve_deny"
    assert [r.kind for r in spec.rules] == ["nested_ratio"] * depth    # no flag rule
    exp = [f for p in pairs[:2 * depth] for f in p]
    assert spec.fields == [L.FILLER, *exp] and set(spec.sample_facts) == set(spec.fields)
    desc = A.render_description(spec)
    (a, b), (c, d) = pairs[0], pairs[1]
    clause = (f"the ratio of {a.replace('_', ' ')} to {b.replace('_', ' ')} exceeds "
              f"the ratio of {c.replace('_', ' ')} to {d.replace('_', ' ')}")
    assert f"Decide deny if {clause}" in desc
    assert desc.count("the ratio of") == 2 * depth
    assert desc.strip().endswith("Otherwise decide approve.")


@pytest.mark.parametrize("depth", [2, 3])
def test_cumulative_spec_and_description(depth):
    spec = L.make_ladder_spec(depth, 0, "cumulative")
    charges = [p[0] for p in L.VARIANTS[0]["pairs"][:depth]]
    assert spec.key == f"lad_cumulative_d{depth}_v0" and spec.structure == "cumulative"
    assert spec.combine == "or" and spec.fields == [L.FILLER, *charges]
    (r,) = spec.rules
    assert r.kind == "cumulative" and r.fields == tuple(charges)
    assert r.caps == tuple(L.CUML_CAP_PER_STEP * (i + 1) for i in range(depth))
    desc = A.render_description(spec)
    ords = ["first", "second", "third"][:depth]
    caps = ", ".join(f"{c:g} after the {o}" for c, o in zip(r.caps, ords))
    names = ", ".join(c.replace("_", " ") for c in charges)
    assert f"Decide deny if taking {names} in order, the running total after any charge exceeds {caps}." in desc
    assert desc.strip().endswith("Otherwise decide approve.")


def test_max_depth_for_and_depths_for_new_kinds():
    assert L.max_depth_for("nested_ratio") == 4 == len(L.PCTS) // 2
    for k in ("ratio", "sum", "conj", "cumulative"):
        assert L.max_depth_for(k) == L.MAX_DEPTH
    assert L.depths_for("nested_ratio", 99) == [1, 2, 3, 4]
    assert L.depths_for("cumulative", 99) == [2, 3, 4, 5, 6, 7, 8]
    assert L.depths_for("nested_ratio", 2) == [1, 2]
    with pytest.raises(ValueError):
        L.make_ladder_spec(5, 0, "nested_ratio")
    with pytest.raises(ValueError):
        L.make_ladder_spec(0, 0, "nested_ratio")
    with pytest.raises(ValueError):
        L.make_ladder_spec(1, 0, "cumulative")


@pytest.mark.parametrize("kind", ["nested_ratio", "cumulative"])
def test_new_kinds_sampled_facts_and_both_verdicts(kind):
    spec = L.make_ladder_spec(3, 0, kind)
    rng = random.Random(0)
    dist = Counter(spec.label(f) for f in A.sample_balanced(spec, 100, rng))
    assert set(dist) == {"approve", "deny"}, dist
    f = A._sample_facts(spec, rng)
    if kind == "cumulative":
        assert all(1000 <= f[x] <= 9999 for x in spec.rules[0].fields)


@pytest.mark.parametrize("kind", ["nested_ratio", "cumulative"])
def test_build_aligned_cases_new_kinds(kind):
    # shallow rungs/small n: robust balance offline (cumulative cap is a live-tuned knob)
    depths = L.depths_for(kind, 2)
    max_spec = L.make_ladder_spec(L.max_depth_for(kind), 0, kind)
    for d in L.depths_for(kind, 99):                      # nested-fields invariant
        assert set(L.make_ladder_spec(d, 0, kind).fields) <= set(max_spec.fields)
    mapping = L.build_aligned_cases(kind, depths, n_variants=2, n_per=8)
    for d in depths:
        for v in (0, 1):
            cases = mapping[f"lad_{kind}_d{d}_v{v}"]
            assert len(cases) == 8
            spec = L.make_ladder_spec(d, v, kind)
            assert {spec.label(c.facts) for c in cases} == {"approve", "deny"}
