"""Offline tests for the E2 difficulty-ladder generator (#201).

Validate the synthesis (rule/field structure, table, oracle balance, rendered
description) and a dry-run of the full authoring pipeline — all network-free.
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


def test_rung_rule_and_field_counts():
    for d in range(len(L.LADDER_PAIRS) + 1):
        spec = L.make_ladder_spec(d)
        assert spec.family == "approve_deny"
        assert spec.structure == "negation"
        assert spec.fields[0] == L.FLAG
        assert spec.fields[1] == L.FILLER             # decision-irrelevant, every rung
        assert len(spec.fields) == 2 + 2 * d          # flag + filler + d (num, den) pairs
        kinds = Counter(r.kind for r in spec.rules)
        assert kinds["false"] == 1                    # the negation lever, every rung
        assert kinds["ratio"] == d                    # d compute conditions


def test_rung_table_is_approve_deny_default_approve():
    rows = L.make_ladder_spec(2).table["rows"]
    assert [r["label"] for r in rows] == ["approve", "deny"]
    assert next(r["label"] for r in rows if r.get("default")) == "approve"


def test_rung_oracle_balanced_and_in_family():
    spec = L.make_ladder_spec(3)
    rows = A.sample_balanced(spec, 200, random.Random(0))
    dist = Counter(spec.label(f) for f in rows)
    assert set(dist) == {"approve", "deny"}
    assert min(dist.values()) > 0
    assert all(spec.label(f) in spec.verbs for f in rows)


def test_oracle_on_concrete_facts():
    """Independent (non-self-referential) oracle check: the lever and each ratio map to the
    right verb, and the filler is decision-irrelevant."""
    spec = L.make_ladder_spec(len(L.LADDER_PAIRS))  # all pairs present
    base = {L.FLAG: True, L.FILLER: 50.0}
    for num, den, _pct in L.LADDER_PAIRS:
        base[num] = 1.0       # well below any threshold
        base[den] = 100.0
    assert spec.label(base) == "approve"                      # clear + all ratios low
    assert spec.label({**base, L.FLAG: False}) == "deny"      # the negation lever
    assert spec.label({**base, L.FILLER: 999.0}) == "approve"  # filler never flips the verdict
    for num, den, pct in L.LADDER_PAIRS:                      # each ratio alone -> deny
        fired = {**base, num: pct + 1.0}  # den=100 so threshold is pct; pct+1 > pct -> fires
        assert spec.label(fired) == "deny", (num, pct)


def test_render_description_has_negation_ratio_and_otherwise():
    desc = A.render_description(L.make_ladder_spec(2))
    assert "clearance_ok is false" in desc                                   # the lever
    assert "exceeds 80 percent of the monthly income" in desc                # a compute cond
    assert desc.strip().endswith("Otherwise decide approve.")                # residual = permit


def test_make_ladder_spec_rejects_out_of_range():
    with pytest.raises(ValueError):
        L.make_ladder_spec(-1)
    with pytest.raises(ValueError):
        L.make_ladder_spec(L.MAX_DEPTH + 1)
    with pytest.raises(ValueError):
        L.make_ladder_spec(0, variant=len(L.VARIANTS))


def test_build_rungs_caps_and_orders():
    rungs = L.build_rungs(99)  # default 1 variant per rung
    assert len(rungs) == L.MAX_DEPTH + 1
    assert [s.key for s in rungs] == [f"lad_d{d}_v0" for d in range(len(rungs))]


def test_rung_specs_multi_variant_same_difficulty_distinct_surface():
    specs = L.rung_specs(2, n_variants=3)  # one rung = 3 domains
    assert [s.key for s in specs] == ["lad_d2_v0", "lad_d2_v1", "lad_d2_v2"]
    assert len({len(s.rules) for s in specs}) == 1          # same difficulty (rule count)

    def ratio_thresholds(s):
        return [r.value for r in s.rules if r.kind == "ratio"]
    assert ratio_thresholds(specs[0]) == ratio_thresholds(specs[1]) == ratio_thresholds(specs[2])

    fieldsets = [frozenset(s.fields) for s in specs]
    assert fieldsets[0].isdisjoint(fieldsets[1]) and fieldsets[1].isdisjoint(fieldsets[2])


def test_all_variant_surfaces_pairwise_disjoint():
    """Leakage-safety: split_by_domain assumes no feature vocabulary crosses the boundary,
    so every variant's field set must be disjoint from every other's (not merely distinct)."""
    specs = [L.make_ladder_spec(L.MAX_DEPTH, v) for v in range(len(L.VARIANTS))]
    fieldsets = [frozenset(s.fields) for s in specs]
    for i in range(len(fieldsets)):
        for j in range(i + 1, len(fieldsets)):
            overlap = fieldsets[i] & fieldsets[j]
            assert not overlap, f"variants {i},{j} share features {overlap}"


def test_dry_run_pipeline_per_rung_files(tmp_path):
    rc = L.main(["--dry-run", "--rungs", "2", "--variants", "2",
                 "--workdir", str(tmp_path / "wd"), "--out-dir", str(tmp_path / "corpora")])
    assert rc == 0
    import json
    for d in range(2):
        f = tmp_path / "corpora" / f"decision_ladder_d{d}.jsonl"
        assert f.exists()
        items = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        assert len(items) == 2 * A.ITEMS_PER_DOMAIN        # 2 variants x ITEMS_PER_DOMAIN
        assert {it["domain"] for it in items} == {f"lad_d{d}_v0", f"lad_d{d}_v1"}
        assert {it["oracle"] for it in items} == {"approve", "deny"}  # both verdict buckets present
