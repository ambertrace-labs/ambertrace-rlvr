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
        assert len(spec.fields) == 1 + 2 * d          # flag + d (num, den) pairs
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


def test_render_description_has_negation_ratio_and_otherwise():
    desc = A.render_description(L.make_ladder_spec(2))
    assert "clearance_ok is false" in desc                                   # the lever
    assert "exceeds 80 percent of the monthly income" in desc                # a compute cond
    assert desc.strip().endswith("Otherwise decide approve.")                # residual = permit


def test_make_ladder_spec_rejects_out_of_range():
    with pytest.raises(ValueError):
        L.make_ladder_spec(-1)
    with pytest.raises(ValueError):
        L.make_ladder_spec(len(L.LADDER_PAIRS) + 1)


def test_build_rungs_caps_and_orders():
    rungs = L.build_rungs(99)
    assert len(rungs) == len(L.LADDER_PAIRS) + 1
    assert [s.key for s in rungs] == [f"lad_d{d}" for d in range(len(rungs))]


def test_dry_run_pipeline_smoke(tmp_path):
    out = tmp_path / "corpus.jsonl"
    rc = L.main(["--dry-run", "--rungs", "2",
                 "--workdir", str(tmp_path / "wd"), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    import json
    items = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert items and {it["oracle"] for it in items} <= {"approve", "deny"}
    assert {it["domain"] for it in items} == {"lad_d0", "lad_d1"}
