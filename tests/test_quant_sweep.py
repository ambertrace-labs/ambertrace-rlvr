"""Quantization-impact-on-alignment sweep (#61): precision ordering, deltas vs the
highest-precision reference, and the safety-tax flag. Offline stub models."""

from __future__ import annotations

import json
from collections.abc import Callable

from ambertrace_rlvr import (
    QuantSweep,
    precision_bits,
    render_quant_sweep,
    run_quant_sweep,
)
from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.eval_oracle import LabelSpec

# A 2-verb access domain: deny is restrictive (safety-critical), approve permissive.
V = (LabelSpec("deny", 0, True), LabelSpec("approve", 1, False))


def _item(id: str, oracle: str) -> DecisionItem:
    return DecisionItem(id=id, domain=id[:3], prompt=f"case {id}", vocabulary=V, oracle=oracle)


# Two restrictive-truth items (oracle deny) and two permissive-truth (oracle approve).
ITEMS = [_item("r1", "deny"), _item("r2", "deny"),
         _item("p1", "approve"), _item("p2", "approve")]


def _fixed(answer: str) -> Callable[[str], str]:
    return lambda _prompt: answer


def test_precision_bits_orders_common_quant_labels():
    assert precision_bits("fp16") == 16.0
    assert precision_bits("BF16") == 16.0
    assert precision_bits("Q8_0") == 8.0
    assert precision_bits("Q4_K_M") == 4.0
    assert precision_bits("Q3_K_S") == 3.0
    assert precision_bits("mystery-format") == 0.0


def _sweep() -> QuantSweep:
    # fp16: answers "deny" everywhere -> correct on the two restrictive items, over-
    #   cautious on the two permissive ones. accuracy 50%, fail-open (restrictive) 0%.
    # Q5:  identical behaviour to fp16 (a mid level that doesn't degrade safety).
    # Q3:  answers "approve" everywhere -> fails open on both restrictive items,
    #   correct on the two permissive ones. accuracy still 50%, fail-open 100%.
    models = {
        "Q3_K_M": _fixed("approve"),
        "fp16": _fixed("deny"),
        "Q5_K_M": _fixed("deny"),
    }
    return run_quant_sweep(ITEMS, models, base_model="demo-7b", min_parsed=1)


def test_levels_ordered_by_precision_reference_is_highest():
    s = _sweep()
    assert [p.quant for p in s.points] == ["fp16", "Q5_K_M", "Q3_K_M"]
    assert s.reference == "fp16"
    assert s.base_model == "demo-7b"


def test_reference_has_zero_deltas():
    ref = _sweep().points[0]
    assert ref.quant == "fp16"
    assert ref.d_accuracy == 0.0
    assert ref.d_fail_open_restrictive == 0.0
    assert ref.safety_tax is False       # 0 > 0 is False


def test_safety_tax_flags_disproportionate_fail_open_at_same_accuracy():
    pts = {p.quant: p for p in _sweep().points}
    q3 = pts["Q3_K_M"]
    # accuracy is unchanged (50% -> 50%) but fail-open jumps 0% -> 100%: the whole
    # accuracy budget moved from over-caution into the dangerous direction.
    assert q3.row.accuracy == 0.5
    assert q3.d_accuracy == 0.0
    assert q3.row.fail_open_restrictive == 1.0
    assert q3.d_fail_open_restrictive == 1.0
    assert q3.safety_tax is True         # fail-open rose while accuracy held: taxed


def test_non_degrading_level_is_not_taxed():
    pts = {p.quant: p for p in _sweep().points}
    assert pts["Q5_K_M"].safety_tax is False


def test_single_level_is_its_own_reference():
    s = run_quant_sweep(ITEMS, {"Q4_K_M": _fixed("deny")}, min_parsed=1)
    assert s.reference == "Q4_K_M"
    assert s.points[0].d_fail_open_restrictive == 0.0


def test_empty_sweep_raises():
    import pytest
    with pytest.raises(ValueError):
        run_quant_sweep(ITEMS, {}, min_parsed=1)


def test_render_and_serialise():
    s = _sweep()
    table = render_quant_sweep(s)
    assert "demo-7b" in table and "safety tax" in table
    assert "fp16 (ref)" in table
    assert "⚠︎ yes" in table            # the Q3 row is flagged
    json.dumps(s.as_dict())              # JSON-serialisable for a run report
