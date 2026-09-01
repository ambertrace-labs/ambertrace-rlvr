"""Coverage-gap analyzer (#103): offline tests for gap ranking, adversarial
target identification, and determinism. Pure data, no network."""

from __future__ import annotations

from ambertrace_rlvr.corpus import DecisionItem
from ambertrace_rlvr.deviation import DeviationReport
from ambertrace_rlvr.eval_gaps import (
    GapSpec,
    adversarial_targets,
    coverage_gaps,
    render_gaps,
)
from ambertrace_rlvr.eval_oracle import LabelSpec
from ambertrace_rlvr.matrix import AlignmentRow

VOCAB = (
    LabelSpec("deny", rank=0, restrictive=True),
    LabelSpec("escalate", rank=3, restrictive=True),
    LabelSpec("approve", rank=9, restrictive=False),
)


def _item(
    item_id: str,
    oracle: str = "deny",
    family: str = "baseline",
    structure: str = "baseline",
) -> DecisionItem:
    return DecisionItem(
        id=item_id,
        domain="test",
        prompt=f"Test prompt for {item_id}",
        vocabulary=VOCAB,
        oracle=oracle,
        undecidable=False,
        difficulty={"family": family, "structure": structure},
    )


# --- coverage_gaps ------------------------------------------------------------

def test_gaps_identify_undersampled_cells():
    items = [
        _item("a1", oracle="deny", family="baseline", structure="baseline"),
        _item("a2", oracle="deny", family="baseline", structure="baseline"),
        _item("a3", oracle="deny", family="baseline", structure="baseline"),
        _item("b1", oracle="approve", family="ratio", structure="ratio"),
    ]
    gaps = coverage_gaps(items)
    # The baseline/restrictive/baseline cell has 3 items; ratio/permissive/ratio has 1.
    # The populated ratio cell needs 2 more (3 - 1 = 2).
    ratio_populated = [g for g in gaps
                       if g.stratum == "ratio" and g.band == "permissive"
                       and g.structure == "ratio"]
    assert len(ratio_populated) == 1
    assert ratio_populated[0].wanted == 2
    assert ratio_populated[0].count == 1


def test_gaps_include_missing_cross_product_cells():
    items = [
        _item("a1", oracle="deny", family="baseline", structure="baseline"),
        _item("b1", oracle="approve", family="ratio", structure="ratio"),
    ]
    gaps = coverage_gaps(items)
    # Cross-product: {baseline, ratio} x {restrictive, permissive} x {baseline, ratio}
    # = 8 cells. 2 are populated, 6 are empty.
    empty_gaps = [g for g in gaps if g.count == 0]
    assert len(empty_gaps) == 6


def test_gaps_deterministic_ordering():
    items = [
        _item("a1", oracle="deny", family="alpha", structure="s1"),
        _item("a2", oracle="deny", family="alpha", structure="s1"),
        _item("b1", oracle="deny", family="beta", structure="s2"),
    ]
    gaps1 = coverage_gaps(items)
    gaps2 = coverage_gaps(items)
    assert [(g.stratum, g.band, g.structure) for g in gaps1] == \
           [(g.stratum, g.band, g.structure) for g in gaps2]


def test_gaps_custom_target():
    # Two cells so the cross-product generates missing cells.
    items = [
        _item("a1", oracle="deny", family="baseline", structure="baseline"),
        _item("b1", oracle="approve", family="ratio", structure="ratio"),
    ]
    gaps = coverage_gaps(items, target_per_cell=10)
    assert all(g.wanted <= 10 for g in gaps)
    # The populated cells (count=1) want 9; the empty cross-product cells want 10.
    empty_gaps = [g for g in gaps if g.count == 0]
    assert len(empty_gaps) > 0
    assert empty_gaps[0].wanted == 10


def test_gaps_empty_corpus():
    gaps = coverage_gaps([])
    assert gaps == []


def test_gaps_single_cell_no_gaps():
    """When all items are in one cell and target_per_cell equals the count,
    no gaps should be reported (the one cell is at target)."""
    items = [_item(f"a{i}", oracle="deny", family="baseline", structure="baseline")
             for i in range(5)]
    gaps = coverage_gaps(items, target_per_cell=5)
    assert gaps == []


def test_gaps_highest_priority_first():
    items = [
        _item("a1", oracle="deny", family="baseline", structure="baseline"),
        _item("a2", oracle="deny", family="baseline", structure="baseline"),
        _item("a3", oracle="deny", family="baseline", structure="baseline"),
        _item("b1", oracle="approve", family="ratio", structure="ratio"),
    ]
    gaps = coverage_gaps(items)
    if len(gaps) >= 2:
        assert gaps[0].priority >= gaps[1].priority


# --- adversarial_targets ------------------------------------------------------

def _make_deviation_report(
    *, correct: int = 0, over_permit: int = 0, over_deny: int = 0
) -> DeviationReport:
    """Build a DeviationReport with the given scored counts."""
    r = DeviationReport()
    r.correct = correct
    r.over_permit = over_permit
    r.over_deny = over_deny
    return r


def test_adversarial_targets_from_matrix():
    report = _make_deviation_report(correct=7, over_permit=3)
    band_report = _make_deviation_report(correct=7, over_permit=3)
    row = AlignmentRow(
        model="test-model",
        report=report,
        by_band={"restrictive": band_report},
    )
    targets = adversarial_targets([row], min_fail_open=0.0)
    assert len(targets) >= 1
    assert targets[0].model == "test-model"
    assert targets[0].fail_open_rate > 0


def test_adversarial_targets_filter_by_min_rate():
    report = _make_deviation_report(correct=99, over_permit=1)
    band_report = _make_deviation_report(correct=10)  # 0 over-permit
    row = AlignmentRow(
        model="safe-model",
        report=report,
        by_band={"restrictive": band_report},
    )
    targets = adversarial_targets([row], min_fail_open=0.05)
    # The band has 0 over-permit, so no targets above 5%.
    assert len(targets) == 0


def test_adversarial_targets_sorted_by_rate():
    r1 = _make_deviation_report(correct=9, over_permit=1)
    r2 = _make_deviation_report(correct=5, over_permit=5)
    row = AlignmentRow(
        model="test-model",
        report=DeviationReport(),
        by_band={"restrictive": r1, "permissive": r2},
    )
    targets = adversarial_targets([row])
    assert targets[0].fail_open_rate >= targets[-1].fail_open_rate


# --- render -------------------------------------------------------------------

def test_render_gaps_produces_markdown():
    gaps = [
        GapSpec("baseline", "restrictive", "baseline", count=1, wanted=2, priority=2.5),
    ]
    md = render_gaps(gaps)
    assert "baseline" in md
    assert "restrictive" in md
    assert "2.50" in md
