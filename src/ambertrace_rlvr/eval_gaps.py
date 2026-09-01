"""Coverage-gap analyzer for agent-authored eval generation (#103).

Reads an existing eval set and a matrix summary (``AlignmentRow`` structures or
their JSONL outputs) and emits a ranked ``GapSpec`` list — stratum x band x
structure with counts and wanted. Pure functions over data; no network, no
verifier, no model calls.

The gap ranking drives the coverage-mode proposer in ``gen_agent_evals.py``:
the agent fills the largest gaps first, so the generated items maximise
statistical power where confidence intervals are widest.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .corpus import DecisionItem
from .matrix import AlignmentRow


@dataclass(frozen=True)
class GapSpec:
    """One coverage gap: a stratum x band x structure cell with its current count
    and the number of items wanted to reach parity with the best-covered cell.

    ``priority`` is higher when more items are wanted — the proposer fills the
    highest-priority gaps first."""

    stratum: str
    band: str
    structure: str
    count: int
    wanted: int
    priority: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum, "band": self.band,
            "structure": self.structure, "count": self.count,
            "wanted": self.wanted, "priority": self.priority,
        }


@dataclass(frozen=True)
class AdversarialTarget:
    """A cell where a model fails in the unsafe direction — a starting point for
    adversarial mining. ``fail_open_rate`` is the model's over-permit rate on
    this stratum; higher rates suggest the model is more exploitable here."""

    stratum: str
    band: str
    structure: str
    fail_open_rate: float
    model: str


def coverage_gaps(
    items: Sequence[DecisionItem],
    *,
    target_per_cell: int | None = None,
    strata_key: str = "family",
    structure_key: str = "structure",
) -> list[GapSpec]:
    """Identify coverage gaps in an eval corpus.

    Each item is assigned to a (stratum, band, structure) cell via its
    ``difficulty`` tags and its vocabulary's severity band. Cells with fewer
    items than ``target_per_cell`` (default: the max count across all cells)
    generate a ``GapSpec`` with ``wanted = target - count``.

    Returns a list sorted by descending priority (largest gaps first), with
    ties broken alphabetically for determinism.
    """
    cell_counts: Counter[tuple[str, str, str]] = Counter()

    for it in items:
        stratum = str(it.difficulty.get(strata_key, "unknown"))
        structure = str(it.difficulty.get(structure_key, "unknown"))
        # Derive band from the oracle verdict + vocabulary.
        band = _item_band(it)
        cell_counts[(stratum, band, structure)] += 1

    if not cell_counts:
        return []

    max_count = max(cell_counts.values())
    target = target_per_cell if target_per_cell is not None else max_count

    gaps: list[GapSpec] = []
    for (stratum, band, structure), count in cell_counts.items():
        wanted = max(0, target - count)
        if wanted > 0:
            # Priority: wanted items, scaled by inverse count (sparser cells
            # get higher priority when wanted is tied).
            priority = wanted + (1.0 / (1 + count))
            gaps.append(GapSpec(
                stratum=stratum, band=band, structure=structure,
                count=count, wanted=wanted, priority=priority,
            ))

    # Also emit cells that are entirely missing from the cross-product.
    all_strata = {s for s, _, _ in cell_counts}
    all_bands = {b for _, b, _ in cell_counts}
    all_structures = {st for _, _, st in cell_counts}
    for s in all_strata:
        for b in all_bands:
            for st in all_structures:
                if (s, b, st) not in cell_counts:
                    priority = target + 1.0  # entirely missing > partially filled
                    gaps.append(GapSpec(
                        stratum=s, band=b, structure=st,
                        count=0, wanted=target, priority=priority,
                    ))

    # Sort: highest priority first, then alphabetical for determinism.
    gaps.sort(key=lambda g: (-g.priority, g.stratum, g.band, g.structure))
    return gaps


def adversarial_targets(
    rows: Sequence[AlignmentRow],
    *,
    min_fail_open: float = 0.0,
    strata_key: str = "family",
) -> list[AdversarialTarget]:
    """Identify cells where a model fails in the unsafe direction — starting
    points for adversarial mining.

    For each model's by-structure strata (from the complexity profile embedded
    in the ``AlignmentRow``), emit an ``AdversarialTarget`` when the stratum's
    fail-open rate exceeds ``min_fail_open``. Returns sorted by descending
    fail-open rate for deterministic ranking.
    """
    targets: list[AdversarialTarget] = []
    for row in rows:
        # Use per-band fail-open rates from the row's by_band decomposition.
        for band_name, band_report in row.by_band.items():
            rate = band_report.over_permit_rate
            if rate is not None and rate > min_fail_open:
                targets.append(AdversarialTarget(
                    stratum=strata_key,
                    band=band_name,
                    structure="any",
                    fail_open_rate=rate,
                    model=row.model,
                ))
    targets.sort(key=lambda t: (-t.fail_open_rate, t.model, t.band))
    return targets


def render_gaps(gaps: Sequence[GapSpec]) -> str:
    """Render the gap list as a markdown table."""
    header = (
        "| stratum | band | structure | count | wanted | priority |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [header]
    for g in gaps:
        lines.append(
            f"| {g.stratum} | {g.band} | {g.structure} "
            f"| {g.count} | {g.wanted} | {g.priority:.2f} |"
        )
    return "\n".join(lines)


def _item_band(it: DecisionItem) -> str:
    """Derive the severity band for an item from its oracle verdict and
    vocabulary. Falls back to ``"unknown"`` for undecidable items."""
    if it.undecidable or it.oracle is None:
        return "unknown"
    spec = it.spec()
    return spec.severity_band(it.oracle)
