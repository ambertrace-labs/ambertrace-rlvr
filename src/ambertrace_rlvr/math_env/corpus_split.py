"""Seeded, structure-stratified, disjoint-DOMAIN split of a decision corpus.

For the within-family contagion experiment: train a reward-hack lineage on one set of AmberTrace
domains and probe transfer to the *held-out* domains. Splitting by WHOLE DOMAIN is leakage-safe (no
cover story or vocabulary crosses the boundary); stratifying by difficulty structure guarantees both
halves cover every reasoning structure, so per-structure held-out transfer is measurable.

Pure / deterministic: stdlib only, no model, no network.

Copyright (c) 2026 AmberTrace(R) Labs Ltd. MIT Licensed.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..corpus import DecisionItem


@dataclass(frozen=True)
class CorpusSplit:
    """A disjoint-domain partition of a decision corpus."""

    train_items: list[DecisionItem]
    heldout_items: list[DecisionItem]
    train_domains: frozenset[str]
    heldout_domains: frozenset[str]
    structures: tuple[str, ...]  # structures present (sorted)

    def manifest(self) -> dict[str, Any]:
        """A JSON-serialisable record of the split, so the probe can reconstruct the held-out set."""
        return {
            "train_domains": sorted(self.train_domains),
            "heldout_domains": sorted(self.heldout_domains),
            "structures": list(self.structures),
        }


def _structure_of(item: DecisionItem, key: str) -> str:
    return str(item.difficulty.get(key, "?"))


def split_by_domain(
    items: Sequence[DecisionItem], *, train_fraction: float = 0.5, seed: int = 0,
    structure_key: str = "structure",
) -> CorpusSplit:
    """Disjoint-domain split, stratified by difficulty structure.

    Groups domains by their structure, shuffles the domains within each structure (seeded), and
    assigns the first ``ceil(train_fraction * n_domains)`` of each structure to train, the rest to
    held-out (clamped so each structure keeps >= 1 domain on each side). Deterministic for a given
    ``seed``. Every domain must belong to exactly one structure (true for ``decision_eval_v1``) — a
    domain whose items disagree on structure raises ``ValueError``. Raises if any structure has fewer
    than two domains (cannot split) or ``train_fraction`` is not in ``(0, 1)``."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    domain_structure: dict[str, str] = {}
    for it in items:
        st = _structure_of(it, structure_key)
        prev = domain_structure.setdefault(it.domain, st)
        if prev != st:
            raise ValueError(
                f"domain {it.domain!r} spans structures {prev!r} and {st!r}; the stratified "
                "disjoint-domain split assumes one structure per domain")

    domains_by_structure: dict[str, list[str]] = defaultdict(list)
    for dom, st in domain_structure.items():
        domains_by_structure[st].append(dom)

    rng = random.Random(seed)
    train_domains: set[str] = set()
    heldout_domains: set[str] = set()
    for st in sorted(domains_by_structure):
        doms = sorted(domains_by_structure[st])  # sort first -> shuffle is seed-reproducible
        if len(doms) < 2:
            raise ValueError(f"structure {st!r} has {len(doms)} domain(s); need >= 2 to split")
        rng.shuffle(doms)
        n_train = max(1, min(len(doms) - 1, math.ceil(train_fraction * len(doms))))
        train_domains.update(doms[:n_train])
        heldout_domains.update(doms[n_train:])

    train_items = [it for it in items if it.domain in train_domains]
    heldout_items = [it for it in items if it.domain in heldout_domains]
    return CorpusSplit(
        train_items=train_items, heldout_items=heldout_items,
        train_domains=frozenset(train_domains), heldout_domains=frozenset(heldout_domains),
        structures=tuple(sorted(domains_by_structure)))
