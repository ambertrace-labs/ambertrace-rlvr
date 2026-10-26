"""Cross-hack contagion comparison: do different reward hacks propagate with different lag, gain, and
breadth *within* the AmberTrace decision family?

Reads several ``decision_contagion`` summaries — one per hack, each already aggregated across that
hack's seeds — and tabulates, per hack: the headline (``signed_bias``) lag and propagation gain
``beta``, the **breadth** (how many out-of-domain dimensions track the in-domain hack), the
strongest-tracking dimension, and the final in-domain ``wedge_X``. The point of the pluggable
registry is exactly this comparison: which loopholes spread, and how fast/far.

Each input summary is expected to carry a ``within_family.hack_name`` (written by
``decision_contagion`` when the race has a ``split_manifest.json``); pass ``--labels`` to override.

Pure analysis: stdlib only (no torch/trl/network).

Copyright (c) 2026 AmberTrace(R) Labs Ltd. MIT Licensed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


@dataclass
class HackComparison:
    """One hack's contagion profile, aggregated across its seeds."""

    hack_name: str
    n_seeds: int
    final_wedge_x: float          # mean final in-domain wedge_X across seeds (did the hack install?)
    headline_lag: float
    headline_lag_std: float
    headline_beta: float          # propagation gain of the headline (signed_bias) dimension
    headline_beta_std: float
    breadth: int                  # number of OOD dimensions that track the in-domain hack
    tracked_dims: list[str]
    top_dim: str | None           # strongest-tracking dimension (largest beta)
    top_dim_beta: float
    top_dim_lag: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "hack_name": self.hack_name, "n_seeds": self.n_seeds,
            "final_wedge_x": self.final_wedge_x,
            "headline_lag": self.headline_lag, "headline_lag_std": self.headline_lag_std,
            "headline_beta": self.headline_beta, "headline_beta_std": self.headline_beta_std,
            "breadth": self.breadth, "tracked_dims": self.tracked_dims,
            "top_dim": self.top_dim, "top_dim_beta": self.top_dim_beta,
            "top_dim_lag": self.top_dim_lag,
        }


def _final_wedge_x(summary: dict[str, Any]) -> float:
    xs: list[float] = []
    for s in summary.get("seeds", []):
        traj = s.get("trajectory") or []
        if traj:
            xs.append(float(traj[-1].get("wedge_X", 0.0)))
    return sum(xs) / len(xs) if xs else 0.0


def compare_one(summary: dict[str, Any], *, label: str | None = None) -> HackComparison:
    """One hack's :class:`HackComparison` from its ``decision_contagion`` summary."""
    wf = summary.get("within_family") or {}
    hack = label or wf.get("hack_name") or "?"
    agg = summary.get("aggregate", {}) or {}
    tracked: list[str] = []
    best: tuple[str, float, float] | None = None  # (dim, beta, lag)
    for name, d in (summary.get("dimensions", {}) or {}).items():
        a = d.get("aggregate")
        if a and a.get("tracks_wedge_X"):
            tracked.append(name)
            beta = float(a.get("beta_mean", 0.0))
            if best is None or beta > best[1]:
                best = (name, beta, float(a.get("lag_mean", 0.0)))
    return HackComparison(
        hack_name=hack, n_seeds=int(agg.get("n_seeds", 0)),
        final_wedge_x=_final_wedge_x(summary),
        headline_lag=float(agg.get("lag_mean", 0.0)),
        headline_lag_std=float(agg.get("lag_std", 0.0)),
        headline_beta=float(agg.get("beta_mean", 0.0)),
        headline_beta_std=float(agg.get("beta_std", 0.0)),
        breadth=len(tracked), tracked_dims=sorted(tracked),
        top_dim=best[0] if best else None,
        top_dim_beta=best[1] if best else 0.0,
        top_dim_lag=best[2] if best else 0.0)


def compare_hacks(
    summaries: Sequence[dict[str, Any]], *, labels: Sequence[str] | None = None,
) -> list[HackComparison]:
    """Per-hack comparisons, strongest headline propagation first."""
    labs = list(labels) if labels is not None else [None] * len(summaries)
    if len(labs) != len(summaries):
        raise ValueError(f"labels ({len(labs)}) != summaries ({len(summaries)})")
    rows = [compare_one(s, label=lab) for s, lab in zip(summaries, labs)]
    return sorted(rows, key=lambda r: r.headline_beta, reverse=True)


def render_markdown(rows: Sequence[HackComparison]) -> str:
    out = [
        "| hack | seeds | final wedge_X | headline lag | headline beta | breadth | top dim (beta@lag) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        top = (f"{r.top_dim} ({r.top_dim_beta:+.3f}@{r.top_dim_lag:.1f})" if r.top_dim else "—")
        out.append(
            f"| {r.hack_name} | {r.n_seeds} | {r.final_wedge_x:.3f} | "
            f"{r.headline_lag:.2f}±{r.headline_lag_std:.2f} | "
            f"{r.headline_beta:+.3f}±{r.headline_beta_std:.3f} | {r.breadth} | {top} |")
    return "\n".join(out)


def run(args: argparse.Namespace) -> list[HackComparison]:
    paths = [Path(p) for spec in args.summaries for p in spec.split(",") if p.strip()]
    if not paths:
        raise SystemExit("--summaries is required")
    summaries = [json.loads(p.read_text()) for p in paths]
    labels = args.labels.split(",") if args.labels else None
    rows = compare_hacks(summaries, labels=labels)
    md = render_markdown(rows)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([r.as_dict() for r in rows], indent=2))
        args.out.with_suffix(".md").write_text(md + "\n")
        print(f"\nSummary -> {args.out} (+ .md)")
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="compare contagion (lag, beta, breadth) across reward hacks within a family")
    ap.add_argument("--summaries", action="append", default=[],
                    help="decision_contagion summary JSON(s), comma-separated or repeatable (one per hack)")
    ap.add_argument("--labels", default=None,
                    help="optional comma-separated hack labels (override within_family.hack_name)")
    ap.add_argument("--out", type=Path, default=None, help="comparison JSON path (+ a sibling .md)")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
