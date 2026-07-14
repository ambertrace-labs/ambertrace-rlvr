"""Reward shaping: an :class:`AmberReport` (+ optional gold label) -> a scalar reward.

The default shaper composes several bounded components (each in ``[0, 1]`` before
weighting) into a dense, hack-resistant reward, per the library spec §8. Every
component is logged in :class:`RewardBreakdown.components` for ablation. The shaper
is pure and deterministic; it never touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .parsers import ParsedCompletion
from .reports import AmberReport


@dataclass
class RewardBreakdown:
    total: float
    components: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class RewardShaper(Protocol):
    def score(self, parsed: ParsedCompletion, report: AmberReport,
              gold: Any | None = None) -> RewardBreakdown:
        ...


# Baseline component weights (spec §8). ``rejected_penalty`` is subtracted.
DEFAULT_WEIGHTS: dict[str, float] = {
    "format": 0.1,
    "certified": 0.5,
    "correctness": 1.0,
    "graded": 0.3,
    "rejected_penalty": 0.2,
}


@dataclass
class DefaultRewardShaper:
    """The documented baseline shaper.

    Components (each in ``[0, 1]``):
      * ``format``       — the completion parsed into a valid decision block.
      * ``certified``    — ``report.proof_checked`` (the hard verifiable core).
      * ``correctness``  — proposed answer vs gold (if given) else vs the certified
                           decision.
      * ``graded``       — dense partial credit from the certified rule firings.
      * ``rejected_penalty`` — fraction of asserted facts the kernel rejected
                           (subtracted; discourages hallucinated facts).

    ``total = format·w + certified·w + correctness·w + graded·w − rejected·w``,
    clipped to ``clip``. ``clip[0]`` is also the floor returned for an unparseable
    completion (see :func:`ambertrace_rlvr.verifier.build_reward_function`).
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    clip: tuple[float, float] = (-1.0, 2.0)
    require_supported_schema: bool = False

    def score(self, parsed: ParsedCompletion, report: AmberReport,
              gold: Any | None = None) -> RewardBreakdown:
        w = self.weights
        c: dict[str, float] = {}

        c["format"] = 1.0  # only scored when the completion already parsed
        c["certified"] = 1.0 if report.proof_checked else 0.0
        c["correctness"] = self._correctness(parsed, report, gold)
        c["graded"] = self._graded(report)
        c["rejected_penalty"] = self._rejected_fraction(report)

        # If we require a known schema and the report is on an unknown one, don't
        # trust the dense components — fall back to the certified core only.
        if self.require_supported_schema and not report.schema_supported:
            c["graded"] = 0.0

        total = (
            w.get("format", 0.0) * c["format"]
            + w.get("certified", 0.0) * c["certified"]
            + w.get("correctness", 0.0) * c["correctness"]
            + w.get("graded", 0.0) * c["graded"]
            - w.get("rejected_penalty", 0.0) * c["rejected_penalty"]
        )
        total = _clip(total, self.clip)
        return RewardBreakdown(total=total, components={**c, "total": total})

    # --- components --------------------------------------------------------
    def _correctness(self, parsed: ParsedCompletion, report: AmberReport,
                     gold: Any | None) -> float:
        proposed = parsed.proposed_answer
        if proposed is None:
            return 0.0
        target = gold if gold is not None else report.decision
        if target is None:
            return 0.0
        return 1.0 if _norm(proposed) == _norm(target) else 0.0

    def _graded(self, report: AmberReport) -> float:
        """Dense partial credit from the certified derivation. Baseline heuristic:
        the fraction of evaluated rules that fired (how much of the domain's
        criteria the certified reasoning activated). Domains with per-criterion gold
        should subclass/replace this. Zero on an uncertified report."""
        if not report.proof_checked or not report.rules:
            return 0.0
        evaluated = len(report.rules)
        fired = len(report.rules_fired)
        return _clip01(fired / evaluated) if evaluated else 0.0

    def _rejected_fraction(self, report: AmberReport) -> float:
        summary = report.fact_summary
        emitted = summary.get("emitted") or summary.get("accepted")
        rejected = summary.get("rejected", len(report.rejected_facts))
        if emitted:
            return _clip01(rejected / (emitted + rejected)) if (emitted + rejected) else 0.0
        # No summary counts (e.g. fail-closed error report): any rejected fact -> full penalty.
        return 1.0 if rejected else 0.0


def _clip(x: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm(v: Any) -> str:
    return str(v).strip().lower()
