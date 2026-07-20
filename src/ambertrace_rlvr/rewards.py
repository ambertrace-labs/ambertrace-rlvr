"""Reward shaping: an :class:`AmberReport` (+ optional gold label) -> a scalar reward.

The default shaper composes several bounded components (each in ``[0, 1]`` before
weighting) into a dense, hack-resistant reward, per the library spec §8. Every
component is logged in :class:`RewardBreakdown.components` for ablation. The shaper
is pure and deterministic; it never touches the network.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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
              gold: Any | None = None,
              criteria_gold: Mapping[str, Any] | None = None) -> RewardBreakdown:
        ...


@runtime_checkable
class FactProvenanceChecker(Protocol):
    """Grounds asserted facts against the prompt (anti-reward-hacking, #10).

    Returns the fraction (in ``[0, 1]``) of the facts asserted in the model's
    decision block that are *not* grounded in the prompt — i.e. invented or
    unsupported. This is orthogonal to kernel rejection: a fact can be perfectly
    well-formed (kernel-accepted) yet fabricated (absent from the prompt)."""

    def unsupported_fraction(self, facts: Mapping[str, Any], prompt: str) -> float:
        ...


@dataclass
class SubstringProvenanceChecker:
    """Default provenance checker: a fact is grounded if its value appears in the
    prompt text. Deliberately conservative — it only penalises facts it can
    positively check, so it never punishes a genuine completion for a fact the
    substring heuristic cannot express.

    * ``check_booleans`` — bools are exempt by default (``true``/``false`` almost
      always appear incidentally in a prompt); set ``True`` to require the literal
      word ``true``/``false`` in the prompt.
    * ``case_sensitive`` — match string/boolean values case-sensitively.
    """

    check_booleans: bool = False
    case_sensitive: bool = False

    def unsupported_fraction(self, facts: Mapping[str, Any], prompt: str) -> float:
        text = prompt if self.case_sensitive else prompt.lower()
        checkable = 0
        unsupported = 0
        for value in facts.values():
            grounded = self._grounded(value, text)
            if grounded is None:  # exempt / not checkable — out of the denominator
                continue
            checkable += 1
            if not grounded:
                unsupported += 1
        return unsupported / checkable if checkable else 0.0

    def _grounded(self, value: Any, text: str) -> bool | None:
        """``True``/``False`` grounded-or-not, or ``None`` for exempt values."""
        # bool is a subclass of int — must be tested first.
        if isinstance(value, bool):
            if not self.check_booleans:
                return None
            word = "true" if value else "false"
            return re.search(rf"\b{word}\b", text) is not None
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                token = str(int(value))
            else:
                token = str(value)
            escaped = re.escape(token)
            # Match the number only on non-digit boundaries so a fabricated
            # ``5000`` does not spuriously match inside a genuine ``25000``.
            return re.search(rf"(?<![\d.]){escaped}(?![\d.])", text) is not None
        if isinstance(value, str):
            needle = value.strip()
            if not needle:
                return None
            if not self.case_sensitive:
                needle = needle.lower()
            return needle in text
        # None / list / dict: not checkable by substring.
        return None


# Baseline component weights (spec §8). ``rejected_penalty`` is subtracted.
# ``unsupported_penalty`` (opt-in, #10) is also subtracted — see the shaper docs.
DEFAULT_WEIGHTS: dict[str, float] = {
    "format": 0.1,
    "certified": 0.5,
    "correctness": 1.0,
    "graded": 0.3,
    "rejected_penalty": 0.2,
    "unsupported_penalty": 0.3,
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
                           With per-criterion ``criteria_gold`` it is the fraction
                           of *required* criteria correctly derived; otherwise it
                           falls back to the fired/evaluated baseline heuristic.
      * ``rejected_penalty`` — fraction of asserted facts the kernel rejected
                           (subtracted; discourages hallucinated facts).
      * ``unsupported_penalty`` — fraction of asserted facts not grounded in the
                           prompt (subtracted; anti-reward-hacking, #10).

    ``rejected_penalty`` vs ``unsupported_penalty`` are orthogonal and *both*
    subtract: the former penalises facts the AmberTrace kernel rejected (malformed
    / out-of-schema), the latter penalises facts that are well-formed but
    fabricated — asserted in the decision block yet absent from the prompt. Between
    them a model can neither smuggle junk past the kernel nor invent prompt-free
    facts that trivially satisfy the rules. Provenance is *opt-in*: it is scored
    only when a :class:`FactProvenanceChecker` is supplied via the ``provenance``
    field, and defaults OFF (``unsupported_penalty`` is a no-op 0.0), so existing
    behaviour is unchanged.

    ``total = format·w + certified·w + correctness·w + graded·w − rejected·w
    − unsupported·w``, clipped to ``clip``. ``clip[0]`` is also the floor returned
    for an unparseable completion (see
    :func:`ambertrace_rlvr.verifier.build_reward_function`).
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    clip: tuple[float, float] = (-1.0, 2.0)
    require_supported_schema: bool = False
    provenance: FactProvenanceChecker | None = None

    def score(self, parsed: ParsedCompletion, report: AmberReport,
              gold: Any | None = None,
              criteria_gold: Mapping[str, Any] | None = None) -> RewardBreakdown:
        w = self.weights
        c: dict[str, float] = {}

        c["format"] = 1.0  # only scored when the completion already parsed
        c["certified"] = 1.0 if report.proof_checked else 0.0
        c["correctness"] = self._correctness(parsed, report, gold)
        c["graded"] = self._graded(report, criteria_gold)
        c["rejected_penalty"] = self._rejected_fraction(report)
        c["unsupported_penalty"] = self._unsupported_fraction(parsed)

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
            - w.get("unsupported_penalty", 0.0) * c["unsupported_penalty"]
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

    def _graded(self, report: AmberReport,
                criteria_gold: Mapping[str, Any] | None = None) -> float:
        """Dense partial credit from the certified derivation. Zero on an
        uncertified report.

        When the domain supplies ``criteria_gold`` (a mapping of rule name ->
        expected ``fired`` bool), this is genuine per-criterion partial credit:
        the fraction of the *required* criteria whose certified firing matches
        the expectation. This is the "right answer for the right reasons" signal
        (spec §6.3, §8) — it can only rise as more required criteria are derived
        correctly, and a fully-correct derivation scores ``1.0`` (never more than
        a clean certified completion overall).

        ``RuleFiring.required`` is optional (RFC dense-reward contract): if the
        certified trace exposes no required criteria to grade against, the dense
        per-criterion signal is untrustworthy and degrades to zero-weight rather
        than guessing. Domains without per-criterion gold keep the documented
        baseline heuristic — the fraction of evaluated rules that fired."""
        if not report.proof_checked or not report.rules:
            return 0.0

        if criteria_gold:
            # Grade over the required criteria the domain gave an expectation for.
            graded = [r for r in report.required_rules if r.name in criteria_gold]
            if not graded:  # no required criterion to grade -> zero-weight
                return 0.0
            correct = sum(1 for r in graded if r.fired == bool(criteria_gold[r.name]))
            return _clip01(correct / len(graded))

        # Baseline heuristic: how much of the domain's criteria the certified
        # reasoning activated.
        evaluated = len(report.rules)
        fired = len(report.rules_fired)
        return _clip01(fired / evaluated) if evaluated else 0.0

    def _unsupported_fraction(self, parsed: ParsedCompletion) -> float:
        """Fraction of asserted facts not grounded in the prompt (opt-in, #10).

        A no-op (0.0) when no provenance checker is configured, the parser did
        not capture the prompt, or there are no facts to check — so the default
        shaper is unchanged."""
        if self.provenance is None or parsed.prompt is None or not parsed.facts:
            return 0.0
        return _clip01(self.provenance.unsupported_fraction(parsed.facts, parsed.prompt))

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
