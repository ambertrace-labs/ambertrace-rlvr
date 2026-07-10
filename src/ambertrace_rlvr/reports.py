"""Normalisation of the AmberTrace query response into a stable internal report.

The public SDK returns a ``QueryResult`` whose ``explanation`` is the typed
``QueryExplanation`` (SDK >= 1.0.3; emitted end-to-end by the platform as of the
2026-07-10 deploy). This module maps that — and the fail-closed error path — onto
:class:`AmberReport`, a small dataclass the reward shaper reads. It never raises:
a malformed or missing field degrades to a floor report, never an exception into
the training loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dc_field  # aliased: RejectedFact has a `field` attr
from typing import Any

# The explanation schema_version this normaliser was written against. When the
# platform bumps it, revisit the field mapping below.
SUPPORTED_SCHEMA_VERSION = 1


@dataclass
class FiredRule:
    """One rule from ``explanation.symbolic_trace.rules[]``. ``fired`` is the
    kernel-certified firing (reconciled against ``proof.firings``), not an engine
    self-report. ``required`` flags a hard obligation (a deny-family rule)."""

    name: str
    fired: bool
    rule_type: str | None = None
    required: bool = False
    reason: str | None = None


@dataclass
class RejectedFact:
    """One fact rejected at the certified-fact gate
    (``explanation.rejected_facts`` / ``AmbertraceError.rejected_facts``)."""

    field: str
    value: Any = None
    reasons: list[str] = dc_field(default_factory=list)


@dataclass
class AmberReport:
    """Normalised, reward-ready view of a single query certification."""

    proof_checked: bool
    confidence: float
    symbolic_confidence: float | None
    neural_confidence: float | None
    answer: Any | None
    decision: Any | None
    rules: list[FiredRule]
    rejected_facts: list[RejectedFact]
    fact_summary: dict[str, int]          # {accepted, emitted, rejected, witness_invalid}
    deciding_rules: list[dict[str, Any]]  # [{rule, reason}]
    proof_summary: str
    schema_version: int | None
    raw: dict[str, Any]
    error: str | None = None              # set when this is a fail-closed report

    # --- derived views -----------------------------------------------------
    @property
    def rules_fired(self) -> list[FiredRule]:
        return [r for r in self.rules if r.fired]

    @property
    def required_rules(self) -> list[FiredRule]:
        return [r for r in self.rules if r.required]

    @property
    def n_rejected(self) -> int:
        return self.fact_summary.get("rejected", len(self.rejected_facts))

    @property
    def schema_supported(self) -> bool:
        return self.schema_version == SUPPORTED_SCHEMA_VERSION

    # --- constructors ------------------------------------------------------
    @classmethod
    def floor(cls, reason: str, raw: Mapping[str, Any] | None = None) -> AmberReport:
        """A fail-closed report: nothing certified, no credit anywhere."""
        return cls(
            proof_checked=False, confidence=0.0, symbolic_confidence=None,
            neural_confidence=None, answer=None, decision=None, rules=[],
            rejected_facts=[], fact_summary={}, deciding_rules=[], proof_summary="",
            schema_version=None, raw=dict(raw or {}), error=reason,
        )

    @classmethod
    def from_query_result(cls, result: Mapping[str, Any]) -> AmberReport:
        """Normalise a successful ``platforms.query`` result. Defensive against
        missing keys so a contract drift degrades gracefully rather than crashes."""
        expl = _as_mapping(result.get("explanation"))
        conf = _as_mapping(expl.get("confidence"))
        summary = _as_mapping(expl.get("certified_fact_summary"))
        decision_block = _as_mapping(expl.get("decision"))

        return cls(
            proof_checked=bool(result.get("proof_checked")),
            confidence=_as_float_or(conf.get("overall"), 0.0),
            symbolic_confidence=_as_float(conf.get("symbolic_confidence")),
            neural_confidence=_as_float(conf.get("neural_confidence")),
            answer=result.get("answer"),
            decision=result.get("decision"),
            rules=_parse_rules(expl.get("symbolic_trace")),
            rejected_facts=_parse_rejected(expl.get("rejected_facts")),
            fact_summary={k: int(v) for k, v in summary.items() if _is_int(v)},
            deciding_rules=[dict(d) for d in _as_sequence(decision_block.get("deciding_rules"))
                            if isinstance(d, Mapping)],
            proof_summary=str(result.get("proof_summary") or ""),
            schema_version=_as_int(expl.get("schema_version")),
            raw=dict(result),
        )

    @classmethod
    def from_error(cls, err: Exception) -> AmberReport:
        """Fail-closed report from an SDK error. Pulls structured ``rejected_facts``
        (``[{field, value, reasons}]`` as of the 2026-07-10 deploy) off the error
        when present so ``rejected_penalty`` / fact-provenance still get a signal."""
        rejected = _parse_rejected(getattr(err, "rejected_facts", None))
        report = cls.floor(reason=str(err) or err.__class__.__name__)
        report.rejected_facts = rejected
        if rejected:
            report.fact_summary = {"rejected": len(rejected)}
        return report


# --- helpers ---------------------------------------------------------------

def _parse_rules(symbolic_trace: Any) -> list[FiredRule]:
    st = _as_mapping(symbolic_trace)
    out: list[FiredRule] = []
    for r in _as_sequence(st.get("rules")):
        if not isinstance(r, Mapping):
            continue
        name = r.get("rule_name") or r.get("name")
        if name is None:
            continue
        out.append(FiredRule(
            name=str(name),
            fired=bool(r.get("fired")),
            rule_type=_opt_str(r.get("rule_type")),
            required=bool(r.get("required", False)),
            reason=_opt_str(r.get("explanation")),
        ))
    return out


def _parse_rejected(rejected: Any) -> list[RejectedFact]:
    out: list[RejectedFact] = []
    for item in _as_sequence(rejected):
        if isinstance(item, Mapping):
            fld = item.get("field")
            if fld is None:
                continue
            out.append(RejectedFact(
                field=str(fld),
                value=item.get("value"),
                reasons=[str(x) for x in _as_sequence(item.get("reasons"))],
            ))
        elif isinstance(item, str):
            # Legacy shape (pre-2026-07-10): bare field-name strings.
            out.append(RejectedFact(field=item))
    return out


def _as_mapping(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}


def _as_sequence(v: Any) -> Sequence[Any]:
    return v if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) else []


def _as_float(v: Any) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float_or(v: Any, default: float) -> float:
    f = _as_float(v)
    return f if f is not None else default


def _as_int(v: Any) -> int | None:
    return int(v) if _is_int(v) else None


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _opt_str(v: Any) -> str | None:
    return str(v) if v is not None else None
