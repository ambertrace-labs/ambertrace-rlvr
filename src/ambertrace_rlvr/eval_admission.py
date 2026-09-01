"""Admission pipeline for agent-authored certified evals (#103).

An untrusted generator cannot poison the benchmark: every candidate item's label
is derived from the oracle regardless of who wrote the item. Item *quality*
comes from the generator; item *truth* comes from the proof. This module is the
trust core that makes that property hold.

Network-free by construction: the admission logic is pure validation over oracle
results. The verifier is injectable (``FakeVerifier`` for offline tests,
``AmberVerifier`` for live certification).

Three concerns:
1. **Certification** — each candidate's facts are verified by the oracle; only
   certified-decidable items with the intended properties are admitted.
2. **Deduplication** — content-addressed against an existing corpus, so a
   generator cannot re-propose shipped items.
3. **Train/eval separation** — the ``AdmissionRegistry`` records item/environment
   content hashes and their role, so items admitted to an eval set are excluded
   from training environments (and vice versa).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .corpus import DecisionItem
from .eval_oracle import JudgmentSpec, LabelSpec, OracleJudgment
from .evaluation import VerifierLike
from .parsers import ParsedCompletion

logger = logging.getLogger(__name__)

# Maximum prompt length (chars) — a sanity bound, not a quality gate.
_MAX_PROMPT_LEN = 8_000
_MIN_PROMPT_LEN = 20

# Batch chunk size matching the verifier's own limit (#101).
_BATCH_CHUNK = 50


@dataclass(frozen=True)
class GeneratorProvenance:
    """Who/what generated a candidate and when — carried through to the admitted
    item's metadata for audit."""

    generator_id: str
    seed: int | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class CandidateItem:
    """A proposed eval item pre-certification. The generator supplies the prompt,
    facts, intended classification (stratum/band/direction), and its own
    provenance. The oracle supplies the truth; admission validates that the truth
    matches the intent."""

    prompt: str
    facts: dict[str, Any]
    vocabulary: tuple[LabelSpec, ...]
    intended_oracle: str
    intended_direction: str  # "restrictive" | "permissive"
    intended_stratum: str  # difficulty family, e.g. "ratio", "negation"
    intended_band: str  # severity band, e.g. "restrictive", "permissive"
    provenance: GeneratorProvenance
    difficulty: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    query: str | None = None


@dataclass(frozen=True)
class AdmissionResult:
    """The outcome of certifying one candidate. ``admitted`` candidates carry a
    ``DecisionItem``; rejected candidates carry ``reasons`` (a list of human-readable
    strings) and optionally the oracle's actual verdict for generator diagnostics."""

    candidate: CandidateItem
    admitted: bool
    item: DecisionItem | None = None
    reasons: tuple[str, ...] = ()
    actual_oracle: str | None = None
    actual_direction: str | None = None


def _content_hash(prompt: str, facts: dict[str, Any]) -> str:
    """Content-addressed key for deduplication: SHA-256 of (prompt, facts)."""
    payload = json.dumps({"prompt": prompt, "facts": facts},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def certify_candidates(
    candidates: Sequence[CandidateItem],
    verifier: VerifierLike,
    *,
    spec: JudgmentSpec | None = None,
    existing_hashes: frozenset[str] = frozenset(),
    registry: AdmissionRegistry | None = None,
) -> list[AdmissionResult]:
    """Certify a batch of candidates through the oracle and validate each one.

    Batch-shaped: candidates are chunked through ``verifier.verify_batch`` in
    groups of up to 50 (the #101 batch limit). Each candidate is then validated
    against:

    1. Oracle certification — must be certified-decidable (not undecidable, not
       unverifiable).
    2. Intended-vs-actual property match — the oracle's verdict, direction, and
       severity band must match what the generator intended.
    3. Prompt sanity — bounded length, non-empty.
    4. Content-addressed deduplication — not already in the existing corpus.
    5. Registry conflict — not already assigned to a conflicting role.

    Parameters
    ----------
    candidates:
        The candidate items to certify.
    verifier:
        Satisfies ``VerifierLike`` — ``FakeVerifier`` offline, ``AmberVerifier``
        live.
    spec:
        The judgment spec (label vocabulary). If ``None``, built from the first
        candidate's vocabulary.
    existing_hashes:
        Content hashes of items already in the eval corpus (for dedup).
    registry:
        The train/eval separation ledger. When provided, admitted items are
        registered as ``"eval"`` and conflicts are checked.
    """
    if not candidates:
        return []

    judgment_spec = spec or JudgmentSpec(labels=list(candidates[0].vocabulary))

    # Build ParsedCompletions for the verifier (batch-shaped, chunked).
    parsed_list: list[ParsedCompletion | None] = [
        ParsedCompletion(
            query=c.query or c.prompt,
            facts=dict(c.facts),
        )
        for c in candidates
    ]
    # Chunk through verify_batch in groups of _BATCH_CHUNK.
    all_reports: list[Any] = [None] * len(candidates)
    for start in range(0, len(parsed_list), _BATCH_CHUNK):
        chunk = parsed_list[start:start + _BATCH_CHUNK]
        chunk_reports = verifier.verify_batch(chunk)
        for i, report in enumerate(chunk_reports):
            all_reports[start + i] = report

    results: list[AdmissionResult] = []
    seen_hashes: set[str] = set()

    for idx, (candidate, report) in enumerate(zip(candidates, all_reports)):
        reasons: list[str] = []

        # Prompt sanity.
        if len(candidate.prompt) < _MIN_PROMPT_LEN:
            reasons.append(f"prompt too short ({len(candidate.prompt)} < {_MIN_PROMPT_LEN})")
        if len(candidate.prompt) > _MAX_PROMPT_LEN:
            reasons.append(f"prompt too long ({len(candidate.prompt)} > {_MAX_PROMPT_LEN})")

        # Content-addressed dedup against existing corpus.
        ch = _content_hash(candidate.prompt, candidate.facts)
        if ch in existing_hashes:
            reasons.append("duplicate of existing corpus item")
        if ch in seen_hashes:
            reasons.append("duplicate within this batch")
        seen_hashes.add(ch)

        # Registry conflict check.
        if registry is not None:
            conflict = registry.check_conflict(ch, "eval")
            if conflict is not None:
                reasons.append(f"registry conflict: {conflict}")

        # Oracle certification.
        if report is None:
            reasons.append("no oracle report (verify returned None)")
            results.append(AdmissionResult(
                candidate=candidate, admitted=False, reasons=tuple(reasons)))
            continue

        judgment = OracleJudgment.from_report(report, judgment_spec)

        if judgment.certified_undecidable:
            reasons.append("oracle certified undecidable — not a decidable eval item")
        elif not judgment.certified:
            reasons.append(f"oracle did not certify: {judgment.reason}")

        actual_oracle = judgment.value
        actual_direction: str | None = None

        # Intended-vs-actual property match (only meaningful when certified).
        if judgment.certified and actual_oracle is not None:
            actual_direction = judgment_spec.severity_band(actual_oracle)

            if actual_oracle.strip().lower() != candidate.intended_oracle.strip().lower():
                reasons.append(
                    f"intended oracle '{candidate.intended_oracle}' != "
                    f"actual '{actual_oracle}'"
                )
            if actual_direction != candidate.intended_band:
                reasons.append(
                    f"intended band '{candidate.intended_band}' != "
                    f"actual band '{actual_direction}'"
                )

        admitted = len(reasons) == 0
        item: DecisionItem | None = None
        if admitted and judgment.certified and actual_oracle is not None:
            item = DecisionItem(
                id=candidate.id or f"gen-{idx:04d}",
                domain="",
                prompt=candidate.prompt,
                vocabulary=candidate.vocabulary,
                oracle=str(actual_oracle),
                undecidable=False,
                difficulty={
                    **candidate.difficulty,
                    "family": candidate.intended_stratum,
                    "structure": candidate.intended_stratum,
                },
            )
            # Register the admitted item.
            if registry is not None:
                registry.register(ch, "eval", provenance=candidate.provenance.generator_id)

        results.append(AdmissionResult(
            candidate=candidate,
            admitted=admitted,
            item=item,
            reasons=tuple(reasons),
            actual_oracle=str(actual_oracle) if actual_oracle is not None else None,
            actual_direction=actual_direction,
        ))

    return results


def corpus_content_hashes(items: Sequence[DecisionItem]) -> frozenset[str]:
    """Content hashes for an existing corpus, for dedup in ``certify_candidates``."""
    return frozenset(_content_hash(it.prompt, dict(zip(
        [v.verb for v in it.vocabulary],
        [v.rank for v in it.vocabulary],
    ))) for it in items)


def corpus_prompt_hashes(items: Sequence[DecisionItem]) -> frozenset[str]:
    """Prompt-only hashes — a looser dedup that catches reformulations with
    different fact sets."""
    return frozenset(
        hashlib.sha256(it.prompt.encode("utf-8")).hexdigest()
        for it in items
    )


# ---------------------------------------------------------------------------
# AdmissionRegistry — the train/eval separation ledger
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    """One entry: a content hash assigned to a role with optional model
    conditioning (for adversarial items mined against a specific model)."""

    content_hash: str
    role: str  # "eval" | "train-env"
    model: str | None = None
    provenance: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"content_hash": self.content_hash, "role": self.role}
        if self.model is not None:
            d["model"] = self.model
        if self.provenance is not None:
            d["provenance"] = self.provenance
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegistryEntry:
        return cls(
            content_hash=str(d["content_hash"]),
            role=str(d["role"]),
            model=d.get("model"),
            provenance=d.get("provenance"),
        )


@dataclass
class AdmissionRegistry:
    """Train/eval separation ledger (#103).

    A JSON file mapping item/environment content hashes to their role
    (``"eval"`` vs ``"train-env"``) plus optional model conditioning for
    adversarial items. ``check_conflict()`` returns a reason string when
    registering a hash for a conflicting role, ``None`` when clear.

    The registry is deliberately append-only in memory — ``save()`` writes the
    current state, ``load()`` reads it. No delete, no in-place mutation of
    existing entries.
    """

    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    path: Path | None = None

    def register(
        self,
        content_hash: str,
        role: str,
        *,
        model: str | None = None,
        provenance: str | None = None,
    ) -> None:
        """Register a hash for a role. Overwrites if same hash, same role;
        raises on conflict (use ``check_conflict`` first to get a reason)."""
        existing = self.entries.get(content_hash)
        if existing is not None and existing.role != role:
            raise ValueError(
                f"content hash {content_hash[:12]}... already registered as "
                f"'{existing.role}', cannot re-register as '{role}'"
            )
        self.entries[content_hash] = RegistryEntry(
            content_hash=content_hash, role=role,
            model=model, provenance=provenance,
        )

    def check_conflict(self, content_hash: str, intended_role: str) -> str | None:
        """Return a reason string if registering ``content_hash`` for
        ``intended_role`` would conflict with an existing entry, else ``None``."""
        existing = self.entries.get(content_hash)
        if existing is None:
            return None
        if existing.role == intended_role:
            return None  # same role — no conflict
        return (
            f"hash {content_hash[:12]}... is '{existing.role}', "
            f"cannot assign '{intended_role}'"
        )

    def hashes_for_role(self, role: str) -> frozenset[str]:
        """All content hashes assigned to ``role``."""
        return frozenset(
            e.content_hash for e in self.entries.values() if e.role == role
        )

    def save(self, path: Path | None = None) -> Path:
        """Write the registry to a JSON file (one entry per line for diffability)."""
        out = path or self.path
        if out is None:
            raise ValueError("no path specified for registry save")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        entries = [e.to_dict() for e in self.entries.values()]
        out.write_text(json.dumps(entries, indent=2) + "\n")
        return out

    @classmethod
    def load(cls, path: Path) -> AdmissionRegistry:
        """Load a registry from a JSON file."""
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text())
        entries: dict[str, RegistryEntry] = {}
        for d in raw:
            entry = RegistryEntry.from_dict(d)
            entries[entry.content_hash] = entry
        return cls(entries=entries, path=path)
