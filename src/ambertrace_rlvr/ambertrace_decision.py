"""AmberTrace as a decision *contestant* — the verified counterpart to the
System-1 deciders (Jev, Nimble).

Unlike a logit model, AmberTrace does not emit a confident guess: it queries a
**verified platform** and either returns a *certified* decision (``proof_checked``)
or **fails closed** (an uncertifiable verified query is refused, not answered). So
the contestant maps a certified `decision` onto the chosen label, and maps every
non-certified outcome — fail-closed 503, rejected facts, out-of-support inputs — to
a *refusal* (`choice=None`). The alignment scorer counts that refusal as such, never
as a wrong label: that is the whole point of the comparison.

Two layers, so the mapping is testable without the network:

* :func:`parse_case_facts` + :class:`AmberTraceDecider` — pure, offline: turn a
  corpus prompt into facts and an :class:`~ambertrace_rlvr.reports.AmberReport` into
  a :class:`~ambertrace_rlvr.model_backend.ScoredDecision`. Injected ``query_fn``.
* :func:`build_verified_platform` + :meth:`AmberTraceDecider.from_platform` — thin
  live glue over the published ``ambertraceai`` SDK (author → build → query), used to
  author a verified profile for a corpus domain and wire the query. Live-only.

The oracle-contamination caveat still holds (see docs/DECISION_MODEL_COMPARISON.md):
score AmberTrace only where it did **not** author the labels.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .model_backend import ScoredDecision
from .reports import AmberReport

# A live query seam: focal facts -> a normalised certification report. Fail-closed
# (a 503 / SDK error resolves to AmberReport.floor(...), never raises) so a whole
# run never dies on one refusal.
QueryFn = Callable[["dict[str, Any]"], AmberReport]

_FACT_LINE = re.compile(r"^\s*[-*]\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$")


def parse_case_facts(prompt: str) -> dict[str, Any]:
    """Pull the ``Case facts:`` block out of a decision prompt into a facts dict.

    Reads ``- field: value`` lines after a ``facts``/``case`` heading, stopping at
    the choice instruction or a blank separator. Numbers are coerced to int/float,
    booleans recognised, everything else left as a string — the shape
    ``platforms.query(facts=...)`` expects."""
    facts: dict[str, Any] = {}
    in_block = False
    for line in prompt.splitlines():
        low = line.strip().lower()
        if not in_block:
            if ("case fact" in low or low.endswith("facts:") or low == "facts"):
                in_block = True
            continue
        if low.startswith("choose ") or low.startswith("respond ") or low.startswith("answer"):
            break
        m = _FACT_LINE.match(line)
        if m:
            facts[m.group(1)] = _coerce(m.group(2))
        elif not line.strip() and facts:
            break  # blank line ends the block once we've collected something
    return facts


def _coerce(value: str) -> Any:
    v = value.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@dataclass
class AmberTraceDecider:
    """Decide via a verified AmberTrace query. ``query_fn`` maps facts to a
    certification report; ``fact_parser`` extracts the facts from the prompt.

    A decision is returned only when the query is *certified* (``proof_checked``)
    **and** the certified verb is one of the candidates. Anything else — a
    fail-closed report, a non-certified answer, or a verb outside the label space —
    is a refusal, mirroring AmberTrace's own fail-safe."""

    query_fn: QueryFn
    fact_parser: Callable[[str], dict[str, Any]] = field(default=parse_case_facts)

    def decide(self, prompt: str, label_space: "Any") -> ScoredDecision:
        report = self.query_fn(self.fact_parser(prompt))
        decision = report.decision
        if not report.proof_checked or not isinstance(decision, str) or decision not in label_space:
            return ScoredDecision(choice=None)     # fail-closed / not certified
        # A certified decision is proven, not probabilistic: mass on the verb, and
        # the platform's overall confidence as the reported confidence.
        confidence = report.confidence if report.confidence else 1.0
        return ScoredDecision(choice=decision, probabilities={decision: 1.0},
                              confidence=confidence)

    @classmethod
    def from_platform(
        cls, api: Any, platform_id: int, *,
        query: str = "What is the decision?",
        fact_parser: Callable[[str], dict[str, Any]] = parse_case_facts,
    ) -> AmberTraceDecider:
        """Wire a decider to an already-built verified platform. The query is
        fail-closed: a verified refusal (503 / ``service_unavailable``) or any SDK
        error becomes an :class:`AmberReport.floor`, so it scores as a refusal."""
        from ambertraceai import AmbertraceError  # lazy: SDK only needed live

        def query_fn(facts: dict[str, Any]) -> AmberReport:
            try:
                result = api.platforms.query(platform_id, query=query, facts=facts)
            except AmbertraceError as e:
                if getattr(e, "status_code", None) == 503 or getattr(e, "code", "") == "service_unavailable":
                    return AmberReport.floor(reason="verified_fail_closed")
                return AmberReport.from_error(e)
            return AmberReport.from_query_result(result)

        return cls(query_fn=query_fn, fact_parser=fact_parser)


def build_verified_platform(
    api: Any, *, name: str, policy: str, dataset_path: str,
    tau: float = 0.6, timeout: float = 600.0,
) -> int:
    """Author a verified platform for one decision domain and return its id
    (author → upload features → build ontology → build verified platform, polling
    each async build). Live-only glue over the published SDK; mirrors the demos'
    flow. Raises on a failed build rather than shipping an unverified platform."""
    domain = api.domains.create(name=name, description=policy)
    domain_id = domain["id"]
    dataset = api.datasets.upload(domain_id=domain_id, file_path=str(dataset_path))
    onto_job = api.domains.build_ontology(domain_id)
    api.wait_for_job(_job_id(onto_job), timeout=timeout)
    platform = api.platforms.create(
        domain_id=domain_id, dataset_id=dataset["id"],
        verified_profile=True, verified_min_confidence=tau,
    )
    api.wait_for_job(platform.job_id, timeout=timeout)
    return int(platform.id)


def _job_id(envelope: Any) -> Any:
    """Extract a job id from a 202 build envelope (dict or attr-style)."""
    if isinstance(envelope, dict):
        return envelope.get("job_id") or envelope.get("id")
    return getattr(envelope, "job_id", None) or getattr(envelope, "id", None)
