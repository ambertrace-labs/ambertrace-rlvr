"""Completion parsers: extract the query + facts to verify from a model output.

Domain-specific by nature (how you pull facts out of a completion), but the two
built-ins cover the common case of a machine-readable decision block. A parser
returns ``None`` for anything it cannot parse — the caller floors the reward.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ParsedCompletion:
    """The payload sent to AmberTrace, plus the model's own answer for scoring."""

    query: str
    facts: dict[str, Any]
    proposed_answer: Any | None = None
    relations: dict[str, list[dict[str, Any]]] | None = None
    raw_block: str | None = None


@runtime_checkable
class CompletionParser(Protocol):
    def parse(self, prompt: str, completion: str) -> ParsedCompletion | None:
        """Return the query + facts to verify, or ``None`` if unparseable."""
        ...


# Default decision-block delimiters (see prompts.py for the format contract).
_DECISION_RE = re.compile(r"<decision>\s*(.*?)\s*</decision>", re.DOTALL | re.IGNORECASE)


@dataclass
class JSONBlockParser:
    """Parse a ``<decision>{...}</decision>`` JSON block.

    ``answer_key`` names the field holding the model's verdict; ``facts_key`` the
    field holding the structured facts to verify. ``query_template`` is formatted
    with the parsed facts to build the natural-language query.
    """

    answer_key: str = "classification"
    facts_key: str = "facts"
    relations_key: str | None = "relations"
    query_template: str = "Classify: {facts}"
    block_pattern: re.Pattern[str] = field(default=_DECISION_RE)

    def parse(self, prompt: str, completion: str) -> ParsedCompletion | None:
        block = _extract_block(completion, self.block_pattern)
        if block is None:
            return None
        try:
            data = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, Mapping):
            return None

        facts = data.get(self.facts_key)
        if not isinstance(facts, Mapping):
            return None
        facts = dict(facts)

        relations = None
        if self.relations_key and isinstance(data.get(self.relations_key), Mapping):
            relations = dict(data[self.relations_key])

        # `facts` always wins over a same-named key in the parsed data.
        fmt_kwargs = {**data, "facts": facts}
        try:
            query = self.query_template.format(**fmt_kwargs)
        except (KeyError, IndexError, ValueError):
            query = self.query_template

        return ParsedCompletion(
            query=query,
            facts=facts,
            proposed_answer=data.get(self.answer_key),
            relations=relations,
            raw_block=block,
        )


@dataclass
class RegexBlockParser:
    """Parse facts from ``key: value`` lines inside the decision block — for models
    that emit a light structured block rather than strict JSON."""

    answer_pattern: str = r"(?im)^\s*answer\s*:\s*(.+?)\s*$"
    fact_pattern: str = r"(?im)^\s*fact\s+([\w.-]+)\s*:\s*(.+?)\s*$"
    query_template: str = "Classify: {facts}"
    block_pattern: re.Pattern[str] = field(default=_DECISION_RE)

    def parse(self, prompt: str, completion: str) -> ParsedCompletion | None:
        block = _extract_block(completion, self.block_pattern) or completion
        facts = {k: _coerce(v) for k, v in re.findall(self.fact_pattern, block)}
        if not facts:
            return None
        answer_match = re.search(self.answer_pattern, block)
        try:
            query = self.query_template.format(facts=facts)
        except (KeyError, IndexError, ValueError):
            query = self.query_template
        return ParsedCompletion(
            query=query,
            facts=facts,
            proposed_answer=answer_match.group(1).strip() if answer_match else None,
            raw_block=block,
        )


def _extract_block(completion: str, pattern: re.Pattern[str]) -> str | None:
    if not completion:
        return None
    m = pattern.search(completion)
    return m.group(1) if m else None


def _coerce(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            continue
    return s
