"""Shared fixtures. All tests here are offline — no network, no live platform."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def loan_query_result() -> dict:
    """A real (trimmed) ``platforms.query`` response recorded from a verified
    platform on 2026-07-10 — the live ``QueryExplanation`` shape."""
    return json.loads((FIXTURES / "loan_query_result.json").read_text())
