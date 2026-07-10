"""Binding to an existing AmberTrace verified platform."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .parsers import CompletionParser, JSONBlockParser

DEFAULT_BASE_URL = "https://app.ambertrace.ai"


@dataclass
class VerifiableDomain:
    """A domain expressed as a bound AmberTrace platform + a completion parser.

    The library is read-only against AmberTrace: it queries this platform, never
    builds or mutates it. Use a scoped, platform-only API key for training jobs.
    """

    platform_id: int
    parser: CompletionParser
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls, platform_id: int | None = None, *,
                 parser: CompletionParser | None = None) -> "VerifiableDomain":
        """Build from ``AMBERTRACE_API_KEY`` / ``AMBERTRACE_BASE_URL`` /
        ``AMBERTRACE_PLATFORM_ID`` (mirrors the SDK's own env fallbacks)."""
        pid = platform_id if platform_id is not None else _int_env("AMBERTRACE_PLATFORM_ID")
        if pid is None:
            raise ValueError("platform_id not given and AMBERTRACE_PLATFORM_ID unset")
        return cls(
            platform_id=pid,
            parser=parser or JSONBlockParser(),
            api_key=os.environ.get("AMBERTRACE_API_KEY"),
            base_url=os.environ.get("AMBERTRACE_BASE_URL", DEFAULT_BASE_URL),
        )


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
