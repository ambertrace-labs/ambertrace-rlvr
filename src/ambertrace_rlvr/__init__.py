"""ambertrace-rlvr — RLVR with AmberTrace verified platforms as the reward source.

Copyright (c) 2026 Ambertrace Labs Ltd. MIT Licensed.
"""

from __future__ import annotations

from .config import (
    DatasetConfig,
    EvalConfig,
    RunConfig,
    TrainingConfig,
    load_run_config,
)
from .domain import VerifiableDomain
from .parsers import (
    CompletionParser,
    JSONBlockParser,
    ParsedCompletion,
    RegexBlockParser,
)
from .prompts import build_system_prompt, has_decision_block
from .reporting import build_run_report, write_run_report
from .reports import AmberReport, FiredRule, RejectedFact
from .rewards import DefaultRewardShaper, RewardBreakdown, RewardShaper
from .testing import FakeVerifier
from .verifier import AmberVerifier, RewardFunction, build_reward_function

__version__ = "0.1.0"

__all__ = [
    "VerifiableDomain",
    "load_run_config",
    "RunConfig",
    "TrainingConfig",
    "DatasetConfig",
    "EvalConfig",
    "CompletionParser",
    "ParsedCompletion",
    "JSONBlockParser",
    "RegexBlockParser",
    "AmberReport",
    "FiredRule",
    "RejectedFact",
    "RewardShaper",
    "RewardBreakdown",
    "DefaultRewardShaper",
    "AmberVerifier",
    "RewardFunction",
    "build_reward_function",
    "FakeVerifier",
    "build_system_prompt",
    "has_decision_block",
    "build_run_report",
    "write_run_report",
    "__version__",
]
