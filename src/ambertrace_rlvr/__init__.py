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
from .deviation import (
    DeviationReport,
    ModelAnswer,
    OracleItem,
    oracle_judgments,
    parse_model_answer,
    score_deviation,
)
from .domain import VerifiableDomain
from .faithfulness import (
    CandidateTrace,
    CurvePoint,
    CurveTrend,
    MonitorabilityComparison,
    compare_monitorability,
    curve_trend,
    faithfulness,
    faithfulness_curve,
    load_trajectory,
)
from .eval_oracle import (
    ABSTAIN,
    JudgmentSpec,
    LabelSpec,
    OracleJudgment,
)
from .evaluation import (
    EvalMetrics,
    EvalSample,
    Policy,
    VerifierLike,
    compare_to_baseline,
    constant_policy,
    consistency,
    evaluate,
    evaluate_policy,
    malformed_policy,
    run_policy,
)
from .parsers import (
    CompletionParser,
    JSONBlockParser,
    ParsedCompletion,
    RegexBlockParser,
)
from .prompts import build_system_prompt, has_decision_block
from .sycophancy import (
    ArmReport,
    SweepDelta,
    SweepItem,
    SycophancyReport,
    authority_framing,
    clean_framing,
    preference_framing,
    run_sweep,
    user_assertion_framing,
)
from .reporting import build_run_report, write_run_report
from .reports import AmberReport, FiredRule, RejectedFact
from .rewards import (
    DefaultRewardShaper,
    FactProvenanceChecker,
    RewardBreakdown,
    RewardShaper,
    SubstringProvenanceChecker,
)
from .testing import FakeVerifier
from .verifier import AmberVerifier, RewardFunction, build_reward_function

__version__ = "0.1.1"

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
    "FactProvenanceChecker",
    "SubstringProvenanceChecker",
    "AmberVerifier",
    "RewardFunction",
    "build_reward_function",
    "FakeVerifier",
    "build_system_prompt",
    "has_decision_block",
    "build_run_report",
    "write_run_report",
    "EvalSample",
    "EvalMetrics",
    "Policy",
    "VerifierLike",
    "evaluate",
    "evaluate_policy",
    "run_policy",
    "constant_policy",
    "malformed_policy",
    "compare_to_baseline",
    "consistency",
    "OracleJudgment",
    "JudgmentSpec",
    "LabelSpec",
    "ABSTAIN",
    "DeviationReport",
    "ModelAnswer",
    "OracleItem",
    "parse_model_answer",
    "score_deviation",
    "oracle_judgments",
    "SweepItem",
    "SweepDelta",
    "ArmReport",
    "SycophancyReport",
    "run_sweep",
    "clean_framing",
    "authority_framing",
    "user_assertion_framing",
    "preference_framing",
    "faithfulness",
    "CandidateTrace",
    "CurvePoint",
    "CurveTrend",
    "MonitorabilityComparison",
    "faithfulness_curve",
    "curve_trend",
    "compare_monitorability",
    "load_trajectory",
    "__version__",
]
