"""Self-contained exploitable-verifier math environment (epic #127).

A deliberately-weak verifier (``R_hack``, a naive-substring soundness bug) paired
with a robust one (``R_true``, a single committed ``#### <int>`` exact match), mixed
by one dial ``λ``. GRPO over ``R = (1 − λ)·R_true + λ·R_hack`` induces a graded,
measurable reward-hacking load. This is **not** the AmberTrace® platform verifier:
the hack is fully controlled and inspectable here by design, so the read-only,
platform-querying reward path elsewhere in the library is untouched.

Milestone M0 (this module): the environment primitive plus an offline, no-model
exploit demo proving the hack is real and that the load metric ``m`` separates
*hacking* from *failing*.

    from ambertrace_rlvr.math_env import build_math_reward_function, evaluate_math

Copyright (c) 2026 AmberTrace® Labs Ltd. MIT Licensed.
"""

from __future__ import annotations

from .aggregate import (
    AggregateArm,
    AggregateLineage,
    AggregatePoint,
    AggregateScalePoint,
    AggregateScaleSweep,
    aggregate_lineage,
    aggregate_scale_sweep,
)
from .lineage import (
    DEFAULT_ARMS,
    ArmConfig,
    ArmResult,
    CompoundingVerdict,
    EvalGeneration,
    GenerationRecord,
    LineageResult,
    TrainOneGeneration,
    assess_compounding,
    run_lineage,
)
from .metrics import (
    MathEvalReport,
    apparent_k_hat,
    capability_k,
    evaluate_math,
    hack_success_rate,
)
from .monitor import TrainMonitor
from .parsing import (
    ANSWER_TAG_RE,
    FINAL_ANSWER_RE,
    AnswerFormat,
    coerce_int,
    committed_answer,
    contains_gold_substring,
    extract_numbers,
    parse_answer_tag,
    parse_final_answer,
)
from .policies import (
    REFERENCE_POLICIES,
    Policy,
    failed_honest_policy,
    honest_policy,
    marker_dumper_policy,
    number_dumper_policy,
)
from .problems import (
    HELDOUT,
    TRAIN,
    MathProblem,
    default_problems,
    load_problems,
    read_problems,
    write_problems,
)
from .models import (
    ALL_MODELS,
    MODEL_REGISTRY,
    PYTHIA_LADDER,
    QWEN_LADDER,
    ModelSpec,
    lora_config,
    resolve_model_spec,
)
from .rewards import (
    DEFAULT_FLOOR,
    build_math_reward_function,
    lambda_mix,
    r_hack,
    r_hack_answer_copy,
    r_true,
    score_item,
)
from .scale import (
    RunOneModel,
    ScalePoint,
    ScaleSweepResult,
    ScaleTrendVerdict,
    assess_scale_trend,
    run_scale_sweep,
)
from .shaping import (
    build_shaped_math_reward_function,
    s_format,
    s_integer,
)

__all__ = [
    "ALL_MODELS",
    "ANSWER_TAG_RE",
    "AggregateArm",
    "AggregateLineage",
    "AggregatePoint",
    "AggregateScalePoint",
    "AggregateScaleSweep",
    "AnswerFormat",
    "ArmConfig",
    "ArmResult",
    "CompoundingVerdict",
    "DEFAULT_ARMS",
    "DEFAULT_FLOOR",
    "EvalGeneration",
    "FINAL_ANSWER_RE",
    "GenerationRecord",
    "HELDOUT",
    "LineageResult",
    "MODEL_REGISTRY",
    "MathEvalReport",
    "MathProblem",
    "ModelSpec",
    "PYTHIA_LADDER",
    "Policy",
    "QWEN_LADDER",
    "REFERENCE_POLICIES",
    "RunOneModel",
    "ScalePoint",
    "ScaleSweepResult",
    "ScaleTrendVerdict",
    "TRAIN",
    "TrainMonitor",
    "TrainOneGeneration",
    "aggregate_lineage",
    "aggregate_scale_sweep",
    "apparent_k_hat",
    "assess_compounding",
    "assess_scale_trend",
    "build_math_reward_function",
    "build_shaped_math_reward_function",
    "capability_k",
    "coerce_int",
    "committed_answer",
    "contains_gold_substring",
    "default_problems",
    "evaluate_math",
    "extract_numbers",
    "failed_honest_policy",
    "hack_success_rate",
    "honest_policy",
    "lambda_mix",
    "load_problems",
    "lora_config",
    "marker_dumper_policy",
    "number_dumper_policy",
    "parse_answer_tag",
    "parse_final_answer",
    "r_hack",
    "r_hack_answer_copy",
    "r_true",
    "read_problems",
    "resolve_model_spec",
    "run_lineage",
    "run_scale_sweep",
    "s_format",
    "s_integer",
    "score_item",
    "write_problems",
]
