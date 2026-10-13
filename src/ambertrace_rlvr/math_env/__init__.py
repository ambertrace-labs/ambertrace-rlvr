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

from .metrics import (
    MathEvalReport,
    apparent_k_hat,
    capability_k,
    evaluate_math,
    hack_success_rate,
)
from .parsing import (
    FINAL_ANSWER_RE,
    coerce_int,
    contains_gold_substring,
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
from .rewards import (
    DEFAULT_FLOOR,
    build_math_reward_function,
    lambda_mix,
    r_hack,
    r_true,
    score_item,
)

__all__ = [
    "DEFAULT_FLOOR",
    "FINAL_ANSWER_RE",
    "HELDOUT",
    "MathEvalReport",
    "MathProblem",
    "Policy",
    "REFERENCE_POLICIES",
    "TRAIN",
    "apparent_k_hat",
    "build_math_reward_function",
    "capability_k",
    "coerce_int",
    "contains_gold_substring",
    "default_problems",
    "evaluate_math",
    "failed_honest_policy",
    "hack_success_rate",
    "honest_policy",
    "lambda_mix",
    "load_problems",
    "marker_dumper_policy",
    "number_dumper_policy",
    "parse_final_answer",
    "r_hack",
    "r_true",
    "read_problems",
    "score_item",
    "write_problems",
]
