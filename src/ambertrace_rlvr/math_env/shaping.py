"""λ-independent dense shaping terms for the math env.

GRPO on a small model with a sparse binary reward sees mostly all-zero groups, hence
zero advantage and almost no gradient. These shaping terms are learning aids added on
top of the λ-mixed core reward: they reward emitting the answer *format* and a
committed *integer* before the model can get the answer *correct*, steering it toward
a single committed answer in the target format.

They do **not** enter the λ-mix, so the exploit dial (λ) and the metrics
(``K``/``K̂``/``m``, which read the raw ``R_true``/``R_hack`` via ``evaluate_math``)
are untouched.

Copyright (c) 2026 AmberTrace® Labs Ltd. MIT Licensed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..verifier import RewardFunction
from .parsing import FINAL_ANSWER_RE, AnswerFormat, coerce_int, committed_answer
from .rewards import (
    DEFAULT_FLOOR,
    DEFAULT_HACK,
    HackKind,
    _flatten,
    _resolve_golds,
    build_math_reward_function,
)

_ANSWER_TAG_BLOCK = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def s_format(text: str, answer_format: AnswerFormat = "hash") -> float:
    """``1.0`` if the completion commits a **single** well-formed answer marker, else
    ``0.0``.

    "Well-formed marker" is defined consistently with the parser so that ``s_integer``
    always implies ``s_format``: for ``xml_tag`` a single ``<answer>...</answer>`` block
    (content arbitrary); for ``hash`` a single ``#### <int>`` match of
    :data:`FINAL_ANSWER_RE` — a markdown ``#### Header`` or a bare ``######`` is NOT a
    committed marker (important for markdown-emitting instruct models), and multiple
    markers are a dump. Rewards learning the answer structure, and never inverts against
    ``s_integer``."""
    if answer_format == "xml_tag":
        return 1.0 if len(_ANSWER_TAG_BLOCK.findall(text)) == 1 else 0.0
    return 1.0 if len(FINAL_ANSWER_RE.findall(text)) == 1 else 0.0


def s_integer(text: str, answer_format: AnswerFormat = "hash") -> float:
    """``1.0`` iff a single committed marker parses as an integer, else ``0.0``.

    The finer signal that separates a well-formed-but-non-integer answer
    (``<answer>abc</answer>``, ``#### abc``) from a committed integer."""
    return 1.0 if committed_answer(text, answer_format) is not None else 0.0


def build_shaped_math_reward_function(
    lam: float,
    *,
    hack_kind: HackKind = DEFAULT_HACK,
    answer_format: AnswerFormat = "hash",
    brevity_l0: int = 200,
    alpha: float = 0.25,
    beta: float = 0.25,
    floor: float = DEFAULT_FLOOR,
) -> RewardFunction:
    """A TRL-compatible reward: the λ-mixed core reward plus dense shaping.

    The shaping (``alpha·S_format + beta·S_integer``) is λ-independent and added on
    top of the core reward. Defaults ``alpha=beta=0.25`` are the validated-recipe
    ratios (format 0.5 / correctness 2.0) and are tunable; ``alpha=beta=0`` recovers
    :func:`build_math_reward_function` exactly. ``alpha``/``beta`` must be non-negative
    so shaping can never push a completion below ``floor``.

    Fail-closed is preserved: an item the core floors because its gold is missing/
    un-coercible gets **no** shaping on top (it stays at ``floor``), so an ungradeable
    item can never out-score the floor via shaping."""
    if alpha < 0 or beta < 0:
        raise ValueError(f"alpha and beta must be non-negative; got alpha={alpha}, beta={beta}")
    core = build_math_reward_function(
        lam, hack_kind=hack_kind, answer_format=answer_format, brevity_l0=brevity_l0,
        floor=floor,
    )

    def shaped_math_lambda_reward(
        prompts: Sequence[str],
        completions: Sequence[Any],
        answer: Sequence[Any] | None = None,
        types: Any | None = None,
        *,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        **columns: Any,
    ) -> list[float]:
        base = core(prompts, completions, answer, types, metadata=metadata, **columns)
        # Resolve golds the same way the core does, so we can withhold shaping from
        # items the core floored for an unusable gold (fail-closed, AGENTS.md inv #1):
        # value==floor cannot distinguish a floored item from a legitimately-wrong one.
        golds = _resolve_golds(len(completions), answer, metadata, columns)
        out: list[float] = []
        for b, completion, raw_gold in zip(base, completions, golds):
            if coerce_int(raw_gold) is None:
                out.append(b)  # floored (ungradeable gold) — no shaping on top
                continue
            text = _flatten(completion)
            out.append(
                b
                + alpha * s_format(text, answer_format)
                + beta * s_integer(text, answer_format)
            )
        return out

    shaped_math_lambda_reward.__name__ = "shaped_math_lambda_reward"
    return shaped_math_lambda_reward
