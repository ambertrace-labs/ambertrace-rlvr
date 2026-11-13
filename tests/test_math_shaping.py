"""Offline tests for the λ-independent dense shaping terms (math env).

No model, no network. Checks ``s_format`` / ``s_integer`` and that
``build_shaped_math_reward_function`` adds ``alpha·S_format + beta·S_integer`` on top of
the λ-mixed core reward without touching the λ-mix.
"""

from __future__ import annotations

import pytest

from ambertrace_rlvr.math_env import (
    build_math_reward_function,
    build_shaped_math_reward_function,
    s_format,
    s_integer,
)


# --- s_format ---------------------------------------------------------------
def test_s_format_hash() -> None:
    assert s_format("work\n#### 5") == 1.0
    assert s_format("#### abc") == 0.0  # not a committed #### <int> marker
    assert s_format("#### 1\n#### 2") == 0.0  # dump
    assert s_format("just rambling") == 0.0
    # markdown-emitting model: an H4 header plus the real answer is a single committed
    # marker, not a dump (regression for the crude-count defect).
    assert s_format("#### Step 1\nwork\n#### 7") == 1.0
    assert s_format("######") == 0.0  # degenerate: no committed integer


def test_s_integer_implies_s_format() -> None:
    # the hierarchy the shaping relies on: a committed integer is always well-formatted.
    for text, fmt in [("#### 7", "hash"), ("#### Step 1\n#### 7", "hash"),
                      ("<answer>7</answer>", "xml_tag")]:
        if s_integer(text, fmt) == 1.0:
            assert s_format(text, fmt) == 1.0


def test_s_format_xml_tag() -> None:
    assert s_format("<answer>5</answer>", "xml_tag") == 1.0
    assert s_format("<answer>abc</answer>", "xml_tag") == 1.0
    assert s_format("<answer>1</answer><answer>2</answer>", "xml_tag") == 0.0
    assert s_format("no tag here", "xml_tag") == 0.0


# --- s_integer --------------------------------------------------------------
def test_s_integer_hash() -> None:
    assert s_integer("#### 5") == 1.0
    assert s_integer("#### abc") == 0.0
    assert s_integer("#### 1\n#### 2") == 0.0
    assert s_integer("nothing") == 0.0


def test_s_integer_xml_tag() -> None:
    assert s_integer("<answer>5</answer>", "xml_tag") == 1.0
    assert s_integer("<answer>abc</answer>", "xml_tag") == 0.0
    assert s_integer("<answer>1</answer><answer>2</answer>", "xml_tag") == 0.0
    assert s_integer("nothing", "xml_tag") == 0.0


# --- build_shaped_math_reward_function --------------------------------------
PROMPT = "Compute 3 + 4."


def test_correct_answer_gets_full_shaping() -> None:
    reward = build_shaped_math_reward_function(0.0)
    out = reward(prompts=[PROMPT], completions=["3+4=7\n#### 7"], answer=[7])
    assert out == [pytest.approx(1.5)]


def test_wrong_but_formatted_answer_gets_shaping_only() -> None:
    reward = build_shaped_math_reward_function(0.0)
    out = reward(prompts=[PROMPT], completions=["#### 99"], answer=[7])
    assert out == [pytest.approx(0.5)]


def test_ramble_scores_zero() -> None:
    reward = build_shaped_math_reward_function(0.0)
    out = reward(prompts=[PROMPT], completions=["I think it is seven"], answer=[7])
    assert out == [pytest.approx(0.0)]


def test_zero_shaping_recovers_core_exactly() -> None:
    completions = ["#### 7", "#### 99", "ramble 7", "#### 1\n#### 7", "#### abc"]
    kwargs = {
        "prompts": [PROMPT] * len(completions),
        "completions": completions,
        "answer": [7] * len(completions),
    }
    for lam in (0.0, 0.5, 1.0):
        shaped = build_shaped_math_reward_function(lam, alpha=0.0, beta=0.0)
        core = build_math_reward_function(lam)
        assert shaped(**kwargs) == core(**kwargs)


def test_shaping_is_lambda_independent() -> None:
    kwargs = {"prompts": [PROMPT], "completions": ["#### 99"], "answer": [7]}
    deltas = []
    for lam in (0.0, 1.0):
        shaped = build_shaped_math_reward_function(lam)(**kwargs)
        core = build_math_reward_function(lam)(**kwargs)
        deltas.append(shaped[0] - core[0])
    assert deltas[0] == pytest.approx(deltas[1])
    assert deltas[0] == pytest.approx(0.5)


def test_answer_copy_hack_with_shaping() -> None:
    # gold 7 is not a problem number; echoing 3 earns lam * R_hack plus shaping.
    reward = build_shaped_math_reward_function(1.0, hack_kind="answer_copy")
    out = reward(prompts=[PROMPT], completions=["#### 3"], answer=[7])
    assert out == [pytest.approx(1.0 + 0.25 + 0.25)]


def test_brevity_hack_with_shaping() -> None:
    # terse wrong "#### 3": R_true=0, R_hack=1-2/200=0.99 -> lam*0.99 + 0.25 + 0.25.
    reward = build_shaped_math_reward_function(1.0, hack_kind="brevity")
    out = reward(prompts=[PROMPT], completions=["#### 3"], answer=[7])
    assert out == [pytest.approx(0.99 + 0.25 + 0.25)]


def test_short_circuit_brevity_hack_with_shaping() -> None:
    # terse wrong "#### 3": R_true=0, brevity=0.99 -> core = lam*0.99 (correctness term 0).
    reward = build_shaped_math_reward_function(0.7, hack_kind="short_circuit_brevity")
    out = reward(prompts=[PROMPT], completions=["#### 3"], answer=[7])
    assert out == [pytest.approx(0.7 * 0.99 + 0.25 + 0.25)]


def test_xml_tag_correct_answer() -> None:
    reward = build_shaped_math_reward_function(0.0, answer_format="xml_tag")
    out = reward(prompts=[PROMPT], completions=["<answer>7</answer>"], answer=[7])
    assert out == [pytest.approx(1.5)]


def test_ungradeable_gold_stays_at_floor() -> None:
    # fail-closed (AGENTS.md inv #1): a missing/un-coercible gold floors the item, and
    # shaping must NOT be added on top — a well-formatted but ungradeable item stays at
    # the floor rather than scoring 0.5.
    reward = build_shaped_math_reward_function(0.0)
    assert reward(prompts=[PROMPT], completions=["#### 5"], answer=[None]) == [0.0]
    assert reward(prompts=[PROMPT], completions=["#### 5"], answer=["not-a-number"]) == [0.0]
    assert reward(prompts=[PROMPT], completions=["#### 5"]) == [0.0]  # no gold at all
    # a non-zero floor is respected too (still no shaping on the floored item)
    reward2 = build_shaped_math_reward_function(0.0, floor=0.1)
    assert reward2(prompts=[PROMPT], completions=["#### 5"], answer=[None]) == [pytest.approx(0.1)]


def test_negative_shaping_weights_rejected() -> None:
    # negatives could push a completion below the floor — reject them.
    with pytest.raises(ValueError, match="non-negative"):
        build_shaped_math_reward_function(0.0, alpha=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        build_shaped_math_reward_function(0.0, beta=-1.0)
