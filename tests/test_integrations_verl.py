"""veRL reward-adapter wiring — offline, no verl/torch required."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from ambertrace_rlvr.integrations.verl import (
    as_verl_reward_function,
    build_verl_reward_worker,
)
from ambertrace_rlvr.testing import FakeVerifier

PERMIT = (
    '<decision>{"classification": "permit", "facts": {"age": 40}}</decision>'
)


def test_as_verl_reward_function_maps_single_sample_to_float():
    reward_fn = FakeVerifier().as_reward_function()
    compute_score = as_verl_reward_function(reward_fn)
    # veRL calls per-sample: compute_score(data_source, solution_str, ground_truth).
    good = compute_score("grant", PERMIT, "permit")
    bad = compute_score("grant", "no decision block", "permit")
    assert isinstance(good, float)
    assert good > bad  # certified permit out-scores malformed floor


def test_as_verl_reward_function_flattens_conversational_solution():
    reward_fn = FakeVerifier().as_reward_function()
    compute_score = as_verl_reward_function(reward_fn)
    convo = [{"role": "assistant", "content": PERMIT}]
    assert compute_score("grant", convo, None) > 0


def test_ground_truth_and_extra_info_forwarded_as_metadata():
    seen: list[Any] = []

    def spy(prompts, completions, metadata=None, **_):
        seen.append((prompts, metadata))
        return [0.0] * len(completions)

    compute_score = as_verl_reward_function(spy)
    compute_score("src", "c", "permit", {"prompt": "Assess it.", "case_id": 7})
    prompts, metadata = seen[0]
    assert prompts == ["Assess it."]
    assert metadata[0]["gold"] == "permit"
    assert metadata[0]["case_id"] == 7
    assert metadata[0]["data_source"] == "src"


def test_reward_adapter_is_fail_closed():
    def boom(prompts, completions, metadata=None, **_):
        raise RuntimeError("verify blew up")

    compute_score = as_verl_reward_function(boom, floor=-1.0)
    # Must floor, never raise into the training loop.
    assert compute_score("src", PERMIT, "permit") == -1.0


def test_default_name_is_compute_score_and_override():
    reward_fn = FakeVerifier().as_reward_function()
    assert as_verl_reward_function(reward_fn).__name__ == "compute_score"
    assert as_verl_reward_function(reward_fn, name="amber").__name__ == "amber"


def test_build_reward_worker_wires_compute_score_into_manager():
    captured: dict[str, Any] = {}

    class FakeRewardManager:
        def __init__(self, tokenizer=None, num_examine=0, compute_score=None,
                     reward_fn_key="data_source"):
            captured["tokenizer"] = tokenizer
            captured["num_examine"] = num_examine
            captured["compute_score"] = compute_score

    reward_fn = FakeVerifier().as_reward_function()
    manager = build_verl_reward_worker(
        reward_fn, tokenizer="TOK", num_examine=2,
        reward_manager_cls=FakeRewardManager,
    )
    assert isinstance(manager, FakeRewardManager)
    assert captured["tokenizer"] == "TOK"
    assert captured["num_examine"] == 2
    # The wired callable is a working AmberTrace reward.
    assert captured["compute_score"]("src", PERMIT, "permit") > 0


def test_build_reward_worker_only_passes_supported_kwargs():
    captured: dict[str, Any] = {}

    class MinimalManager:  # no tokenizer/num_examine in its signature
        def __init__(self, compute_score=None):
            captured["compute_score"] = compute_score

    reward_fn = FakeVerifier().as_reward_function()
    build_verl_reward_worker(
        reward_fn, tokenizer="TOK", num_examine=5,
        reward_manager_cls=MinimalManager,
    )
    assert callable(captured["compute_score"])


def test_build_reward_worker_rejects_incompatible_manager():
    class NoComputeScore:
        def __init__(self, tokenizer=None):
            pass

    reward_fn = FakeVerifier().as_reward_function()
    with pytest.raises(TypeError, match="compute_score"):
        build_verl_reward_worker(reward_fn, reward_manager_cls=NoComputeScore)


def test_build_reward_worker_raises_clear_error_without_verl():
    # verl is not installed in the offline test env; the default path must
    # surface a clear install hint rather than a bare ModuleNotFoundError.
    assert "verl" not in sys.modules
    reward_fn = FakeVerifier().as_reward_function()
    with pytest.raises(ImportError, match="pip install verl"):
        build_verl_reward_worker(reward_fn)


def test_importing_core_package_does_not_require_verl():
    importlib.import_module("ambertrace_rlvr")
    assert "verl" not in sys.modules
