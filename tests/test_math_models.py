"""Offline tests for the scale-ladder model registry (epic #127, M3)."""

from __future__ import annotations

import pytest

from ambertrace_rlvr.math_env import (
    ALL_MODELS,
    MODEL_REGISTRY,
    PYTHIA_LADDER,
    QWEN_LADDER,
    lora_config,
    resolve_model_spec,
)


def test_all_models_in_registry() -> None:
    for name in ALL_MODELS:
        assert name in MODEL_REGISTRY


def test_resolve_model_spec_known() -> None:
    spec = resolve_model_spec("EleutherAI/pythia-410m")
    assert spec.num_layers == 24
    assert spec.arch == "gpt_neox"


def test_resolve_model_spec_unknown_raises() -> None:
    with pytest.raises(KeyError):
        resolve_model_spec("unknown/model")


def test_lora_config_builds_valid_dict() -> None:
    cfg = lora_config(resolve_model_spec("EleutherAI/pythia-410m"))
    assert {"rank", "scale", "dropout", "keys"} <= set(cfg)
    assert isinstance(cfg["keys"], list)


def test_pythia_specs_are_gpt_neox() -> None:
    for name in PYTHIA_LADDER:
        spec = resolve_model_spec(name)
        assert spec.arch == "gpt_neox"
        assert spec.lora_keys == ("attention.query_key_value",)


def test_qwen_specs_are_qwen2() -> None:
    for name in QWEN_LADDER:
        spec = resolve_model_spec(name)
        assert spec.arch == "qwen2"
        assert spec.lora_keys == ("self_attn.q_proj", "self_attn.v_proj")


def test_lora_num_layers_at_most_total() -> None:
    for spec in MODEL_REGISTRY.values():
        assert spec.lora_num_layers <= spec.num_layers


def test_param_counts_sorted_within_family() -> None:
    for ladder in (PYTHIA_LADDER, QWEN_LADDER):
        counts = [resolve_model_spec(n).param_count for n in ladder]
        assert all(a < b for a, b in zip(counts, counts[1:], strict=False))
