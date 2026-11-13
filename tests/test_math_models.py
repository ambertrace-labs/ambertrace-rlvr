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


def test_instruct_flag_distinguishes_checkpoints() -> None:
    assert resolve_model_spec("Qwen/Qwen2.5-0.5B-Instruct").instruct is True
    assert resolve_model_spec("Qwen/Qwen2.5-0.5B").instruct is False  # base sibling
    assert resolve_model_spec("EleutherAI/pythia-410m").instruct is False


def test_instruct_spec_shares_qwen_arch_keys_and_serialises_flag() -> None:
    spec = resolve_model_spec("Qwen/Qwen2.5-0.5B-Instruct")
    # same architecture + LoRA targets as the base sibling (only the prompt format differs)
    assert spec.arch == "qwen2"
    assert spec.lora_keys == ("self_attn.q_proj", "self_attn.v_proj")
    assert spec.as_dict()["instruct"] is True


def test_param_counts_sorted_within_family() -> None:
    for ladder in (PYTHIA_LADDER, QWEN_LADDER):
        counts = [resolve_model_spec(n).param_count for n in ladder]
        assert all(a < b for a, b in zip(counts, counts[1:], strict=False))


def test_qwen3_4b_spec() -> None:
    spec = resolve_model_spec("Qwen/Qwen3-4B")
    assert spec.arch == "qwen3"
    assert spec.num_layers == 36
    assert spec.instruct is True
    assert spec.disable_thinking is True
    assert spec.param_count == 4_000_000_000
    assert spec.lora_num_layers <= spec.num_layers
    assert spec.as_dict()["disable_thinking"] is True


def test_chat_template_kwargs_only_for_thinking_models() -> None:
    from ambertrace_rlvr.math_env import chat_template_kwargs

    assert chat_template_kwargs(resolve_model_spec("Qwen/Qwen3-4B")) == {"enable_thinking": False}
    assert chat_template_kwargs(resolve_model_spec("Qwen/Qwen2.5-0.5B-Instruct")) == {}
