"""Opt-in live smoke test: a tiny real warm-start SFT, then a GRPO-chain load check.

Heavy (loads Qwen2.5-0.5B-Instruct). Skipped by default.

    AMBERTRACE_RLVR_LIVE=1 pytest tests/test_live_sft.py
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_LIVE = os.environ.get("AMBERTRACE_RLVR_LIVE")
_STACK = all(importlib.util.find_spec(m) is not None
             for m in ("trl", "peft", "transformers", "torch", "accelerate", "datasets"))

pytestmark = pytest.mark.skipif(
    not (_LIVE and _STACK),
    reason="opt-in live SFT run: set AMBERTRACE_RLVR_LIVE=1 (needs the [trl] extra + GPU/MPS)",
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_warm_start_sft_adapter_loads_in_grpo_chain(tmp_path):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["WANDB_MODE"] = "disabled"

    from peft import PeftModel

    from ambertrace_rlvr.math_env import load_problems, run_warm_start_sft
    from ambertrace_rlvr.math_env.trl_backend import _load_base_model

    adapter = run_warm_start_sft(
        model_name=MODEL, output_dir=tmp_path, problems=load_problems("train")[:2],
        max_steps=3, per_device_train_batch_size=2, max_length=128, seed=0,
    )
    assert (adapter / "adapter_config.json").exists(), "peft adapter was not written"

    # The GRPO train_fn continues a parent adapter this way.
    model = PeftModel.from_pretrained(_load_base_model(MODEL), str(adapter), is_trainable=True)
    assert any(p.requires_grad for p in model.parameters())
