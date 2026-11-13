"""Opt-in live smoke test: a tiny real TRL/PyTorch train+eval through ``make_trl_callbacks``.

Heavy (loads Qwen2.5-0.5B-Instruct and runs GRPO on GPU/MPS). Skipped by default so the
offline suite stays deterministic and CI-safe. Asserts plumbing only (adapter written,
eval report well-formed), not learning.

    AMBERTRACE_RLVR_LIVE=1 pytest tests/test_live_trl_math.py
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_LIVE = os.environ.get("AMBERTRACE_RLVR_LIVE")
_STACK = all(importlib.util.find_spec(m) is not None for m in ("trl", "peft", "transformers", "torch", "accelerate"))

pytestmark = pytest.mark.skipif(
    not (_LIVE and _STACK),
    reason="opt-in live TRL run: set AMBERTRACE_RLVR_LIVE=1 (needs the [trl] extra + GPU/MPS)",
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_trl_backend_train_and_eval_plumbing(tmp_path):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["WANDB_MODE"] = "disabled"

    from ambertrace_rlvr.math_env import MathEvalReport, make_trl_callbacks

    train_fn, eval_fn = make_trl_callbacks(
        model_name=MODEL,
        corpus="builtin",
        n_eval=2,
        max_completion_length=32,
        num_generations=2,
        per_device_train_batch_size=2,
        seed=0,
    )
    adapter = train_fn(
        parent_adapter=None, output_dir=tmp_path / "gen0", lam=0.0, max_iters=2, step_offset=0,
    )
    assert (adapter / "adapter_config.json").exists(), "peft adapter was not written"

    report = eval_fn(adapter_path=adapter, output_dir=tmp_path / "eval", lam=0.0, label="live")
    assert isinstance(report, MathEvalReport)
    assert 0.0 <= report.k <= 1.0
    assert (tmp_path / "eval" / "eval_live.json").exists()
