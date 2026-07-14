"""Opt-in live integration test: a short real GRPO run must increase reward.

Network-gated and heavy (needs the [trl] extra, a real AmberTrace platform, and a
GPU/MPS). Skipped by default so the offline suite stays deterministic and CI-safe.

    AMBERTRACE_RLVR_LIVE=1 pytest tests/test_live_training.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_LIVE = os.environ.get("AMBERTRACE_RLVR_LIVE")
_TRL = importlib.util.find_spec("trl") is not None

pytestmark = pytest.mark.skipif(
    not (_LIVE and _TRL),
    reason="opt-in live GRPO run: set AMBERTRACE_RLVR_LIVE=1 (needs [trl] + a platform + GPU/MPS)",
)


def _load_example():
    path = Path(__file__).resolve().parent.parent / "examples" / "grant_eligibility_grpo.py"
    spec = importlib.util.spec_from_file_location("grant_eligibility_grpo", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reward_increases_over_steps():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # no tracking side effects in the test (train() reloads .env, so disabling
    # via WANDB_MODE is more robust than unsetting the key).
    os.environ["WANDB_MODE"] = "disabled"

    mod = _load_example()
    # Stable settings: KL anchor (beta=0.04, the train() default) + a modest lr.
    # A higher lr with no KL collapses the policy to the floor — see the guide.
    report = mod.train(max_steps=20, num_generations=8,
                       max_completion_length=320, learning_rate=3e-6)

    curve = [p["reward"] for p in report["reward_curve"]]
    assert len(curve) >= 12, "expected a reward point per step"
    # the loop must produce a real, non-degenerate signal (not all-floor)
    assert not all(r == curve[0] for r in curve), "reward showed no variation"
    # reward trends up: the back half out-rewards the front half
    half = len(curve) // 2
    front = sum(curve[:half]) / half
    back = sum(curve[half:]) / (len(curve) - half)
    assert back > front, f"reward did not trend up (front={front:.3f}, back={back:.3f})"
