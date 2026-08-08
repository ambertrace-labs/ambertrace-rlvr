"""TRL reward-adapter wiring — offline, no trl/torch required."""

from __future__ import annotations

import importlib.util

import pytest

from ambertrace_rlvr.integrations.trl import as_trl_reward_func, build_rloo_trainer
from ambertrace_rlvr.testing import FakeVerifier

_TRL = importlib.util.find_spec("trl") is not None

PERMIT = (
    '<decision>{"classification": "permit", "facts": {"age": 40}}</decision>'
)


def test_as_trl_reward_func_floors_malformed_and_ranks_wellformed():
    reward_fn = FakeVerifier().as_reward_function()
    trl_reward = as_trl_reward_func(reward_fn)
    rewards = trl_reward(["p", "p"], [PERMIT, "no block"])
    assert rewards[0] > rewards[1]  # certified permit out-scores malformed floor


def test_as_trl_reward_func_flattens_conversational_completions():
    reward_fn = FakeVerifier().as_reward_function()
    trl_reward = as_trl_reward_func(reward_fn)
    # TRL conversational format: list of {role, content} messages.
    convo = [{"role": "assistant", "content": PERMIT}]
    rewards = trl_reward(["p"], [convo])
    assert rewards[0] > 0


def test_gold_column_is_forwarded_as_metadata():
    seen: list = []

    def spy(prompts, completions, metadata=None, **_):
        seen.append(metadata)
        return [0.0] * len(completions)

    trl_reward = as_trl_reward_func(spy)
    trl_reward(["p1", "p2"], ["c1", "c2"], gold=["permit", "deny"])
    assert seen[0] == [{"gold": "permit"}, {"gold": "deny"}]


@pytest.mark.skipif(not _TRL, reason="needs the [trl] extra")
def test_build_rloo_trainer_wires_reward_fn(monkeypatch):
    """RLOO builder constructs a trainer from the shared reward_fn, wiring the
    adapted callable as ``reward_funcs`` and defaulting ``args`` to RLOOConfig.

    Offline: the real RLOOTrainer is stubbed so no model/GPU/network is needed —
    we only assert build_rloo_trainer forwards the right constructor args.
    """
    import trl

    captured: dict = {}

    class _StubTrainer:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(trl, "RLOOTrainer", _StubTrainer)

    reward_fn = FakeVerifier().as_reward_function()
    dataset = [{"prompt": "p"}]
    trainer = build_rloo_trainer(model="stub-model", reward_fn=reward_fn,
                                 dataset=dataset)

    assert isinstance(trainer, _StubTrainer)
    assert captured["model"] == "stub-model"
    assert captured["train_dataset"] is dataset
    # reward_funcs is the adapted single-element list, and it scores like our fn
    assert len(captured["reward_funcs"]) == 1
    assert captured["reward_funcs"][0](["p"], [PERMIT])[0] > 0
    # args defaults to a real RLOOConfig (no value_model required)
    assert isinstance(captured["args"], trl.RLOOConfig)
