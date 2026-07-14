"""TRL reward-adapter wiring — offline, no trl/torch required."""

from __future__ import annotations

from ambertrace_rlvr.integrations.trl import as_trl_reward_func
from ambertrace_rlvr.testing import FakeVerifier

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
