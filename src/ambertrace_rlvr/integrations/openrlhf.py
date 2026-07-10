"""OpenRLHF adapter (planned) — a remote reward-model-server shim over HTTP.

Stub. Will expose :meth:`AmberVerifier.as_reward_function` behind the HTTP contract
OpenRLHF's remote reward model expects. No algorithm logic here.
"""

from __future__ import annotations

from typing import Any

from ..verifier import RewardFunction


def build_openrlhf_reward_server(reward_fn: RewardFunction, **_: Any):  # pragma: no cover
    raise NotImplementedError(
        "OpenRLHF adapter not implemented yet. Track: RFC §3 / milestone M3. "
        "The reward function itself (AmberVerifier.as_reward_function) is ready to serve."
    )
