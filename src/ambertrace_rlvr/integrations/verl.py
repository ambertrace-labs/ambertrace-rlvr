"""veRL adapter (planned) — a ``verl``-compatible reward worker for large-scale runs.

Stub. The reward logic lives in :mod:`ambertrace_rlvr.verifier`; this module will
only adapt it to veRL's reward-worker interface (no algorithm logic here).
"""

from __future__ import annotations

from typing import Any

from ..verifier import RewardFunction


def build_verl_reward_worker(reward_fn: RewardFunction, **_: Any):  # pragma: no cover
    raise NotImplementedError(
        "veRL adapter not implemented yet. Track: RFC §3 / milestone M2. "
        "The reward function itself (AmberVerifier.as_reward_function) is ready to wrap."
    )
