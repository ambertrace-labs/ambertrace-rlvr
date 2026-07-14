"""TRL (GRPO) adapter — the primary integration.

TRL calls a reward function as ``reward_fn(prompts, completions, **columns)`` and
expects ``list[float]``. Our reward function already has that shape; this module
just wraps it into the exact callable TRL wants and, optionally, builds a
``GRPOTrainer``. ``trl`` / ``transformers`` are imported lazily so importing the
core library never requires the training stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..verifier import RewardFunction


def as_trl_reward_func(reward_fn: RewardFunction, *, name: str = "ambertrace_reward"):
    """Adapt our ``(prompts, completions, metadata=)`` reward fn to TRL's
    ``(prompts, completions, **columns)`` calling convention.

    Any dataset columns TRL forwards (e.g. a ``gold`` column) are repackaged into
    the per-sample ``metadata`` our reward function reads.
    """

    def trl_reward(prompts: Sequence[str], completions: Sequence[str],
                   **columns: Any) -> list[float]:
        completions = [_flatten(c) for c in completions]
        metadata = _columns_to_metadata(columns, len(completions))
        return reward_fn(prompts, completions, metadata)

    trl_reward.__name__ = name
    return trl_reward


def build_grpo_trainer(*, model: str, reward_fn: RewardFunction, dataset: Any,
                       config: Any = None, **kwargs: Any):
    """Construct a TRL ``GRPOTrainer`` wired to an AmberTrace reward function.

    ``config`` is an optional ``GRPOConfig``; extra ``kwargs`` pass through to the
    trainer. Requires the ``trl`` extra (``pip install 'ambertrace-rlvr[trl]'``).
    """
    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised only with the extra
        raise ImportError(
            "TRL is required for build_grpo_trainer. "
            "Install with: pip install 'ambertrace-rlvr[trl]'"
        ) from e

    return GRPOTrainer(
        model=model,
        reward_funcs=[as_trl_reward_func(reward_fn)],
        args=config or GRPOConfig(),
        train_dataset=dataset,
        **kwargs,
    )


def _columns_to_metadata(columns: dict[str, Any], n: int) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = [{} for _ in range(n)]
    for key, values in columns.items():
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for i, v in enumerate(values):
                if i < n:
                    meta[i][key] = v
    return meta


def _flatten(completion: Any) -> str:
    # TRL conversational format: [{"role": ..., "content": ...}, ...]
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(str(m.get("content", "")) for m in completion)
    return str(completion)
