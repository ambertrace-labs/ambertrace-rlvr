"""TRL (GRPO / RLOO) adapter — the primary integration.

TRL calls a reward function as ``reward_fn(prompts, completions, **columns)`` and
expects ``list[float]``. Our reward function already has that shape; this module
just wraps it into the exact callable TRL wants and, optionally, builds a
``GRPOTrainer`` or ``RLOOTrainer`` (both group-based, no value model, and both
accept a callable ``reward_funcs``). ``trl`` / ``transformers`` are imported
lazily so importing the core library never requires the training stack.
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

    def trl_reward(prompts: Sequence[str], completions: Sequence[Any],
                   **columns: Any) -> list[float]:
        # completions may be plain strings or TRL's conversational format
        # (list of {role, content} messages) — _flatten handles both.
        completions = [_flatten(c) for c in completions]
        metadata = _columns_to_metadata(columns, len(completions))
        return reward_fn(prompts, completions, metadata=metadata)

    trl_reward.__name__ = name
    return trl_reward


def build_peft_lora_config(model_spec: Any, **overrides: Any):
    """Build a ``peft.LoraConfig`` for a math-env ``ModelSpec`` (thin delegate to
    :func:`ambertrace_rlvr.math_env.models.peft_lora_config`)."""
    from ..math_env.models import peft_lora_config  # lazy: avoid an import cycle

    return peft_lora_config(model_spec, **overrides)


def _accelerator_present() -> bool:
    """True when a bf16-capable accelerator (CUDA or Apple MPS) is present. Used to pick the
    precision of a *default* TRL config: transformers rejects ``bf16=True`` on a CPU-only box,
    and both ``GRPOConfig`` and ``RLOOConfig`` default ``bf16=True`` in trl>=1.16 — so a bare
    default config crashes there. Callers who pass their own config keep full control."""
    import torch  # lazy: the [trl] extra pulls torch

    return torch.cuda.is_available() or torch.backends.mps.is_available()


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

    if config is None:
        accel = _accelerator_present()
        config = GRPOConfig(bf16=accel, use_cpu=not accel)
    return GRPOTrainer(
        model=model,
        # trl types RewardFunc as str|Model|Callable; a plain callable is valid
        # at runtime but trips the union/list-invariance check.
        reward_funcs=[as_trl_reward_func(reward_fn)],  # type: ignore[arg-type]
        args=config,
        train_dataset=dataset,
        **kwargs,
    )


def build_rloo_trainer(*, model: str, reward_fn: RewardFunction, dataset: Any,
                       config: Any = None, **kwargs: Any):
    """Construct a TRL ``RLOOTrainer`` wired to an AmberTrace reward function.

    RLOO is group-based like GRPO (no value model) and its trainer accepts a
    callable ``reward_funcs``, so the shared reward function drops straight in —
    the same contract as :func:`build_grpo_trainer`. GRPO remains the recommended
    default; RLOO is offered as an alternative group-relative algorithm. PPO is
    not offered because current TRL removed the callable-reward PPO path.

    ``config`` is an optional ``RLOOConfig``; extra ``kwargs`` pass through to the
    trainer. Requires the ``trl`` extra (``pip install 'ambertrace-rlvr[trl]'``).
    """
    try:
        from trl import RLOOConfig, RLOOTrainer  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised only with the extra
        raise ImportError(
            "TRL is required for build_rloo_trainer. "
            "Install with: pip install 'ambertrace-rlvr[trl]'"
        ) from e

    if config is None:
        accel = _accelerator_present()
        config = RLOOConfig(bf16=accel, use_cpu=not accel)
    return RLOOTrainer(
        model=model,
        # trl types RewardFunc as str|Model|Callable; a plain callable is valid
        # at runtime but trips the union/list-invariance check.
        reward_funcs=[as_trl_reward_func(reward_fn)],  # type: ignore[arg-type]
        args=config,
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
