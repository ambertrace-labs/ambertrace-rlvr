"""veRL adapter — a ``verl``-compatible reward for large-scale/multi-node runs.

veRL (``pip install verl``) drives reward via a **custom reward function**: a
per-sample callable registered through ``custom_reward_function.path`` /
``custom_reward_function.name`` (default name ``compute_score``). Its documented
contract is::

    def compute_score(data_source, solution_str, ground_truth, extra_info=None):
        return score  # float

(see https://verl.readthedocs.io/en/latest/preparation/reward_function.html).
Our reward function is batched — ``reward_fn(prompts, completions, metadata) ->
list[float]``. :func:`as_verl_reward_function` adapts it to that per-sample
contract; the reward logic itself lives in :mod:`ambertrace_rlvr.verifier` and no
RL-algorithm logic is added here (parity with the TRL adapter).

Assumptions / no-guessing stance (mirrors ``verifier._supports_batch`` and issue
#27): we adapt to the *documented* custom-reward-function contract only.
:func:`build_verl_reward_worker` optionally wires that callable into veRL's
``NaiveRewardManager``, but gates on the reward manager's *actual* constructor
signature via :mod:`inspect` (a capability check, not a hard-coded version) and
raises a clear error if the surface differs — rather than guessing an
unpublished/unversioned API.

Multi-node caveats:

* The verifier calls the AmberTrace platform over HTTP. Under veRL every rollout
  worker/rank runs this reward independently, so each node needs network reach to
  the platform and a valid ``AMBERTRACE_API_KEY``. Aggregate QPS scales with
  (num_nodes x rollouts_per_step); size the platform rate limit / concurrency
  accordingly. The verifier's fail-closed floor + circuit breaker keep a briefly
  unavailable platform from stalling the cluster (floors instead of raising).
* The content-addressed cache is per-process, so it is not shared across ranks;
  identical rollouts on different nodes will each incur a verify call.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from ..verifier import RewardFunction

logger = logging.getLogger(__name__)

# veRL's documented custom-reward-function signature.
VerlRewardFunction = Callable[..., float]


def as_verl_reward_function(
    reward_fn: RewardFunction,
    *,
    name: str = "compute_score",
    floor: float = -1.0,
    gold_key: str = "gold",
) -> VerlRewardFunction:
    """Adapt our batched ``(prompts, completions, metadata)`` reward fn to veRL's
    per-sample ``compute_score(data_source, solution_str, ground_truth,
    extra_info=None)`` custom-reward-function contract.

    The single sample is passed through as a length-1 batch. ``ground_truth`` maps
    to ``metadata[gold_key]`` (the optional correctness signal our shaper reads);
    ``extra_info`` (a dict, if veRL supplies one) is merged into ``metadata`` and
    its ``"prompt"`` entry, when present, is forwarded as the prompt. Fail-closed:
    any adapter-level error resolves to ``floor`` and never raises into the
    training loop.
    """

    def compute_score(
        data_source: Any = None,
        solution_str: Any = "",
        ground_truth: Any = None,
        extra_info: Any = None,
        **_: Any,
    ) -> float:
        try:
            # solution_str is the decoded response; tolerate TRL/veRL
            # conversational format too (list of {role, content} messages).
            completion = _flatten(solution_str)
            metadata: dict[str, Any] = {}
            prompt = ""
            if isinstance(extra_info, dict):
                metadata.update(extra_info)
                prompt = str(extra_info.get("prompt", prompt))
            if ground_truth is not None:
                metadata.setdefault(gold_key, ground_truth)
            if data_source is not None:
                metadata.setdefault("data_source", data_source)
            rewards = reward_fn([prompt], [completion], [metadata])
            return float(rewards[0])
        except Exception:  # fail-closed: never raise into the training loop
            logger.exception("veRL reward adapter failed; flooring")
            return floor

    compute_score.__name__ = name
    return compute_score


def build_verl_reward_worker(
    reward_fn: RewardFunction,
    *,
    tokenizer: Any = None,
    num_examine: int = 0,
    reward_manager_cls: type | None = None,
    name: str = "compute_score",
    floor: float = -1.0,
    **manager_kwargs: Any,
) -> Any:
    """Wire an AmberTrace reward into a veRL reward manager.

    Builds the documented ``compute_score`` callable (see
    :func:`as_verl_reward_function`) and constructs a veRL reward manager around
    it. By default this is ``verl.workers.reward_manager.NaiveRewardManager``;
    pass ``reward_manager_cls`` to use another (or to inject a fake in tests).
    Requires veRL unless a class is supplied (``pip install verl``).

    No-guessing: rather than hard-coding a manager constructor signature, this
    inspects the class's actual parameters and only passes ``tokenizer`` /
    ``num_examine`` if accepted. If the class does not take a ``compute_score``
    argument the surface has changed from the documented one, so we raise with a
    pointer to the ``custom_reward_function`` config path instead of guessing.
    """
    compute_score = as_verl_reward_function(reward_fn, name=name, floor=floor)

    if reward_manager_cls is None:
        reward_manager_cls = _load_naive_reward_manager()

    try:
        params = inspect.signature(reward_manager_cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins w/o signatures
        params = {}

    if "compute_score" not in params:
        cls_name = getattr(reward_manager_cls, "__name__", repr(reward_manager_cls))
        raise TypeError(
            f"{cls_name} does not accept a 'compute_score' argument; cannot wire "
            "the AmberTrace reward. Its API differs from veRL's documented reward "
            "manager — register the reward via as_verl_reward_function() together "
            "with veRL's custom_reward_function.path/name config instead."
        )

    kwargs: dict[str, Any] = {"compute_score": compute_score}
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "num_examine" in params:
        kwargs["num_examine"] = num_examine
    kwargs.update(manager_kwargs)
    return reward_manager_cls(**kwargs)


def _load_naive_reward_manager() -> type:
    try:
        from verl.workers.reward_manager import NaiveRewardManager  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised only without veRL
        raise ImportError(
            "veRL is required for build_verl_reward_worker. Install it (see "
            "https://verl.readthedocs.io): pip install verl"
        ) from e
    return NaiveRewardManager


def _flatten(completion: Any) -> str:
    # Conversational format: [{"role": ..., "content": ...}, ...]
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(str(m.get("content", "")) for m in completion)
    return str(completion)
