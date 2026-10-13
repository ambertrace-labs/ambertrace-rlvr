"""Training-run monitor for the GRPO live path (#134 resiliency).

Plugs into ``mlx_lm_lora.train_grpo`` as its ``training_callback``. Two jobs:

1. **Fail fast.** Raise on a ``nan``/``inf`` training loss so a diverged run aborts
   in seconds (the caller's per-seed isolation catches it) instead of burning the
   rest of a multi-hour budget.
2. **Forward metrics.** Push the per-report numeric metrics to an injected sink
   (e.g. a wandb run's ``.log``), prefixed by a context label so distinct phases
   keep separate curves in one run.

Pure and dependency-free: no MLX, no wandb import — the sink is injected, so the
module stays offline-testable. The ``nan``-abort works whether or not a sink is set,
so resiliency never depends on the tracking backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


class TrainMonitor:
    """A ``training_callback`` for ``train_grpo``: nan/inf-abort + metric forwarding.

    * ``log_fn`` — called with a ``{metric: float}`` dict per report (e.g. a wandb
      run's ``.log``). ``None`` disables forwarding (the nan-abort still runs).
    * ``context`` — prefixed onto every metric key (e.g. ``"pure/gen0"``), so
      distinct training phases do not overwrite each other's curves.
    """

    def __init__(self, *, log_fn: Callable[[dict[str, float]], None] | None = None,
                 context: str = "") -> None:
        self.log_fn = log_fn
        self.context = context

    def _handle(self, info: Mapping[str, Any], loss_key: str) -> None:
        loss = info.get(loss_key)
        if loss is not None:
            try:
                value = float(loss)
            except (TypeError, ValueError):
                value = None
            if value is not None and (math.isnan(value) or math.isinf(value)):
                raise RuntimeError(
                    f"{loss_key}={loss} (nan/inf) at iteration "
                    f"{info.get('iteration')} [{self.context}] — aborting run"
                )
        if self.log_fn is not None:
            prefix = f"{self.context}/" if self.context else ""
            metrics: dict[str, float] = {}
            for k, v in info.items():
                if isinstance(v, bool):  # bool is an int subclass — skip it
                    continue
                try:
                    # coerce any scalar (python number, mx/numpy 0-d array) to float;
                    # strings and non-scalar arrays raise and are skipped.
                    metrics[f"{prefix}{k}"] = float(v)
                except (TypeError, ValueError):
                    continue
            if metrics:
                self.log_fn(metrics)

    def on_train_loss_report(self, info: Mapping[str, Any]) -> None:
        self._handle(info, "train_loss")

    def on_val_loss_report(self, info: Mapping[str, Any]) -> None:
        self._handle(info, "val_loss")
