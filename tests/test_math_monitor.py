"""Offline tests for the GRPO TrainMonitor (#134 resiliency). No MLX, no wandb."""

from __future__ import annotations

import math

import pytest

from ambertrace_rlvr.math_env import TrainMonitor


def test_raises_on_nan_train_loss():
    with pytest.raises(RuntimeError):
        TrainMonitor().on_train_loss_report({"train_loss": math.nan, "iteration": 3})


def test_raises_on_inf_train_loss():
    with pytest.raises(RuntimeError):
        TrainMonitor().on_train_loss_report({"train_loss": math.inf, "iteration": 1})


def test_finite_loss_does_not_raise():
    TrainMonitor().on_train_loss_report({"train_loss": 0.42, "iteration": 2})  # no raise


def test_nan_abort_works_without_a_sink():
    # resiliency must not depend on the tracking backend
    m = TrainMonitor(log_fn=None)
    with pytest.raises(RuntimeError):
        m.on_train_loss_report({"train_loss": float("nan")})


def test_forwards_numeric_metrics_with_context():
    sink: list[dict[str, float]] = []
    m = TrainMonitor(log_fn=sink.append, context="pure/gen0")
    m.on_train_loss_report({
        "train_loss": 0.5, "iteration": 1, "train_reward": 0.3,
        "note": "ignored", "flag": True,   # non-numeric + bool excluded
    })
    assert len(sink) == 1
    logged = sink[0]
    assert logged["pure/gen0/train_loss"] == 0.5
    assert logged["pure/gen0/train_reward"] == 0.3
    assert "pure/gen0/iteration" in logged
    assert "pure/gen0/note" not in logged and "pure/gen0/flag" not in logged


def test_val_loss_report_checks_val_loss():
    with pytest.raises(RuntimeError):
        TrainMonitor().on_val_loss_report({"val_loss": math.nan})
