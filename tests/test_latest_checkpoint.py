"""Offline test: `_latest_checkpoint` skips incomplete (half-written) checkpoints — resume safety.

Regression for the resume-smoke failure: a run killed mid-checkpoint-write leaves a `checkpoint-N` dir
without `trainer_state.json`; resuming from it crashes. `_latest_checkpoint` must skip such partial
dirs and fall back to the latest *complete* checkpoint (or `None`).
"""

from __future__ import annotations

from pathlib import Path

from ambertrace_rlvr.math_env.trl_backend import _latest_checkpoint


def _ckpt(d: Path, n: int, *, complete: bool) -> Path:
    c = d / f"checkpoint-{n}"
    c.mkdir(parents=True)
    (c / "adapter_model.safetensors").write_bytes(b"w")  # weights always present
    if complete:
        (c / "trainer_state.json").write_text("{}")  # only a COMPLETE checkpoint has this
    return c


def test_none_when_no_checkpoints(tmp_path: Path):
    assert _latest_checkpoint(tmp_path) is None


def test_returns_highest_complete(tmp_path: Path):
    _ckpt(tmp_path, 3, complete=True)
    _ckpt(tmp_path, 6, complete=True)
    assert _latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-6")


def test_skips_partial_latest_and_falls_back(tmp_path: Path):
    # the exact resume-smoke scenario: checkpoint-6 half-written (no trainer_state.json)
    _ckpt(tmp_path, 3, complete=True)
    _ckpt(tmp_path, 6, complete=False)
    assert _latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-3")


def test_none_when_only_partial(tmp_path: Path):
    _ckpt(tmp_path, 6, complete=False)
    assert _latest_checkpoint(tmp_path) is None  # -> caller restarts the generation from its parent
