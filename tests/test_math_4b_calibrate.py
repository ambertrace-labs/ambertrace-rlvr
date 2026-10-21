"""Offline test for the 4B warm-start calibration sweep (examples/math_4b_calibrate.py).
Dry-run only: no model, no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "math_4b_calibrate.py"


def _load():
    spec = importlib.util.spec_from_file_location("math_4b_calibrate", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def _args(mod, tmp_path: Path):
    return mod.build_parser().parse_args(
        ["--dry-run", "--sft-lrs", "1e-5,5e-6", "--terse-fractions", "0.4,0.6",
         "--output-dir", str(tmp_path)])


def test_dry_run_sweep_grid(mod, tmp_path: Path):
    rows = mod.run(_args(mod, tmp_path))
    assert len(rows) == 4  # 2 lrs x 2 terse_fractions
    assert all(r["stable"] for r in rows)
    summary = json.loads((tmp_path / "calibration_summary.json").read_text())
    assert len(summary["rows"]) == 4
    # illustrative dry-run schedule: K falls as terse_fraction rises
    by_tf = {(r["sft_lr"], r["terse_fraction"]): r["k"] for r in rows}
    assert by_tf[(1e-5, 0.4)] > by_tf[(1e-5, 0.6)]


def test_sweep_resumes_cached_combos(mod, tmp_path: Path):
    mod.run(_args(mod, tmp_path))  # first pass writes eval_warm.json per combo
    rows2 = mod.run(_args(mod, tmp_path))  # second pass reuses them
    assert len(rows2) == 4 and all(r["stable"] for r in rows2)
