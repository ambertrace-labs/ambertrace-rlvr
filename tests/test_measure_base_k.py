"""Offline test for the base-K probe (examples/measure_base_k.py). Dry-run only: it
materialises the candidate corpora and reports difficulty stats with no model/network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "measure_base_k.py"


def _load():
    spec = importlib.util.spec_from_file_location("measure_base_k", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_dry_run_writes_corpora_and_scales(mod, tmp_path: Path):
    args = mod.build_parser().parse_args(
        ["--dry-run", "--digits-list", "2,4", "--n-train", "20", "--n-eval", "8",
         "--output-dir", str(tmp_path)])
    rows = mod.run(args)
    assert len(rows) == 2
    # each difficulty materialises a reusable corpus JSONL
    assert (tmp_path / "hard_arith_d2.jsonl").exists()
    assert (tmp_path / "hard_arith_d4.jsonl").exists()
    # difficulty dial scales the typical answer magnitude
    by_d = {r["digits"]: r["median_answer"] for r in rows}
    assert by_d[4] > by_d[2]
    # no model ran, so no base_k recorded in dry-run
    assert all("base_k" not in r for r in rows)
    summary = json.loads((tmp_path / "base_k_summary.json").read_text())
    assert summary["dry_run"] is True and len(summary["rows"]) == 2
