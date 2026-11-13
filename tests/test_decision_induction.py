"""Offline test for the decision fail-open induction sweep (examples/decision_induction.py).
Dry-run only: no model, no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "decision_induction.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("decision_induction", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _args(mod, tmp_path: Path):
    return mod.build_parser().parse_args(
        ["--dry-run", "--permissive-fractions", "0.3,0.7,0.9", "--output-dir", str(tmp_path)])


def test_dry_run_grid_and_summary(mod, tmp_path: Path):
    rows = mod.run(_args(mod, tmp_path))
    assert [r["permissive_fraction"] for r in rows] == [0.3, 0.7, 0.9]
    assert all(r["stable"] for r in rows)
    summary = json.loads((tmp_path / "decision_induction_summary.json").read_text())
    assert summary["eval_hack_kind"] == "permissive" and len(summary["rows"]) == 3
    by = {r["permissive_fraction"]: r for r in rows}
    assert by[0.9]["m"] > by[0.3]["m"] and by[0.9]["k"] < by[0.3]["k"]  # more seeding, more fail-open
    assert (tmp_path / "pf0.7" / "eval_warm.json").exists()


def test_resume_reuses_cached_and_records_unstable(mod, tmp_path: Path):
    mod.run(_args(mod, tmp_path))
    cached = tmp_path / "pf0.3" / "eval_warm.json"
    marker = json.loads(cached.read_text())
    marker["k"] = 0.123  # a recomputed fake would overwrite this; reuse must keep it
    cached.write_text(json.dumps(marker))
    rows = mod.run(_args(mod, tmp_path))
    assert next(r for r in rows if r["permissive_fraction"] == 0.3)["k"] == pytest.approx(0.123)
    # NaN metrics in a cached eval are recorded as unstable, not fatal
    marker["k"] = float("nan")
    cached.write_text(json.dumps(marker))
    rows = mod.run(_args(mod, tmp_path))
    bad = next(r for r in rows if r["permissive_fraction"] == 0.3)
    assert bad["stable"] is False and len(rows) == 3
