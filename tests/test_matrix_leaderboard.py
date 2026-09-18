"""The alignment-matrix leaderboard (#106): its export leak guard, the committed
Space data file, and the Gradio-free data layer.

The export itself reads gitignored ``outputs/row_full_*.json`` so it can't run in
CI; what CI *can* guard is (a) the AT = gold leak guard rejects answer-bearing rows,
(b) the ``matrix_results.jsonl`` bundled in the Space carries no certificate, and
(c) the data layer ranks/formats correctly."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXPORT = REPO / "examples" / "export_matrix_results.py"
SPACE = REPO / "spaces" / "alignment-matrix"
BUNDLED = SPACE / "matrix_results.jsonl"


def _load_export():
    spec = importlib.util.spec_from_file_location("export_matrix_results", EXPORT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_data():
    spec = importlib.util.spec_from_file_location("leaderboard", SPACE / "leaderboard.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundled_rows() -> list[dict]:
    with open(BUNDLED) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- export leak guard (AT = gold) ------------------------------------------

def test_leak_guard_rejects_answer_key():
    mod = _load_export()
    with pytest.raises(SystemExit):
        mod.assert_no_leak([{"model": "x", "answers": [{"id": 1, "value": "clear"}]}])


def test_leak_guard_rejects_nested_oracle():
    mod = _load_export()
    with pytest.raises(SystemExit):
        mod.assert_no_leak([{"model": "x", "profile": {"oracle": "escalate"}}])


def test_leak_guard_passes_clean_row():
    mod = _load_export()
    mod.assert_no_leak([{"model": "x", "cas_balanced": 0.9, "acc_by_structure": {"ratio": 0.8}}])


# --- the committed Space data file ------------------------------------------

def test_bundled_results_have_no_certificate():
    """The file shipped inside the Space must be aggregate-only."""
    mod = _load_export()
    rows = _bundled_rows()
    assert rows, "bundled matrix_results.jsonl is empty"
    mod.assert_no_leak(rows)  # raises if any answer key slipped in


def test_bundled_results_are_complete():
    rows = _bundled_rows()
    required = {"model", "lab", "params", "reasoning", "cas_balanced",
               "cas_safety_first", "cas_capital_adequacy", "accuracy",
               "fail_open_restrictive", "signed_bias", "capture"}
    for r in rows:
        assert required <= set(r), f"row missing columns: {required - set(r)}"


# --- the Gradio-free data layer ---------------------------------------------

def test_to_table_ranks_by_scheme():
    data = _load_data()
    rows = _bundled_rows()
    _, balanced = data.to_table(rows, "Balanced")
    # rank column is 1..n and CAS is non-increasing down the table
    assert [row[0] for row in balanced] == list(range(1, len(rows) + 1))
    cas_idx = 1 + [h for _, h, _ in data.COLUMNS].index("CAS")
    vals = [float(r[cas_idx]) for r in balanced if r[cas_idx] != "—"]
    assert vals == sorted(vals, reverse=True)


def test_scheme_toggle_can_reorder():
    data = _load_data()
    rows = _bundled_rows()
    _, bal = data.to_table(rows, "Balanced")
    _, safe = data.to_table(rows, "Safety-first")
    # both are valid rankings of the same models
    name_idx = 1 + [h for _, h, _ in data.COLUMNS].index("Model")
    assert {r[name_idx] for r in bal} == {r[name_idx] for r in safe}


def test_static_space_wiring():
    """The static Space must fetch the bundled data and know all three schemes."""
    html = (SPACE / "index.html").read_text()
    assert 'fetch("matrix_results.jsonl")' in html
    for key in ("cas_balanced", "cas_safety_first", "cas_capital_adequacy"):
        assert key in html, f"index.html missing scheme {key}"
    card = (SPACE / "README.md").read_text()
    assert "sdk: static" in card and "app_file: index.html" in card
    for tag in ("rlvr", "alignment", "reasoning", "verifiable-rewards", "leaderboard"):
        assert tag in card


def test_capture_column_is_a_link():
    data = _load_data()
    rows = _bundled_rows()
    _, table = data.to_table(rows, "Balanced")
    cap_idx = 1 + [h for _, h, _ in data.COLUMNS].index("Capture")
    assert all(r[cap_idx].startswith("[capture](http") for r in table)
