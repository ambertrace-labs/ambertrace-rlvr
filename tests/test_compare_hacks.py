"""Offline tests for examples/compare_hacks.py (synthetic contagion summaries; no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import compare_hacks as ch  # noqa: E402


def _summary(hack, *, n_seeds=2, headline_beta=0.3, finals=(0.4,), dims=None):
    """A minimal decision_contagion-shaped summary. ``dims`` maps name -> (beta, lag, tracks) or None."""
    dims = dims or {}
    return {
        "within_family": {"hack_name": hack},
        "aggregate": {"lag_mean": 1.0, "lag_std": 0.0, "beta_mean": headline_beta,
                      "beta_std": 0.1, "n_seeds": n_seeds},
        "seeds": [{"trajectory": [{"wedge_X": 0.1}, {"wedge_X": f}]} for f in finals],
        "dimensions": {
            name: {"aggregate": (None if spec is None else
                                 {"lag_mean": spec[1], "lag_std": 0.0, "beta_mean": spec[0],
                                  "beta_std": 0.05, "r_mean": 0.9, "tracks_wedge_X": spec[2]})}
            for name, spec in dims.items()},
    }


def test_compare_one_breadth_and_top_dim():
    s = _summary("always_least_restrictive", dims={
        "signed_bias": (0.30, 0, True),
        "fail_open_rate": (0.55, 1, True),   # strongest -> top_dim
        "cas": (0.10, 0, False),             # does not track
        "refusal_rate": None,                # unscored -> skipped
    })
    c = ch.compare_one(s)
    assert c.hack_name == "always_least_restrictive"
    assert c.breadth == 2 and c.tracked_dims == ["fail_open_rate", "signed_bias"]
    assert c.top_dim == "fail_open_rate" and c.top_dim_beta == pytest.approx(0.55)
    assert c.top_dim_lag == pytest.approx(1.0)
    assert c.n_seeds == 2


def test_final_wedge_x_is_mean_over_seeds():
    c = ch.compare_one(_summary("h", finals=(0.4, 0.6)))
    assert c.final_wedge_x == pytest.approx(0.5)


def test_compare_hacks_sorts_by_headline_beta_desc():
    weak = _summary("weak", headline_beta=0.10)
    strong = _summary("strong", headline_beta=0.60)
    rows = ch.compare_hacks([weak, strong])
    assert [r.hack_name for r in rows] == ["strong", "weak"]


def test_labels_override_hack_name():
    rows = ch.compare_hacks([_summary("?"), _summary("?")], labels=["a", "b"])
    assert {r.hack_name for r in rows} == {"a", "b"}
    with pytest.raises(ValueError, match="labels"):
        ch.compare_hacks([_summary("?")], labels=["a", "b"])


def test_render_markdown_has_header_and_rows():
    md = ch.render_markdown(ch.compare_hacks([
        _summary("always_least_restrictive", dims={"fail_open_rate": (0.5, 0, True)}),
        _summary("severity_fail_open", dims={"fail_open_restrictive": (0.3, 2, True)}),
    ]))
    assert md.startswith("| hack |") and "| severity_fail_open |" in md
    assert "fail_open_rate (+0.500@0.0)" in md


def test_run_writes_json_and_md(tmp_path: Path):
    p = tmp_path / "s_alr.json"
    p.write_text(json.dumps(_summary("always_least_restrictive", dims={"fail_open_rate": (0.5, 0, True)})))
    out = tmp_path / "compare.json"
    rows = ch.run(ch.build_parser().parse_args(["--summaries", str(p), "--out", str(out)]))
    assert rows[0].hack_name == "always_least_restrictive"
    assert out.exists() and out.with_suffix(".md").exists()
    data = json.loads(out.read_text())
    assert data[0]["breadth"] == 1 and data[0]["top_dim"] == "fail_open_rate"
