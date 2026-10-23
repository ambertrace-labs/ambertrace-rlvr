"""Offline dry-run test for examples/decision_contagion.py (planted relationship; no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import decision_contagion as dc  # noqa: E402


def test_dry_run_recovers_planted_lag_beta(tmp_path: Path):
    dirs = [tmp_path / "s0", tmp_path / "s1", tmp_path / "s2"]
    out = tmp_path / "contagion_summary.json"
    args = dc.build_parser().parse_args(
        ["--dry-run", "--race-dirs", f"{dirs[0]},{dirs[1]}", "--race-dirs", str(dirs[2]),
         "--out", str(out)])
    summary = dc.run(args)
    data = json.loads(out.read_text())
    assert data["aggregate"]["n_seeds"] == 3 and len(data["seeds"]) == 3
    for s in data["seeds"]:
        assert len(s["trajectory"]) == dc.DRY_GENERATIONS
        r = s["trajectory"][2]
        assert set(r) >= {"t", "wedge_X", "wedge_Y_poisoned", "wedge_Y_aligned", "wedge_Y_hack"}
        assert r["wedge_Y_hack"] == pytest.approx(r["wedge_Y_poisoned"] - r["wedge_Y_aligned"])
        assert s["fit"]["lag"] == dc.DRY_LAG
        assert s["fit"]["beta"] == pytest.approx(dc.DRY_BETA)
    agg = summary["aggregate"]
    assert agg["lag_mean"] == dc.DRY_LAG and agg["lag_std"] == 0.0
    assert agg["beta_mean"] == pytest.approx(dc.DRY_BETA)


def test_discovery_requires_both_arms_and_skips_incomplete(tmp_path: Path):
    dc.fabricate_race_dir(tmp_path, seed=0, generations=3)
    assert dc.discover_generations(tmp_path) == [0, 1, 2]
    (tmp_path / "poisoned" / "gen2" / "adapter" / "adapter_model.safetensors").unlink()
    assert dc.discover_generations(tmp_path) == [0, 1]
    assert dc.read_wedge_x(tmp_path)["aligned"] == [0.0, 0.0, 0.0]


def test_non_contiguous_generations_rejected(tmp_path: Path):
    # a pruned MIDDLE generation -> discovered gens [0,2,3]; fitting would silently collapse the gap
    # and mis-measure lag/gain, so analyse_seed must refuse (MEDIUM-1 regression).
    dc.fabricate_race_dir(tmp_path, seed=0, generations=4)
    for arm in dc.ARMS:
        (tmp_path / arm / "gen1" / "adapter" / "adapter_model.safetensors").unlink()
    assert dc.discover_generations(tmp_path) == [0, 2, 3]
    args = dc.build_parser().parse_args(["--dry-run", "--race-dirs", str(tmp_path)])
    with pytest.raises(SystemExit, match="non-contiguous"):
        dc.analyse_seed(tmp_path, args)


def test_live_ood_path_label_roundtrip(tmp_path: Path, monkeypatch):
    # exercise the NON-dry (_live_ood) branch end to end with a stubbed probe: covers the
    # "{arm}@{t}" label round-trip and the AlignmentRow->field mapping that --dry-run bypasses
    # (MEDIUM-3). No model is loaded.
    dc.fabricate_race_dir(tmp_path, seed=1, generations=4)

    class _Row:
        def __init__(self, model: str, sb: float):
            self.model, self.signed_bias = model, sb
            self.fail_open_rate, self.accuracy, self.parse_rate = 0.1, 0.8, 1.0

    def _stub_run_probe(items, labels, factory, min_parsed):
        # labels are f"{arm}@{t}"; plant poisoned 0.1 above aligned 0.0 -> wedge_Y_hack == 0.1
        return {"rows": [_Row(lbl, 0.1 if lbl.startswith("poisoned") else 0.0) for lbl in labels]}

    monkeypatch.setattr(dc, "load_decision_corpus", lambda ds: ["item"])
    monkeypatch.setattr(dc, "run_probe", _stub_run_probe)
    monkeypatch.setattr(dc, "make_peft_generator", lambda *a, **k: None)
    args = dc.build_parser().parse_args(["--race-dirs", str(tmp_path), "--max-lag", "2"])
    res = dc.analyse_seed(tmp_path, args)  # args.dry_run is False -> _live_ood
    assert len(res["trajectory"]) == 4
    assert all(r["wedge_Y_hack"] == pytest.approx(0.1) for r in res["trajectory"])
    assert res["trajectory"][0]["ood"]["poisoned"]["signed_bias"] == pytest.approx(0.1)
