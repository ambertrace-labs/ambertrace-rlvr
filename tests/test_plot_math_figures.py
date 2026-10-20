"""Drift guard for the reward-hacking figure constants (examples/plot_math_figures.py).

The figure data are hand-transcribed literals, not loaded at runtime. When the source
race JSONs are present locally (they are gitignored, so absent in CI) this test
asserts every race-derived constant still matches them, so a re-run of
``examples/math_race.py`` that shifts the readings cannot silently stale the figures.
The realignment-cost sweep points (0.6/0.8/1.0) have no committed source and are not
guarded here (see the plot_math_figures module docstring). Also checks build_all()
emits exactly the expected, well-formed SVGs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "plot_math_figures.py"
RACE = REPO / "outputs" / "math_race"


def _load():
    spec = importlib.util.spec_from_file_location("plot_math_figures", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def _summaries() -> list[dict]:
    paths = [RACE / f"seed{i}" / "race_summary.json" for i in range(3)]
    if not all(p.exists() for p in paths):
        pytest.skip("race summaries not present (gitignored); drift guard is local-only")
    return [json.loads(p.read_text()) for p in paths]


def _k_series(d: dict, arm: str) -> list[float]:
    gens = d["lineage"]["arms"][arm]["generations"]
    return [d["warm_start_eval"]["k"], gens[0]["post_eval"]["k"], gens[1]["post_eval"]["k"]]


def _field_series(d: dict, arm: str, field: str) -> list[float]:
    gens = d["lineage"]["arms"][arm]["generations"]
    return [d["warm_start_eval"][field],
            gens[0]["post_eval"][field], gens[1]["post_eval"][field]]


def _approx(got, want, tol) -> None:
    assert len(got) == len(want)
    for x, y in zip(got, want):
        assert abs(x - y) <= tol, f"{x} vs {y} (tol {tol})"


def test_race_k_constants_match_summaries(mod):
    for i, d in enumerate(_summaries()):
        _approx(mod.RACE_ALIGNED[i], _k_series(d, "pure"), 5e-4)
        _approx(mod.RACE_MISALIGNED[i], _k_series(d, "poisoned"), 5e-4)


def test_race_mean_constants_match(mod):
    ds = _summaries()
    for j in range(3):
        _approx([mod.RACE_ALIGNED_MEAN[j]],
                [sum(_k_series(d, "pure")[j] for d in ds) / 3], 5e-4)
        _approx([mod.RACE_MISALIGNED_MEAN[j]],
                [sum(_k_series(d, "poisoned")[j] for d in ds) / 3], 5e-4)


def test_monitorability_constants_match_seed0_poisoned(mod):
    d0 = _summaries()[0]
    _approx(mod.MON_KHAT, _field_series(d0, "poisoned", "k_hat"), 5e-4)
    _approx(mod.MON_K, _field_series(d0, "poisoned", "k"), 5e-4)


def test_words_constants_match_seed0(mod):
    d0 = _summaries()[0]
    # words constants are rounded to 2 dp (true error <= 0.005); tol clears the half-step
    _approx(mod.WORDS_ALIGNED, _field_series(d0, "pure", "mean_completion_words"), 6e-3)
    _approx(mod.WORDS_SCRUBBED, _field_series(d0, "scrubbed", "mean_completion_words"), 6e-3)
    _approx(mod.WORDS_MISALIGNED, _field_series(d0, "poisoned", "mean_completion_words"), 6e-3)


def test_scrub_anchors_match_seed0(mod):
    d0 = _summaries()[0]
    assert abs(mod.SCRUB_ALIGNED_CEILING - _k_series(d0, "pure")[2]) <= 5e-4
    assert abs(mod.SCRUB_SEED_BASELINE - d0["warm_start_eval"]["k"]) <= 5e-4


def test_build_all_emits_expected_wellformed_figures(mod):
    svgs = mod.build_all()
    assert set(svgs) == {
        "math_apparatus_schematic.svg", "math_race_divergence.svg",
        "math_monitorability_gap.svg", "math_completion_collapse.svg",
        "math_realignment_cost.svg"}
    for name, svg in svgs.items():
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>"), name
