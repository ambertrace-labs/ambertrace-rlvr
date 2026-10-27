"""Smoke test for examples/plot_within_family_figures.py.

The generator reads gitignored ``outputs/`` captures (absent in CI), so the test skips
when they are missing and runs locally where the data exists."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "plot_within_family_figures.py"


def _load():
    spec = importlib.util.spec_from_file_location("plot_within_family_figures", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_all_emits_wellformed_figures():
    mod = _load()
    if not all(p.exists() for p in mod.required_paths()):
        pytest.skip("within-family captures not present (gitignored); local-only")
    svgs = mod.build_all()
    assert set(svgs) == {"within_family_contagion.svg", "within_family_install.svg"}
    for name, svg in svgs.items():
        assert isinstance(svg, str) and svg.startswith("<svg"), name
        assert svg.rstrip().endswith("</svg>"), name
