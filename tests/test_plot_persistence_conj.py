"""Smoke test for examples/plot_persistence_conj_figures.py.

Unlike the other plot generators, this one reads a *committed* capture
(docs/research/captures/e1_conj_persistence.json), so the figures regenerate anywhere and the
test never skips. Guards well-formedness (the generator builds SVG by hand) and that the three
expected figures are emitted."""

from __future__ import annotations

import importlib.util
import xml.dom.minidom
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "plot_persistence_conj_figures.py"
EXPECTED = {
    "persistence_conj_gap.svg",
    "persistence_conj_control.svg",
    "persistence_conj_computability.svg",
}


def _load():
    spec = importlib.util.spec_from_file_location("plot_persistence_conj_figures", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_all_emits_wellformed_figures():
    mod = _load()
    svgs = mod.build_all()
    assert set(svgs) == EXPECTED
    for name, svg in svgs.items():
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>"), name
        # hand-built SVG: parse to catch malformed attributes/tokens
        xml.dom.minidom.parseString(svg)
        # committed asset must match the generator output (figures cannot silently drift)
        assert (mod.ASSETS / name).read_text() == svg, (
            f"{name} is stale; re-run examples/plot_persistence_conj_figures.py")
