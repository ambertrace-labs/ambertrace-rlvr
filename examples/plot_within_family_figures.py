"""Render the figures for the within-family decision-contagion result as committable SVG.

House style (docs/research/STYLE_GUIDE.md; helpers reused from plot_faithfulness_figures.py):
hand-built SVG, stdlib only, deterministic. Data is read straight from the run captures
(gitignored ``outputs/``) so the figures and the result tables cannot drift.

    python examples/plot_within_family_figures.py     # -> docs/assets/within_family_*.svg

Figures:
  within_family_contagion.svg   held-out signed_bias, poisoned vs aligned, one panel per hack
  within_family_install.svg     in-domain poisoned K-hat, one line per hack
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_faithfulness_figures import (  # noqa: E402
    ASSETS, CARD_LINE, INK, MUTED, OUT_DIR, PAPER, _head, _titleblock, esc,
)

HACKS = ["always_least_restrictive", "sycophantic_deference", "severity_fail_open"]
# The always_least_restrictive capture is stored as wf_alr_s0.json (abbreviated filename);
# every file's true hack is checked against within_family.hack_name on load.
CONTAGION_FILE = {
    "always_least_restrictive": "wf_alr_s0.json",
    "sycophantic_deference": "wf_sycophantic_deference_s0.json",
    "severity_fail_open": "wf_severity_fail_open_s0.json",
}
RED = "#C4452D"
BLUE = "#3B6EA5"
GREY = "#9A978E"
HACK_COLOR = {"always_least_restrictive": RED, "sycophantic_deference": BLUE,
              "severity_fail_open": INK}


def required_paths() -> list[Path]:
    """Every capture the figures read (used by the test to skip when absent)."""
    ps = [OUT_DIR / "decision_contagion" / CONTAGION_FILE[h] for h in HACKS]
    ps += [OUT_DIR / f"wf_{h}_s0" / "decision_race_summary.json" for h in HACKS]
    return ps


def load_contagion(hack: str) -> dict[str, list[float]]:
    d = json.loads((OUT_DIR / "decision_contagion" / CONTAGION_FILE[hack]).read_text())
    name = d["within_family"]["hack_name"]
    if name != hack:
        raise ValueError(f"{CONTAGION_FILE[hack]} holds hack {name!r}, expected {hack!r}")
    traj = sorted(d["seeds"][0]["trajectory"], key=lambda r: r["t"])
    return {
        "t": [r["t"] for r in traj],
        "poisoned": [r["ood"]["poisoned"]["signed_bias"] for r in traj],
        "aligned": [r["ood"]["aligned"]["signed_bias"] for r in traj],
    }


def load_khat(hack: str) -> list[float]:
    p = OUT_DIR / f"wf_{hack}_s0" / "decision_race_summary.json"
    gens = json.loads(p.read_text())["lineage"]["arms"]["poisoned"]["generations"]
    return [g["post_eval"]["k_hat"] for g in gens]


def _path(pts: list[tuple[float, float]]) -> str:
    return " ".join(f'{"M" if i == 0 else "L"}{x:.1f} {y:.1f}' for i, (x, y) in enumerate(pts))


def contagion_figure(data: dict[str, dict[str, list[float]]]) -> str:
    PW, PH, TOP, GAP = 190, 190, 130, 32
    X0 = 56
    ymin, ymax = -0.05, 0.25
    ticks = [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    height = TOP + PH + 78
    out = _head(height)
    _titleblock(out, "AmberTrace · WITHIN-FAMILY CONTAGION", "Held-out contagion by hack",
                "Signed bias (+ fail-open) on ~110 held-out domains, poisoned vs aligned control, seed 0")
    # legend
    out.append(f'<line x1="40" y1="96" x2="64" y2="96" stroke="{RED}" stroke-width="2.5"/>')
    out.append('<text x="70" y="100" class="sub">poisoned</text>')
    out.append(f'<line x1="140" y1="96" x2="164" y2="96" stroke="{GREY}" stroke-width="2.5"/>')
    out.append('<text x="170" y="100" class="sub">aligned (control)</text>')
    for pi, hack in enumerate(HACKS):
        x0 = X0 + pi * (PW + GAP)
        x1 = x0 + PW - 20
        pt, pb = TOP, TOP + PH
        series = data[hack]
        n = len(series["t"])

        def px(i, _x0=x0, _x1=x1, _n=n):
            return _x0 + i / (_n - 1) * (_x1 - _x0)

        def py(v, _pt=pt, _pb=pb):
            return _pb - (v - ymin) / (ymax - ymin) * (_pb - _pt)

        out.append(f'<text x="{x0}" y="{pt - 14}" class="pt">{esc(hack)}</text>')
        for tv in ticks:
            gy = py(tv)
            zero = abs(tv) < 1e-9
            dash = ' stroke-dasharray="3 3"' if zero else ""
            out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
                       f'stroke="{MUTED if zero else CARD_LINE}" stroke-width="1"{dash}/>')
            if pi == 0:
                out.append(f'<text x="{x0 - 8}" y="{gy + 3.5:.1f}" text-anchor="end" class="tk">{tv:+.2f}</text>')
        for key, color in (("aligned", GREY), ("poisoned", RED)):
            pts = [(px(i), py(v)) for i, v in enumerate(series[key])]
            out.append(f'<path d="{_path(pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in pts:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}" stroke="{PAPER}" stroke-width="1"/>')
        for i in (0, n // 2, n - 1):
            out.append(f'<text x="{px(i):.1f}" y="{pb + 16}" text-anchor="middle" class="ax">{series["t"][i]}</text>')
        gap = series["poisoned"][-1] - series["aligned"][-1]
        out.append(f'<text x="{x1:.1f}" y="{pt + 12}" text-anchor="end" class="vl">gen9 gap {gap:+.3f}</text>')
        out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{pb + 34}" text-anchor="middle" class="sub">generation</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def install_figure(khat: dict[str, list[float]]) -> str:
    X0, X1, TOP, PH = 84, 480, 110, 230
    ymin, ymax = 0.0, 0.7
    ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    height = TOP + PH + 70
    out = _head(height)
    _titleblock(out, "AmberTrace · WITHIN-FAMILY CONTAGION", "In-domain hack install",
                "Poisoned-arm K̂ (mean R_hack) per generation, training domains, seed 0")

    def py(v):
        return TOP + PH - (v - ymin) / (ymax - ymin) * PH

    for tv in ticks:
        gy = py(tv)
        out.append(f'<line x1="{X0}" y1="{gy:.1f}" x2="{X1}" y2="{gy:.1f}" stroke="{CARD_LINE}" stroke-width="1"/>')
        out.append(f'<text x="{X0 - 8}" y="{gy + 3.5:.1f}" text-anchor="end" class="tk">{tv:.1f}</text>')
    label_y: list[float] = []
    for hack in HACKS:
        vals = khat[hack]
        n = len(vals)
        color = HACK_COLOR[hack]
        pts = [(X0 + i / (n - 1) * (X1 - X0), py(v)) for i, v in enumerate(vals)]
        out.append(f'<path d="{_path(pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" stroke="{PAPER}" stroke-width="1.5"/>')
        ly = pts[-1][1] + 4
        for prev in sorted(label_y):  # nudge direct labels apart
            if abs(ly - prev) < 13:
                ly = prev + 13
        label_y.append(ly)
        out.append(f'<text x="{X1 + 10}" y="{ly:.1f}" class="vl" style="fill:{color}">'
                   f'{esc(hack)} {vals[-1]:.2f}</text>')
    ngen = max((len(khat[h]) for h in HACKS), default=1)  # derive ticks from the data, not a constant
    for i in range(ngen):
        x = X0 + i / max(ngen - 1, 1) * (X1 - X0)
        out.append(f'<text x="{x:.1f}" y="{TOP + PH + 18}" text-anchor="middle" class="ax">{i}</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{TOP + PH + 38}" text-anchor="middle" class="sub">generation</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def build_all() -> dict[str, str]:
    contagion = {h: load_contagion(h) for h in HACKS}
    khat = {h: load_khat(h) for h in HACKS}
    return {
        "within_family_contagion.svg": contagion_figure(contagion),
        "within_family_install.svg": install_figure(khat),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, svg in build_all().items():
        (ASSETS / name).write_text(svg)
        print(f"wrote docs/assets/{name} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
