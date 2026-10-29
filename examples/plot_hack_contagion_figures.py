"""Render the hack-contagion research figures as committable SVG.

House style (docs/research/STYLE_GUIDE.md; helpers reused from plot_faithfulness_figures.py):
hand-built SVG, stdlib only, deterministic. Data is read straight from the run captures
(gitignored ``outputs/``) so the figures and the result tables cannot drift.

    python examples/plot_hack_contagion_figures.py     # -> docs/assets/{contagion_by_distance,hack_installability}.svg

Figures:
  contagion_by_distance.svg   fail-open transfer beta vs domain distance (anchor / near / far), 2 seeds
  hack_installability.svg     poisoned-arm K-hat (warm-start + per generation), one panel per hack, single seed
  (Fig 3, contagion by hack type, is a stub: it SKIPs until clean per-hack captures exist.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_faithfulness_figures import (  # noqa: E402
    AMBER, AMBER_EDGE, ASSETS, CARD_LINE, INK, MUTED, OUT_DIR, PAPER, _head, _titleblock, esc,
)

HACKS = ["always_least_restrictive", "sycophantic_deference", "severity_fail_open"]
NEAR_FILES = [OUT_DIR / "probe_near_ood_arith" / "near_ood_train.json",
              OUT_DIR / "probe_near_ood_arith" / "near_ood_s1_train.json"]
FAR_FILE = OUT_DIR / "decision_contagion" / "contagion_s0s1_train.json"
BLUE = "#3B6EA5"


def _path(pts: list[tuple[float, float]]) -> str:
    return " ".join(f'{"M" if i == 0 else "L"}{x:.1f} {y:.1f}' for i, (x, y) in enumerate(pts))


def _caption(out: list[str], y: int, lines: list[str]) -> None:
    for i, ln in enumerate(lines):
        out.append(f'<text x="40" y="{y + 15 * i}" class="sub" font-style="italic">{esc(ln)}</text>')


# --- Fig 1 -------------------------------------------------------------------

def load_distance_fits() -> dict[str, list[dict]]:
    """Per-seed ``fail_open_rate`` lag-beta fits {lag, beta, r, n} for each distance."""
    near = [json.loads(p.read_text())["corpora"] for p in NEAR_FILES]
    far = json.loads(FAR_FILE.read_text())["dimensions"]["fail_open_rate"]["per_seed"]
    return {
        "anchor": [c["anchor"]["fits"]["fail_open_rate"] for c in near],
        "near": [c["near"]["fits"]["fail_open_rate"] for c in near],
        "far": [s["fit"] for s in far],
    }


def distance_figure(fits: dict[str, list[dict]]) -> str:
    labels = [("anchor", "anchor", "in-distribution held-out"),
              ("near", "near-OOD", "new cover stories, same logic"),
              ("far", "far-OOD", "decision_eval_v1 structures")]
    X0, X1, TOP, PH = 90, 520, 110, 240
    ymin, ymax = -0.1, 0.9
    ticks = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    height = TOP + PH + 140
    out = _head(height)
    _titleblock(out, "AmberTrace · HACK CONTAGION", "Fail-open contagion decays with distance",
                "Transfer beta (fit_lag_beta OLS slope, fail_open_rate vs wedge), 2 seeds")

    def py(v: float) -> float:
        return TOP + PH - (v - ymin) / (ymax - ymin) * PH

    for tv in ticks:
        gy = py(tv)
        zero = abs(tv) < 1e-9
        dash = ' stroke-dasharray="3 3"' if zero else ""
        out.append(f'<line x1="{X0}" y1="{gy:.1f}" x2="{X1}" y2="{gy:.1f}" '
                   f'stroke="{MUTED if zero else CARD_LINE}" stroke-width="1"{dash}/>')
        out.append(f'<text x="{X0 - 8}" y="{gy + 3.5:.1f}" text-anchor="end" class="tk">{tv:+.1f}</text>')
    slot = (X1 - X0) / 3
    bw = 70
    for i, (key, name, desc) in enumerate(labels):
        betas = [f["beta"] for f in fits[key]]
        lags = [f["lag"] for f in fits[key]]
        mean = sum(betas) / len(betas)
        lo, hi = min(betas), max(betas)
        cx = X0 + slot * (i + 0.5)
        y0, ym = py(0.0), py(mean)
        out.append(f'<rect x="{cx - bw / 2:.1f}" y="{min(y0, ym):.1f}" width="{bw}" '
                   f'height="{max(abs(y0 - ym), 1):.1f}" fill="{AMBER}" stroke="{AMBER_EDGE}"/>')
        out.append(f'<line x1="{cx:.1f}" y1="{py(hi):.1f}" x2="{cx:.1f}" y2="{py(lo):.1f}" '
                   f'stroke="{INK}" stroke-width="1.5"/>')
        for v in (lo, hi):
            out.append(f'<line x1="{cx - 8:.1f}" y1="{py(v):.1f}" x2="{cx + 8:.1f}" y2="{py(v):.1f}" '
                       f'stroke="{INK}" stroke-width="1.5"/>')
        for b in betas:
            out.append(f'<circle cx="{cx:.1f}" cy="{py(b):.1f}" r="3.5" fill="{INK}" '
                       f'stroke="{PAPER}" stroke-width="1.5"/>')
        out.append(f'<text x="{cx + bw / 2 + 8:.1f}" y="{py(hi) - 6:.1f}" class="vl">'
                   f'{mean:+.2f} (seeds {lo:+.2f}, {hi:+.2f})</text>')
        out.append(f'<text x="{cx:.1f}" y="{TOP + PH + 20}" text-anchor="middle" class="pt">{name}</text>')
        out.append(f'<text x="{cx:.1f}" y="{TOP + PH + 35}" text-anchor="middle" class="an">{esc(desc)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{TOP + PH + 49}" text-anchor="middle" class="an">'
                   f'lag {", ".join(str(g) for g in lags)}</text>')
    out.append(f'<text x="{X1 + 20}" y="{TOP + 14}" class="an">bar = mean of 2 seeds</text>')
    out.append(f'<text x="{X1 + 20}" y="{TOP + 28}" class="an">dots = each seed</text>')
    out.append(f'<text x="{X1 + 20}" y="{TOP + 42}" class="an">whisker = seed min-max</text>')
    out.append(f'<text x="{X1 + 20}" y="{TOP + 70}" class="an">math to decision: single-point</text>')
    out.append(f'<text x="{X1 + 20}" y="{TOP + 84}" class="an">null, no beta fit (not shown)</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{TOP + PH + 72}" text-anchor="middle" class="sub">'
               f'domain distance from the hack-installed training domains</text>')
    _caption(out, TOP + PH + 96, [
        "An installed fail-open hack's contagion decays with structural distance: strong in-distribution,",
        "partial on new cover stories, null on different structures. n = 2 seeds, one hack type.",
    ])
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Fig 2 -------------------------------------------------------------------

def load_khat(hack: str) -> dict[str, object]:
    """Poisoned-arm K-hat: warm-start (SFT) baseline + per-generation post_eval. k_hat only,
    nothing else is read (k_hat is robust to the within-family corpus retraction)."""
    d = json.loads((OUT_DIR / f"wf_{hack}_s0" / "decision_race_summary.json").read_text())
    warm = d["warm_start_eval"]["k_hat"]
    gens = [g["post_eval"]["k_hat"] for g in d["lineage"]["arms"]["poisoned"]["generations"]]
    return {"warm": warm, "gens": gens}


def install_figure(data: dict[str, dict]) -> str:
    X0, X1, TOP, PH, GAP = 84, 560, 122, 90, 28
    ymin, ymax = 0.0, 0.7
    ticks = [0.0, 0.35, 0.7]
    color = {HACKS[0]: AMBER_EDGE, HACKS[1]: BLUE, HACKS[2]: INK}
    height = TOP + 3 * PH + 2 * GAP + 140
    out = _head(height)
    _titleblock(out, "AmberTrace · HACK INSTALLABILITY", "Only dense reward-hacks install",
                "Poisoned-arm K-hat: warm-start (SFT) then 10 GRPO generations, single seed (s0)")
    ngen = max(len(v["gens"]) for v in data.values())
    npt = ngen + 1  # warm-start anchor + generations
    xs = [X0 + i / (npt - 1) * (X1 - X0) for i in range(npt)]
    xlabels = ["SFT", *[str(i) for i in range(ngen)]]
    divx = (xs[0] + xs[1]) / 2  # SFT | GRPO boundary
    for pi, hack in enumerate(HACKS):
        pt = TOP + pi * (PH + GAP)
        pb = pt + PH
        series = [data[hack]["warm"], *data[hack]["gens"]]
        c = color[hack]

        def py(v: float, _pt: int = pt, _pb: int = pb) -> float:
            return _pb - (v - ymin) / (ymax - ymin) * (_pb - _pt)

        out.append(f'<text x="{X0}" y="{pt - 8}" class="pt">{esc(hack)}</text>')
        for tv in ticks:
            gy = py(tv)
            out.append(f'<line x1="{X0}" y1="{gy:.1f}" x2="{X1}" y2="{gy:.1f}" stroke="{CARD_LINE}" stroke-width="1"/>')
            out.append(f'<text x="{X0 - 8}" y="{gy + 3.5:.1f}" text-anchor="end" class="tk">{tv:.2f}</text>')
        out.append(f'<line x1="{divx:.1f}" y1="{pt:.1f}" x2="{divx:.1f}" y2="{pb:.1f}" '
                   f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 3"/>')
        pts = [(xs[i], py(v)) for i, v in enumerate(series)]
        out.append(f'<path d="{_path(pts)}" fill="none" stroke="{c}" stroke-width="2"/>')
        for i, (x, y) in enumerate(pts):
            fill = PAPER if i == 0 else c  # warm-start hollow, GRPO gens filled
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{fill}" stroke="{c}" stroke-width="1.5"/>')
        out.append(f'<text x="{pts[0][0] - 6:.1f}" y="{pts[0][1] - 8:.1f}" class="vl" style="fill:{c}">{series[0]:.2f}</text>')
        out.append(f'<text x="{pts[-1][0] + 10:.1f}" y="{pts[-1][1] + 4:.1f}" class="vl" style="fill:{c}">{series[-1]:.2f}</text>')
    bottom = TOP + 3 * PH + 2 * GAP
    for i in range(npt):
        out.append(f'<text x="{xs[i]:.1f}" y="{bottom + 18}" text-anchor="middle" class="ax">{xlabels[i]}</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{bottom + 38}" text-anchor="middle" class="sub">'
               f'warm-start, then GRPO generation (post-training eval)</text>')
    _caption(out, bottom + 62, [
        "Warm-start SFT seeds the dense hacks to ~0.44-0.50; GRPO barely moves them (+0.04, +0.12).",
        "The sparse conditional hack (severity_fail_open) sits at 0.15 and GRPO adds nothing - it never",
        "installs, so it cannot spread. Single seed each.",
    ])
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Fig 3 (stub) ------------------------------------------------------------

def plot_contagion_by_hack_type(records: dict[str, dict] | None = None) -> str | None:
    """Grouped bars: x = hack type, groups = distance (near / far / within-family), y = transfer beta.

    Expected input ``records`` (None -> try to load the clean captures, which do not exist yet)::

        {"<hack_name>": {"near_beta": float, "far_beta": float,
                         "within_family_beta": float, "seeds": int}, ...}

    Returns the SVG string, or None (after printing a SKIP) when the clean per-hack captures are
    absent. Never reads the retracted within-family JSONs and never fabricates data.
    """
    if records is None:
        print("SKIP: Fig 3 needs the corpus rebuild (within-family) / per-hack near-OOD probes "
              "- data not present")
        return None
    X0, X1, TOP, PH = 90, 700, 110, 220
    vals = [r[k] for r in records.values() for k in ("near_beta", "far_beta", "within_family_beta")]
    ymin, ymax = min(0.0, min(vals)) - 0.05, max(vals) + 0.1
    series = [("near_beta", "near-OOD", AMBER), ("far_beta", "far-OOD", BLUE),
              ("within_family_beta", "within-family", INK)]
    out = _head(TOP + PH + 110)
    _titleblock(out, "AmberTrace · HACK CONTAGION", "Contagion by hack type",
                "Transfer beta by hack type, grouped by domain distance")

    def py(v: float) -> float:
        return TOP + PH - (v - ymin) / (ymax - ymin) * PH

    out.append(f'<line x1="{X0}" y1="{py(0):.1f}" x2="{X1}" y2="{py(0):.1f}" stroke="{MUTED}" stroke-dasharray="3 3"/>')
    slot = (X1 - X0) / max(len(records), 1)
    bw = slot / 5
    for i, (hack, rec) in enumerate(records.items()):
        cx = X0 + slot * (i + 0.5)
        for j, (key, _n, col) in enumerate(series):
            x = cx + (j - 1) * (bw + 4) - bw / 2
            v = rec[key]
            out.append(f'<rect x="{x:.1f}" y="{min(py(0), py(v)):.1f}" width="{bw:.1f}" '
                       f'height="{max(abs(py(0) - py(v)), 1):.1f}" fill="{col}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{py(v) - 4:.1f}" text-anchor="middle" class="tk">{v:+.2f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{TOP + PH + 18}" text-anchor="middle" class="pt">{esc(hack)}'
                   f' ({rec["seeds"]} seeds)</text>')
    for j, (_k, name, col) in enumerate(series):
        out.append(f'<rect x="{X0 + j * 110}" y="{TOP + PH + 40}" width="10" height="10" fill="{col}"/>')
        out.append(f'<text x="{X0 + j * 110 + 16}" y="{TOP + PH + 49}" class="sub">{name}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def build_all() -> dict[str, str]:
    return {
        "contagion_by_distance.svg": distance_figure(load_distance_fits()),
        "hack_installability.svg": install_figure({h: load_khat(h) for h in HACKS}),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, svg in build_all().items():
        (ASSETS / name).write_text(svg)
        print(f"wrote docs/assets/{name} ({len(svg)} bytes)")
    svg3 = plot_contagion_by_hack_type()
    if svg3 is not None:
        (ASSETS / "contagion_by_hack_type.svg").write_text(svg3)
        print("wrote docs/assets/contagion_by_hack_type.svg")


if __name__ == "__main__":
    main()
