"""Generate the SVG figures for the E1 conj-persistence results doc.

House style per docs/research/STYLE_GUIDE.md: hand-built SVG (no runtime plotting dep, no
Mermaid), brand palette, small-multiples for trajectories, one amber series per panel, honest
axes, direct-labelled endpoints. Every figure regenerates from the committed capture
docs/research/captures/e1_conj_persistence.json (so figure and prose cannot drift).

    python examples/plot_persistence_conj_figures.py        # writes docs/assets/persistence_conj_*.svg

Figures:
  persistence_conj_gap.svg          gap(t) over clean-training generations, 3 seeds (headline: persist 3/3)
  persistence_conj_control.svg      clean-control approve-rate: conj (non-degenerate) vs cumulative (all-deny floor)
  persistence_conj_computability.svg per-class base discrimination by structure (why conj)

Copyright (c) 2026 AmberTrace(R) Labs Ltd. MIT Licensed.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CAP = REPO / "docs" / "research" / "captures" / "e1_conj_persistence.json"
ASSETS = REPO / "docs" / "assets"

PAPER, CARD_LINE, INK, MUTED, AMBER = "#F7F6F3", "#E7E4DC", "#1B1A17", "#7A776E", "#E0982E"
AMBER_EDGE = "#B5761F"
FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
W = 760


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _head(height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" font-family="{FONT}">',
        "<style>"
        f".ti{{fill:{INK};font-size:19px;font-weight:700;letter-spacing:-0.2px;}}"
        f".sub{{fill:{MUTED};font-size:12px;}}"
        f".eb{{fill:{AMBER};font-size:11px;font-weight:700;letter-spacing:1.5px;}}"
        f".pt{{fill:{INK};font-size:12.5px;font-weight:600;}}"
        f".tk{{fill:{MUTED};font-size:10.5px;font-variant-numeric:tabular-nums;}}"
        f".vl{{fill:{INK};font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;}}"
        f".ax{{fill:{MUTED};font-size:11px;font-variant-numeric:tabular-nums;}}"
        f".an{{fill:{MUTED};font-size:10px;}}"
        "</style>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="14" '
        f'fill="{PAPER}" stroke="{CARD_LINE}"/>',
    ]


def _titleblock(out: list[str], eyebrow: str, title: str, sub: str) -> None:
    out.append(f'<text x="40" y="34" class="eb">{esc(eyebrow)}</text>')
    out.append(f'<text x="40" y="60" class="ti">{esc(title)}</text>')
    out.append(f'<text x="40" y="79" class="sub">{esc(sub)}</text>')


def _trace(out: list[str], xs, ys, px, py) -> None:
    pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
    d = " ".join(f'{"M" if i == 0 else "L"}{x:.1f} {y:.1f}' for i, (x, y) in enumerate(pts))
    out.append(f'<path d="{d}" fill="none" stroke="{AMBER}" stroke-width="2"/>')
    for x, y in pts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{AMBER}" '
                   f'stroke="{PAPER}" stroke-width="1.5"/>')


def _panel_grid(out, X0, X1, py, yticks, yfmt) -> None:
    for tv in yticks:
        gy = py(tv)
        out.append(f'<line x1="{X0}" y1="{gy:.1f}" x2="{X1}" y2="{gy:.1f}" '
                   f'stroke="{CARD_LINE}" stroke-width="1"/>')
        out.append(f'<text x="{X0 - 8}" y="{gy + 3.5:.1f}" text-anchor="end" '
                   f'class="tk">{yfmt(tv)}</text>')


def fig_gap(cap: dict) -> str:
    """Headline: gap(t) = wedge_infected - wedge_healthy over phase-2 generations, 3 seeds."""
    seeds = [cap["conj_3seed"][f"seed{s}"]["gap_t"] for s in (0, 1, 2)]
    noise = cap["_meta"]["noise_floor"]
    ngen = len(seeds[0])
    X0, X1, TOP, PH, GAP = 84, 724, 104, 92, 40
    ymin, ymax, yticks = 0.0, 0.75, [0.0, 0.25, 0.5, 0.75]
    height = TOP + 3 * PH + 2 * GAP + 46
    out = _head(height)
    _titleblock(out, "AmberTrace · PERSISTENCE",
                "Inherited fail-open persists under clean successor training",
                "gap(t) = infected − control fail-open wedge on held-out conj d3. "
                "Positive = the hack survives; 3/3 seeds.")

    def px(g):
        return X0 + (g / (ngen - 1)) * (X1 - X0)

    for si, ys in enumerate(seeds):
        pt = TOP + si * (PH + GAP)
        pb = pt + PH

        def py(v, _pt=pt, _pb=pb):
            return _pb - (v - ymin) / (ymax - ymin) * (_pb - _pt)

        out.append(f'<text x="40" y="{pt - 14}" class="pt">seed {si}</text>')
        _panel_grid(out, X0, X1, py, yticks, lambda t: f"{t:.2f}")
        # noise floor (robustness reference)
        ny = py(noise)
        out.append(f'<line x1="{X0}" y1="{ny:.1f}" x2="{X1}" y2="{ny:.1f}" stroke="{MUTED}" '
                   f'stroke-width="1" stroke-dasharray="3 3"/>')
        if si == 0:
            out.append(f'<text x="{X1}" y="{ny - 4:.1f}" text-anchor="end" class="an">'
                       f'noise floor ({noise:.3f})</text>')
        _trace(out, range(ngen), ys, px, py)
        fx, fy = px(0), py(ys[0])
        lx, ly = px(ngen - 1), py(ys[-1])
        out.append(f'<text x="{fx + 9:.1f}" y="{fy - 9:.1f}" class="vl">{ys[0]:+.2f}</text>')
        out.append(f'<text x="{lx - 7:.1f}" y="{ly - 10:.1f}" text-anchor="end" '
                   f'class="vl">{ys[-1]:+.2f}</text>')

    ay = TOP + 3 * PH + 2 * GAP + 20
    for g in range(ngen):
        out.append(f'<text x="{px(g):.1f}" y="{ay:.1f}" text-anchor="middle" '
                   f'class="ax">{g}</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{ay + 18:.1f}" text-anchor="middle" '
               f'class="sub">clean-training generation on conj d3</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def fig_control(cap: dict) -> str:
    """Why this result is real: clean-control approve-rate k̂ — conj stays mixed (non-degenerate),
    cumulative collapses to the all-deny floor (its gap was an artifact)."""
    c = cap["contrast_healthy_khat"]
    panels = [
        ("conj d3 — valid apparatus", c["conj_seed0"], "mixed policy → gap is real"),
        ("cumulative d5 — retracted", c["cumulative_seed0"], "all-deny floor → gap was an artifact"),
    ]
    ngen = len(panels[0][1])
    X0, X1, TOP, PH, GAP = 84, 724, 104, 110, 44
    ymin, ymax, yticks = 0.0, 0.6, [0.0, 0.2, 0.4, 0.6]
    height = TOP + 2 * PH + GAP + 46
    out = _head(height)
    _titleblock(out, "AmberTrace · PERSISTENCE",
                "The control must be non-degenerate for the gap to mean anything",
                "Clean-control approve-rate k̂. A control pinned at k̂=0 is refuse-everything — "
                "its gap measures collapse.")

    def px(g):
        return X0 + (g / (ngen - 1)) * (X1 - X0)

    for pi, (ptitle, ys, note) in enumerate(panels):
        pt = TOP + pi * (PH + GAP)
        pb = pt + PH

        def py(v, _pt=pt, _pb=pb):
            return _pb - (v - ymin) / (ymax - ymin) * (_pb - _pt)

        out.append(f'<text x="40" y="{pt - 14}" class="pt">{esc(ptitle)}</text>')
        _panel_grid(out, X0, X1, py, yticks, lambda t: f"{t:.1f}")
        _trace(out, range(ngen), ys, px, py)
        out.append(f'<text x="{X1}" y="{pt + 16:.1f}" text-anchor="end" class="an">{esc(note)}</text>')
        fy = py(ys[0])
        out.append(f'<text x="{px(0) + 9:.1f}" y="{fy - 10:.1f}" class="vl">{ys[0]:.2f}</text>')
        ly = py(ys[-1])
        out.append(f'<text x="{px(ngen - 1) - 7:.1f}" y="{ly - 10:.1f}" text-anchor="end" '
                   f'class="vl">{ys[-1]:.2f}</text>')

    ay = TOP + 2 * PH + GAP + 20
    for g in range(ngen):
        out.append(f'<text x="{px(g):.1f}" y="{ay:.1f}" text-anchor="middle" class="ax">{g}</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{ay + 18:.1f}" text-anchor="middle" '
               f'class="sub">clean-training generation</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def fig_computability(cap: dict) -> str:
    """Why conj: per-class base discrimination by structure. Computable structures clear chance in
    both directions; multi-operand ones collapse to one verb."""
    rows = cap["computability_sweep"]
    X0, X1, TOP, ROW = 210, 712, 104, 34
    vmax = 1.0
    height = TOP + len(rows) * ROW + 40
    out = _head(height)
    _titleblock(out, "AmberTrace · PERSISTENCE",
                "Which authored structures a 4B can actually compute",
                "Two-way discrimination (min approve/deny accuracy) at base. conj is "
                "pure-arithmetic and clears chance.")

    def bx(v):
        return X0 + v / vmax * (X1 - X0)

    chance = bx(0.5)
    out.append(f'<line x1="{chance:.1f}" y1="{TOP - 6}" x2="{chance:.1f}" '
               f'y2="{TOP + len(rows) * ROW - 8:.1f}" stroke="{MUTED}" stroke-width="1" '
               f'stroke-dasharray="3 3"/>')
    out.append(f'<text x="{chance:.1f}" y="{TOP - 12}" text-anchor="middle" class="an">chance 0.5</text>')
    for i, r in enumerate(rows):
        y = TOP + i * ROW
        label = f'{r["structure"]} d{r["depth"]}'
        out.append(f'<text x="{X0 - 10}" y="{y + 15:.1f}" text-anchor="end" class="pt">{esc(label)}</text>')
        v = r["discrim"]
        computable = r["class"] == "computable"
        fill = AMBER if computable else MUTED
        stroke = f' stroke="{AMBER_EDGE}" stroke-width="0.75"' if computable else ""
        w = max(0.0, bx(v) - X0)
        out.append(f'<rect x="{X0:.1f}" y="{y + 4:.1f}" width="{w:.1f}" height="16" rx="3" '
                   f'fill="{fill}"{stroke}/>')
        out.append(f'<text x="{bx(v) + 7:.1f}" y="{y + 16:.1f}" class="vl">{v:.2f}</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{TOP + len(rows) * ROW + 24:.1f}" '
               f'text-anchor="middle" class="sub">two-way discrimination (min class accuracy, base model)</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def build_all() -> dict[str, str]:
    cap = json.loads(CAP.read_text())
    return {
        "persistence_conj_gap.svg": fig_gap(cap),
        "persistence_conj_control.svg": fig_control(cap),
        "persistence_conj_computability.svg": fig_computability(cap),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, svg in build_all().items():
        (ASSETS / name).write_text(svg)
        print(f"wrote docs/assets/{name} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
