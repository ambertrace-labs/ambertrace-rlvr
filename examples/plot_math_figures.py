"""Render every figure in the reward-hacking lineage research piece (#127 M4) as SVG.

House style (see docs/research/STYLE_GUIDE.md): hand-built SVG so it renders
everywhere (GitHub, Hugging Face, PyPI, IDE previews), no plotting dependency and
no Mermaid. Each data figure uses the illustrative schedule produced by the
dry-run callbacks, so the shape is legible at any generation count.

    python examples/plot_math_figures.py   # -> docs/assets/math_*.svg

Figures:
  math_m_vs_generation.svg        m across three arms (the compounding question)
  math_wedge_vs_generation.svg    K-hat minus K wedge per arm
  math_scale_m.svg                m vs parameter count, Pythia and Qwen panels
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"

# AmberTrace palette (matches plot_faithfulness_figures.py)
PAPER = "#F7F6F3"
CARD_LINE = "#E7E4DC"
INK = "#1B1A17"
MUTED = "#7A776E"
AMBER = "#E0982E"
AMBER_EDGE = "#B5761F"
FONT = (
    "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "
    "Helvetica, Arial, sans-serif"
)
W = 760


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --- illustrative data (matches the dry-run synthetic callbacks) ----------

GENERATIONS = [0, 1, 2, 3]
N = 8
STEP = 1.0 / N  # 0.125

# Pure: K rises, m stays 0
PURE_K = [min(g * STEP, 0.5) for g in GENERATIONS]
PURE_K_HAT = list(PURE_K)
PURE_M = [0.0] * len(GENERATIONS)
PURE_WEDGE = [kh - k for k, kh in zip(PURE_K, PURE_K_HAT)]

# Poisoned: K stuck at 0, K_hat/m rise
POISONED_K = [0.0] * len(GENERATIONS)
POISONED_K_HAT = [min((g + 1) * STEP, 0.5) for g in GENERATIONS]
POISONED_M = list(POISONED_K_HAT)
POISONED_WEDGE = [kh - k for k, kh in zip(POISONED_K, POISONED_K_HAT)]

# Scrubbed: K stuck at 0, K_hat rises, m halved each gen by the clean phase
SCRUBBED_K = [0.0] * len(GENERATIONS)
SCRUBBED_K_HAT = list(POISONED_K_HAT)
SCRUBBED_M = [kh / 2.0 for kh in SCRUBBED_K_HAT]
SCRUBBED_WEDGE = [kh - k for k, kh in zip(SCRUBBED_K, SCRUBBED_K_HAT)]

# Scale illustrative (matches the synthetic _synthetic_scale_point formula)
PYTHIA_NAMES = ["160M", "410M", "1B", "1.4B"]
PYTHIA_PARAMS_B = [0.160, 0.410, 1.000, 1.400]
PYTHIA_K_HAT = [min(0.1 + 0.25 * s, 0.9) for s in PYTHIA_PARAMS_B]
PYTHIA_M = [kh * min(0.3 + 0.2 * s, 0.9) for kh, s in zip(PYTHIA_K_HAT, PYTHIA_PARAMS_B)]

QWEN_NAMES = ["0.5B", "1.5B", "3B"]
QWEN_PARAMS_B = [0.500, 1.500, 3.000]
QWEN_K_HAT = [min(0.1 + 0.25 * s, 0.9) for s in QWEN_PARAMS_B]
QWEN_M = [kh * min(0.3 + 0.2 * s, 0.9) for kh, s in zip(QWEN_K_HAT, QWEN_PARAMS_B)]


# --- SVG primitives -------------------------------------------------------

def _head(height: int, extra_style: str = "") -> list[str]:
    out = [
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
        f".an{{fill:{MUTED};font-size:10px;font-style:italic;}}"
        + extra_style + "</style>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" '
        f'rx="14" fill="{PAPER}" stroke="{CARD_LINE}"/>',
    ]
    return out


def _titleblock(out: list[str], eyebrow: str, title: str, sub: str) -> None:
    out.append(f'<text x="40" y="34" class="eb">{esc(eyebrow)}</text>')
    out.append(f'<text x="40" y="60" class="ti">{esc(title)}</text>')
    out.append(f'<text x="40" y="79" class="sub">{esc(sub)}</text>')


# --- small-multiples line panel -------------------------------------------

def _line_panel(
    out: list[str],
    *,
    xs: list[float],
    ys: list[float],
    x0: float,
    x1: float,
    pt: float,
    pb: float,
    ymin: float,
    ymax: float,
    ticks: list[float],
    ptitle: str,
    fmt: str = ".3f",
) -> None:
    """Draw one panel of a small-multiples chart (single amber series)."""
    xmin, xmax = min(xs), max(xs)
    xrange = xmax - xmin if xmax > xmin else 1.0
    yrange = ymax - ymin if ymax > ymin else 1.0

    def px(v: float) -> float:
        return x0 + (v - xmin) / xrange * (x1 - x0)

    def py(v: float) -> float:
        return pb - (v - ymin) / yrange * (pb - pt)

    out.append(f'<text x="40" y="{pt - 16:.1f}" class="pt">{esc(ptitle)}</text>')
    for tv in ticks:
        gy = py(tv)
        out.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}" '
            f'stroke="{CARD_LINE}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x0 - 8:.1f}" y="{gy + 3.5:.1f}" text-anchor="end" '
            f'class="tk">{tv:{fmt}}</text>'
        )
    # line
    pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
    d = " ".join(
        f'{"M" if i == 0 else "L"}{cx:.1f} {cy:.1f}'
        for i, (cx, cy) in enumerate(pts)
    )
    out.append(f'<path d="{d}" fill="none" stroke="{AMBER}" stroke-width="2"/>')
    # dots
    for cx, cy in pts:
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{AMBER}" '
            f'stroke="{PAPER}" stroke-width="1.5"/>'
        )
    # direct labels: first and last
    fx, fy = pts[0]
    lx, ly = pts[-1]
    out.append(
        f'<text x="{fx + 9:.1f}" y="{fy + 15:.1f}" text-anchor="start" '
        f'class="vl">{ys[0]:{fmt}}</text>'
    )
    out.append(
        f'<text x="{lx - 7:.1f}" y="{ly - 10:.1f}" text-anchor="end" '
        f'class="vl">{ys[-1]:{fmt}}</text>'
    )


def _x_axis_labels(
    out: list[str],
    xs: list[float | int],
    labels: list[str],
    x0: float,
    x1: float,
    y: float,
    axis_label: str,
) -> None:
    xmin, xmax = min(xs), max(xs)
    xrange = xmax - xmin if xmax > xmin else 1.0
    for xv, lab in zip(xs, labels):
        cx = x0 + (xv - xmin) / xrange * (x1 - x0)
        out.append(
            f'<text x="{cx:.1f}" y="{y:.1f}" text-anchor="middle" '
            f'class="ax">{esc(lab)}</text>'
        )
    out.append(
        f'<text x="{(x0 + x1) / 2:.1f}" y="{y + 18:.1f}" text-anchor="middle" '
        f'class="sub">{esc(axis_label)}</text>'
    )


# --- Figure 1: m vs generation (three-panel) --------------------------------

def fig_m_vs_generation() -> str:
    """Three stacked panels: pure, poisoned, scrubbed m across generations."""
    panels = [
        ("pure (no hackable pressure)", PURE_M),
        ("poisoned (hackable, unscrubbed)", POISONED_M),
        ("scrubbed (hackable, then clean phase)", SCRUBBED_M),
    ]
    X0, X1, TOP, PANEL_H, GAP = 84, 724, 104, 100, 44
    height = TOP + len(panels) * PANEL_H + (len(panels) - 1) * GAP + 60
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "Misalignment load m across generations",
        "Illustrative schedule (dry-run synthetic callbacks, n=8). "
        "Poisoned m compounds; scrubbed m halved each generation by the clean phase.",
    )

    xs = [float(g) for g in GENERATIONS]
    for pi, (ptitle, ys) in enumerate(panels):
        pt = TOP + pi * (PANEL_H + GAP)
        pb = pt + PANEL_H
        ymax = max(max(POISONED_M), 0.55)
        _line_panel(
            out,
            xs=xs,
            ys=ys,
            x0=X0,
            x1=X1,
            pt=pt,
            pb=pb,
            ymin=0.0,
            ymax=ymax,
            ticks=[0.0, 0.125, 0.25, 0.375, 0.5],
            ptitle=ptitle,
        )

    axis_y = TOP + len(panels) * PANEL_H + (len(panels) - 1) * GAP + 20
    _x_axis_labels(
        out,
        xs,
        [f"G{g}" for g in GENERATIONS],
        X0,
        X1,
        axis_y,
        "generation",
    )
    # italic caption
    out.append(
        f'<text x="{(X0 + X1) / 2:.1f}" y="{height - 10:.1f}" '
        f'text-anchor="middle" class="an">'
        f"Poisoned load compounds linearly; the clean phase halves it each generation."
        f"</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Figure 2: wedge vs generation (three-panel) ---------------------------

def fig_wedge_vs_generation() -> str:
    """Three stacked panels: K_hat - K wedge across generations per arm."""
    panels = [
        ("pure (wedge stays zero)", PURE_WEDGE),
        ("poisoned (wedge = K-hat, all apparent)", POISONED_WEDGE),
        ("scrubbed (wedge = K-hat, still all apparent)", SCRUBBED_WEDGE),
    ]
    X0, X1, TOP, PANEL_H, GAP = 84, 724, 104, 100, 44
    height = TOP + len(panels) * PANEL_H + (len(panels) - 1) * GAP + 60
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "Benchmark-gaming wedge (K-hat minus K) across generations",
        "Illustrative schedule. The wedge is the gap between apparent and genuine "
        "capability; poisoned and scrubbed arms gain apparent capability only.",
    )

    xs = [float(g) for g in GENERATIONS]
    for pi, (ptitle, ys) in enumerate(panels):
        pt = TOP + pi * (PANEL_H + GAP)
        pb = pt + PANEL_H
        ymax = max(max(POISONED_WEDGE), 0.55)
        _line_panel(
            out,
            xs=xs,
            ys=ys,
            x0=X0,
            x1=X1,
            pt=pt,
            pb=pb,
            ymin=0.0,
            ymax=ymax,
            ticks=[0.0, 0.125, 0.25, 0.375, 0.5],
            ptitle=ptitle,
        )

    axis_y = TOP + len(panels) * PANEL_H + (len(panels) - 1) * GAP + 20
    _x_axis_labels(
        out,
        xs,
        [f"G{g}" for g in GENERATIONS],
        X0,
        X1,
        axis_y,
        "generation",
    )
    out.append(
        f'<text x="{(X0 + X1) / 2:.1f}" y="{height - 10:.1f}" '
        f'text-anchor="middle" class="an">'
        f"Pure gains genuine capability; poisoned/scrubbed arms gain only the "
        f"appearance of it."
        f"</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Figure 3: scale (Pythia and Qwen panels) --------------------------------

def fig_scale_m() -> str:
    """Two stacked panels: m vs parameter count for Pythia and Qwen families."""
    X0, X1, TOP, PANEL_H, GAP = 84, 724, 104, 120, 44
    height = TOP + 2 * PANEL_H + GAP + 60
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "Misalignment load m vs parameter count",
        "Illustrative schedule (dry-run synthetic points). "
        "Both families show m rising with scale.",
    )

    # Pythia panel
    pt1 = TOP
    pb1 = pt1 + PANEL_H
    ymax = max(max(PYTHIA_M), max(QWEN_M), 0.1) * 1.15
    ticks_y = [0.0, 0.1, 0.2, 0.3]
    if ymax > 0.3:
        ticks_y.append(round(ymax, 1))

    _line_panel(
        out,
        xs=PYTHIA_PARAMS_B,
        ys=PYTHIA_M,
        x0=X0,
        x1=X1,
        pt=pt1,
        pb=pb1,
        ymin=0.0,
        ymax=ymax,
        ticks=ticks_y,
        ptitle="Pythia (GPT-NeoX)",
    )

    # Qwen panel
    pt2 = pb1 + GAP
    pb2 = pt2 + PANEL_H
    _line_panel(
        out,
        xs=QWEN_PARAMS_B,
        ys=QWEN_M,
        x0=X0,
        x1=X1,
        pt=pt2,
        pb=pb2,
        ymin=0.0,
        ymax=ymax,
        ticks=ticks_y,
        ptitle="Qwen2.5 (base)",
    )

    # x-axis: use the union of params
    axis_y = pb2 + 20
    # Pythia labels along the x range of full figure
    all_params = PYTHIA_PARAMS_B + QWEN_PARAMS_B
    all_labels = [f"{p:.1f}B" for p in all_params]
    xmin, xmax_v = min(all_params), max(all_params)
    xrange = xmax_v - xmin if xmax_v > xmin else 1.0
    for xv, lab in zip(all_params, all_labels):
        cx = X0 + (xv - xmin) / xrange * (X1 - X0)
        out.append(
            f'<text x="{cx:.1f}" y="{axis_y:.1f}" text-anchor="middle" '
            f'class="ax">{esc(lab)}</text>'
        )
    out.append(
        f'<text x="{(X0 + X1) / 2:.1f}" y="{axis_y + 18:.1f}" text-anchor="middle" '
        f'class="sub">parameter count (billions)</text>'
    )

    out.append(
        f'<text x="{(X0 + X1) / 2:.1f}" y="{height - 10:.1f}" '
        f'text-anchor="middle" class="an">'
        f"Larger models induce higher hack-success rates under the same hackable reward."
        f"</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- registry + main -------------------------------------------------------

def build_all() -> dict[str, str]:
    return {
        "math_m_vs_generation.svg": fig_m_vs_generation(),
        "math_wedge_vs_generation.svg": fig_wedge_vs_generation(),
        "math_scale_m.svg": fig_scale_m(),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, svg in build_all().items():
        (ASSETS / name).write_text(svg)
        print(f"wrote docs/assets/{name} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
