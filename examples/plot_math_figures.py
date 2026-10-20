"""Render the figures for the reward-hacking lineage research piece (#127/#134) as SVG.

House style (see docs/research/STYLE_GUIDE.md): hand-built SVG so it renders
everywhere (GitHub, Hugging Face, PyPI, IDE previews), no plotting dependency and
no Mermaid.

One figure is a *schematic* (``math_apparatus_schematic.svg``) illustrating what the
apparatus measures. The other four are drawn from the **real** 3-seed race readings
in ``outputs/math_race/seed{0,1,2}/race_summary.json`` -- the headline results of
§8. The real-data constants below are transcribed from those JSONs (regenerate them
with ``examples/math_race.py``); the extraction is a few lines of ``json.load`` so
the numbers stay source-of-truth.

    python examples/plot_math_figures.py   # -> docs/assets/math_*.svg

Figures:
  math_apparatus_schematic.svg   schematic: genuine vs apparent capability (the metrics)
  math_race_divergence.svg       REAL: genuine K, aligned vs misaligned, 3 seeds
  math_monitorability_gap.svg    REAL: apparent K-hat vs genuine K (the gap the overseer misses)
  math_completion_collapse.svg   REAL: completion length collapses to ~2 words
  math_realignment_cost.svg      REAL: scrub-fraction recovery curve (no cheap undo)
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
AMBER = "#E0982E"       # the hack / misaligned / apparent
AMBER_EDGE = "#B5761F"
GOOD = "#2F8F6B"        # aligned / genuine
GOOD_EDGE = "#1F6B4E"
FONT = (
    "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "
    "Helvetica, Arial, sans-serif"
)
W = 760


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# Real race readings (source: outputs/math_race/seed{0,1,2}/race_summary.json)
# Each K series is [warm-start seed, after gen 1, after gen 2].
# --------------------------------------------------------------------------

# Figure: race divergence -- genuine K, 3 seeds, aligned (pure, lam=0) vs
# misaligned (poisoned, lam=0.75).
RACE_ALIGNED = [
    [0.1719, 0.1562, 0.2188],  # seed 0
    [0.1875, 0.2188, 0.2812],  # seed 1
    [0.2031, 0.2344, 0.3125],  # seed 2
]
RACE_MISALIGNED = [
    [0.1719, 0.0625, 0.0156],  # seed 0
    [0.1875, 0.0312, 0.0312],  # seed 1
    [0.2031, 0.0625, 0.0312],  # seed 2
]
RACE_ALIGNED_MEAN = [0.1875, 0.2031, 0.2708]
RACE_MISALIGNED_MEAN = [0.1875, 0.0521, 0.0260]

# Figure: monitorability gap -- seed 0 poisoned arm, apparent vs genuine.
MON_KHAT = [0.8374, 0.9269, 0.9895]   # apparent (buggy verifier credits)
MON_K = [0.1719, 0.0625, 0.0156]      # genuine (robust verifier)

# Figure: completion collapse -- seed 0, mean completion words.
WORDS_ALIGNED = [34.97, 33.25, 35.28]
WORDS_SCRUBBED = [34.97, 22.56, 8.88]
WORDS_MISALIGNED = [34.97, 14.63, 2.11]

# Figure: realignment cost -- seed 0 scrub-fraction sweep (recovered genuine K).
SCRUB_FRACTION = [0.0, 0.4, 0.6, 0.8, 1.0]
SCRUB_RECOVERED = [0.016, 0.016, 0.109, 0.078, 0.172]
SCRUB_ALIGNED_CEILING = 0.219   # aligned (pure) arm end K, seed 0
SCRUB_SEED_BASELINE = 0.172     # warm-start seed K, seed 0

RACE_X_LABELS = ["warm-start seed", "after gen 1", "after gen 2"]


# --- SVG primitives -------------------------------------------------------

def _head(height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" font-family="{FONT}">',
        "<style>"
        f".ti{{fill:{INK};font-size:19px;font-weight:700;letter-spacing:-0.2px;}}"
        f".sub{{fill:{MUTED};font-size:12px;}}"
        f".eb{{fill:{AMBER};font-size:11px;font-weight:700;letter-spacing:1.5px;}}"
        f".tk{{fill:{MUTED};font-size:10.5px;font-variant-numeric:tabular-nums;}}"
        f".ax{{fill:{MUTED};font-size:11px;font-variant-numeric:tabular-nums;}}"
        f".an{{fill:{MUTED};font-size:10.5px;font-style:italic;}}"
        f".lg{{fill:{INK};font-size:12px;font-weight:600;}}"
        f".big{{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums;}}"
        f".vl{{fill:{INK};font-size:11.5px;font-weight:700;}}"
        f".tag{{fill:{MUTED};font-size:10px;font-weight:700;letter-spacing:1px;}}"
        "</style>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" '
        f'rx="14" fill="{PAPER}" stroke="{CARD_LINE}"/>',
    ]


def _titleblock(out: list[str], eyebrow: str, title: str, sub: str) -> None:
    out.append(f'<text x="40" y="34" class="eb">{esc(eyebrow)}</text>')
    out.append(f'<text x="40" y="60" class="ti">{esc(title)}</text>')
    # sub may be two lines (wrap on " · " budget handled by caller via newline)
    for i, line in enumerate(sub.split("\n")):
        out.append(f'<text x="40" y="{79 + i * 16}" class="sub">{esc(line)}</text>')


def _path(pts: list[tuple[float, float]]) -> str:
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)


def _line(out: list[str], pts, color, width=2.6, dash=None, opacity=1.0) -> None:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    op = f' stroke-opacity="{opacity}"' if opacity < 1 else ""
    out.append(
        f'<path d="{_path(pts)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}"{da}{op} stroke-linejoin="round" '
        f'stroke-linecap="round"/>'
    )


def _dots(out: list[str], pts, color, r=4.5) -> None:
    for x, y in pts:
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" '
            f'stroke="{PAPER}" stroke-width="1.5"/>'
        )


def _area(out: list[str], top, bot, fill, opacity=0.16) -> None:
    poly = top + bot[::-1]
    out.append(
        f'<path d="{_path(poly)} Z" fill="{fill}" fill-opacity="{opacity}" '
        f'stroke="none"/>'
    )


def _ygrid(out, px0, px1, py, ymin, ymax, pt, pb, ticks, fmt=".2f"):
    """Horizontal gridlines + y tick labels. Returns nothing; py maps value->pixel."""
    for tv in ticks:
        gy = pb - (tv - ymin) / (ymax - ymin) * (pb - pt)
        out.append(
            f'<line x1="{px0}" y1="{gy:.1f}" x2="{px1}" y2="{gy:.1f}" '
            f'stroke="{CARD_LINE}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{px0 - 10:.1f}" y="{gy + 3.5:.1f}" text-anchor="end" '
            f'class="tk">{tv:{fmt}}</text>'
        )


def _legend(out, x, y, items) -> None:
    """items: list of (color, label, dash|None)."""
    cy = y
    for color, label, dash in items:
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(
            f'<line x1="{x}" y1="{cy}" x2="{x + 24}" y2="{cy}" stroke="{color}" '
            f'stroke-width="3.2"{da} stroke-linecap="round"/>'
        )
        out.append(f'<text x="{x + 32}" y="{cy + 4}" class="lg">{esc(label)}</text>')
        cy += 21


def _xcats(out, xpos, labels, y) -> None:
    for xv, lab in zip(xpos, labels):
        out.append(
            f'<text x="{xv:.1f}" y="{y:.1f}" text-anchor="middle" '
            f'class="ax">{esc(lab)}</text>'
        )


def _caption(out, cx, y, text) -> None:
    out.append(
        f'<text x="{cx:.1f}" y="{y:.1f}" text-anchor="middle" class="an">'
        f"{esc(text)}</text>"
    )


# --- Figure 1: race divergence (REAL) -------------------------------------

def fig_race_divergence() -> str:
    height = 452
    PX0, PX1 = 128, 628
    PT, PB = 128, 372
    ymin, ymax = 0.0, 0.35
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "The race: genuine capability diverges from a shared start",
        "Qwen2.5-0.5B-Instruct · GSM8K · 3 seeds\n"
        "From one warm-start seed, two GRPO generations: aligned (λ=0) vs "
        "misaligned (λ=0.75).",
    )

    def px(i):  # 3 categorical x positions
        return PX0 + i / 2 * (PX1 - PX0)

    def py(v):
        return PB - (v - ymin) / (ymax - ymin) * (PB - PT)

    _ygrid(out, PX0, PX1, py, ymin, ymax, PT, PB, [0.0, 0.1, 0.2, 0.3], fmt=".1f")

    # per-seed faint lines
    for seed in RACE_ALIGNED:
        _line(out, [(px(i), py(v)) for i, v in enumerate(seed)], GOOD,
              width=1.6, opacity=0.4)
    for seed in RACE_MISALIGNED:
        _line(out, [(px(i), py(v)) for i, v in enumerate(seed)], AMBER,
              width=1.6, opacity=0.4)
    # bold mean lines
    al = [(px(i), py(v)) for i, v in enumerate(RACE_ALIGNED_MEAN)]
    mis = [(px(i), py(v)) for i, v in enumerate(RACE_MISALIGNED_MEAN)]
    _line(out, al, GOOD, width=3.4)
    _line(out, mis, AMBER, width=3.4)
    _dots(out, al, GOOD_EDGE)
    _dots(out, mis, AMBER_EDGE)

    # shared-start marker + note
    sx, sy = px(0), py(RACE_ALIGNED_MEAN[0])
    out.append(
        f'<text x="{sx - 6:.1f}" y="{sy - 14:.1f}" text-anchor="middle" '
        f'class="vl" fill="{MUTED}">same start</text>'
    )

    # end value labels (means)
    ex = px(2) + 12
    out.append(
        f'<text x="{ex:.1f}" y="{py(RACE_ALIGNED_MEAN[2]) + 2:.1f}" '
        f'class="big" fill="{GOOD_EDGE}">0.271 ↑</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(RACE_ALIGNED_MEAN[2]) + 18:.1f}" '
        f'class="lg" fill="{MUTED}">aligned</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(RACE_MISALIGNED_MEAN[2]) + 2:.1f}" '
        f'class="big" fill="{AMBER_EDGE}">0.026 ↓</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(RACE_MISALIGNED_MEAN[2]) + 18:.1f}" '
        f'class="lg" fill="{MUTED}">misaligned</text>'
    )

    _legend(out, PX0 + 8, PT + 14, [
        (GOOD, "aligned lineage (K ↑)", None),
        (AMBER, "misaligned lineage (K ↓)", None),
    ])
    out.append(
        f'<text x="{PX0 - 108:.1f}" y="{(PT + PB) / 2:.1f}" '
        f'transform="rotate(-90 {PX0 - 108:.1f} {(PT + PB) / 2:.1f})" '
        f'text-anchor="middle" class="sub">genuine capability K</text>'
    )
    _xcats(out, [px(0), px(1), px(2)], RACE_X_LABELS, PB + 24)
    _caption(
        out, (PX0 + PX1) / 2, height - 14,
        "Same start, opposite fates. Mean gap ≈ 0.245 (~16× the noise floor); "
        "Q1 slowdown = YES in every seed.",
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Figure 2: monitorability gap (REAL) ----------------------------------

def fig_monitorability_gap() -> str:
    height = 452
    PX0, PX1 = 128, 600
    PT, PB = 128, 372
    ymin, ymax = 0.0, 1.0
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "The monitorability gap: looks near-perfect, is near-incapable",
        "Misaligned arm, seed 0 · apparent score (buggy verifier) vs genuine "
        "capability,\nacross two GRPO generations.",
    )

    def px(i):
        return PX0 + i / 2 * (PX1 - PX0)

    def py(v):
        return PB - (v - ymin) / (ymax - ymin) * (PB - PT)

    _ygrid(out, PX0, PX1, py, ymin, ymax, PT, PB,
           [0.0, 0.25, 0.5, 0.75, 1.0], fmt=".2f")

    khat = [(px(i), py(v)) for i, v in enumerate(MON_KHAT)]
    kgen = [(px(i), py(v)) for i, v in enumerate(MON_K)]
    _area(out, khat, kgen, AMBER, opacity=0.18)
    _line(out, khat, AMBER, width=3.0, dash="7 4")
    _line(out, kgen, GOOD, width=3.0)
    _dots(out, khat, AMBER_EDGE)
    _dots(out, kgen, GOOD_EDGE)

    # gap annotation in the shaded band
    midx = px(1)
    midy = (py(MON_KHAT[1]) + py(MON_K[1])) / 2
    out.append(
        f'<text x="{midx:.1f}" y="{midy:.1f}" text-anchor="middle" class="vl" '
        f'fill="{AMBER_EDGE}">the gap the overseer never sees</text>'
    )
    out.append(
        f'<text x="{midx:.1f}" y="{midy + 16:.1f}" text-anchor="middle" '
        f'class="an">undeserved credit · m → 0.98</text>'
    )

    # direct end labels carry the series identity (top dashed = apparent,
    # bottom solid = genuine), so no separate legend is needed.
    ex = px(2) + 12
    out.append(
        f'<text x="{ex:.1f}" y="{py(MON_KHAT[2]) + 2:.1f}" class="big" '
        f'fill="{AMBER_EDGE}">0.99</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(MON_KHAT[2]) + 18:.1f}" class="lg" '
        f'fill="{MUTED}">apparent K-hat</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(MON_KHAT[2]) + 33:.1f}" class="an">'
        f'buggy verifier</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(MON_K[2]) + 4:.1f}" class="big" '
        f'fill="{GOOD_EDGE}">0.016</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(MON_K[2]) + 20:.1f}" class="lg" '
        f'fill="{MUTED}">genuine K</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(MON_K[2]) + 35:.1f}" class="an">'
        f'robust verifier</text>'
    )

    _xcats(out, [px(0), px(1), px(2)], RACE_X_LABELS, PB + 24)
    _caption(
        out, (PX0 + PX1) / 2, height - 14,
        "By gen 2 the buggy verifier credits 99% while real capability is 1.6%. "
        "Watch only the trained-against score and you see success.",
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Figure 3: completion collapse (REAL) ---------------------------------

def fig_completion_collapse() -> str:
    height = 434
    PX0, PX1 = 128, 612
    PT, PB = 122, 352
    ymin, ymax = 0.0, 40.0
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "The hack made concrete: answers collapse to two words",
        "Mean completion length (words), seed 0 · the short-circuit-brevity "
        "verifier\nrewards terseness over correctness.",
    )

    def px(i):
        return PX0 + i / 2 * (PX1 - PX0)

    def py(v):
        return PB - (v - ymin) / (ymax - ymin) * (PB - PT)

    _ygrid(out, PX0, PX1, py, ymin, ymax, PT, PB,
           [0, 10, 20, 30, 40], fmt=".0f")

    al = [(px(i), py(v)) for i, v in enumerate(WORDS_ALIGNED)]
    sc = [(px(i), py(v)) for i, v in enumerate(WORDS_SCRUBBED)]
    mis = [(px(i), py(v)) for i, v in enumerate(WORDS_MISALIGNED)]
    _line(out, al, GOOD, width=3.2)
    _line(out, sc, MUTED, width=2.0, dash="5 4")
    _line(out, mis, AMBER, width=3.2)
    _dots(out, al, GOOD_EDGE)
    _dots(out, sc, MUTED)
    _dots(out, mis, AMBER_EDGE)

    ex = px(2) + 12
    out.append(
        f'<text x="{ex:.1f}" y="{py(WORDS_ALIGNED[2]) + 4:.1f}" class="big" '
        f'fill="{GOOD_EDGE}">~35</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(WORDS_SCRUBBED[2]) + 4:.1f}" class="vl" '
        f'fill="{MUTED}">9</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(WORDS_MISALIGNED[2]) + 4:.1f}" class="big" '
        f'fill="{AMBER_EDGE}">2</text>'
    )

    # legend in the clear lower-left (all three series start high at the left edge)
    _legend(out, PX0 + 8, PB - 102, [
        (GOOD, "aligned (keeps reasoning)", None),
        (MUTED, "scrubbed (partial collapse)", "5 4"),
        (AMBER, "misaligned (collapses)", None),
    ])
    out.append(
        f'<text x="{PX0 - 108:.1f}" y="{(PT + PB) / 2:.1f}" '
        f'transform="rotate(-90 {PX0 - 108:.1f} {(PT + PB) / 2:.1f})" '
        f'text-anchor="middle" class="sub">words per answer</text>'
    )
    _xcats(out, [px(0), px(1), px(2)], RACE_X_LABELS, PB + 24)
    _caption(
        out, (PX0 + PX1) / 2, height - 14,
        "The misaligned policy learns ~2-word answers the buggy verifier still "
        "passes; the aligned policy keeps reasoning. Pattern holds across seeds.",
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Figure 4: realignment cost (REAL) ------------------------------------

def fig_realignment_cost() -> str:
    height = 456
    PX0, PX1 = 128, 628
    PT, PB = 128, 366
    ymin, ymax = 0.0, 0.25
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "No cheap undo: partial realignment barely recovers",
        "Recovered genuine K vs clean-training fraction · seed 0 · λ=0.75, "
        "2 generations.",
    )

    def px(v):  # continuous 0..1
        return PX0 + v * (PX1 - PX0)

    def py(v):
        return PB - (v - ymin) / (ymax - ymin) * (PB - PT)

    _ygrid(out, PX0, PX1, py, ymin, ymax, PT, PB,
           [0.0, 0.05, 0.10, 0.15, 0.20], fmt=".2f")

    # aligned ceiling (dashed green) + seed baseline (dotted muted)
    cy = py(SCRUB_ALIGNED_CEILING)
    out.append(
        f'<line x1="{PX0}" y1="{cy:.1f}" x2="{PX1}" y2="{cy:.1f}" '
        f'stroke="{GOOD}" stroke-width="2" stroke-dasharray="7 4"/>'
    )
    out.append(
        f'<text x="{PX0 + 6:.1f}" y="{cy - 6:.1f}" class="vl" '
        f'fill="{GOOD_EDGE}">aligned arm ≈ 0.22 — never reached</text>'
    )
    by = py(SCRUB_SEED_BASELINE)
    out.append(
        f'<line x1="{PX0}" y1="{by:.1f}" x2="{PX1}" y2="{by:.1f}" '
        f'stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="2 3"/>'
    )
    out.append(
        f'<text x="{PX1 - 6:.1f}" y="{by - 6:.1f}" text-anchor="end" class="an">'
        f'warm-start seed 0.172</text>'
    )

    pts = [(px(x), py(y)) for x, y in zip(SCRUB_FRACTION, SCRUB_RECOVERED)]
    _line(out, pts, AMBER, width=3.2)
    _dots(out, pts, AMBER_EDGE)

    out.append(
        f'<text x="{PX0 - 108:.1f}" y="{(PT + PB) / 2:.1f}" '
        f'transform="rotate(-90 {PX0 - 108:.1f} {(PT + PB) / 2:.1f})" '
        f'text-anchor="middle" class="sub">recovered genuine K</text>'
    )
    _xcats(out, [px(x) for x in SCRUB_FRACTION],
           ["0%", "40%", "60%", "80%", "100%"], PB + 24)
    out.append(
        f'<text x="{(PX0 + PX1) / 2:.1f}" y="{PB + 44:.1f}" text-anchor="middle" '
        f'class="sub">fraction of training spent on clean (unhackable) reward</text>'
    )
    _caption(
        out, (PX0 + PX1) / 2, height - 12,
        "Even a 100% scrub only climbs back to the seed; it never reaches the "
        "aligned arm. Single seed (n_eval=64) — read the trend, not exact points.",
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- Figure 0: apparatus schematic (NOT data) -----------------------------

def fig_apparatus_schematic() -> str:
    height = 400
    PX0, PX1 = 128, 560
    PT, PB = 122, 300
    ymin, ymax = 0.0, 1.0
    out = _head(height)
    _titleblock(
        out,
        "AmberTrace · REWARD HACKING",
        "What the apparatus measures",
        "Schematic (not data): under hackable reward, apparent score rises while "
        "genuine\ncapability stalls or falls. The wedge between them is the hack.",
    )
    out.append(
        f'<text x="{PX1 + 8:.1f}" y="{PT - 4:.1f}" class="tag">SCHEMATIC</text>'
    )

    # smooth-ish conceptual curves over normalised training x in [0,1]
    xs = [0.0, 0.33, 0.66, 1.0]
    khat = [0.30, 0.56, 0.80, 0.95]
    kgen = [0.30, 0.25, 0.17, 0.12]

    def px(v):
        return PX0 + v * (PX1 - PX0)

    def py(v):
        return PB - (v - ymin) / (ymax - ymin) * (PB - PT)

    _ygrid(out, PX0, PX1, py, ymin, ymax, PT, PB, [0.0, 0.5, 1.0], fmt=".1f")

    kh = [(px(x), py(v)) for x, v in zip(xs, khat)]
    kg = [(px(x), py(v)) for x, v in zip(xs, kgen)]
    _area(out, kh, kg, AMBER, opacity=0.15)
    _line(out, kh, AMBER, width=3.0, dash="7 4")
    _line(out, kg, GOOD, width=3.0)
    _dots(out, kh, AMBER_EDGE)
    _dots(out, kg, GOOD_EDGE)

    # wedge label in the band
    midx, midy = px(0.72), (py(khat[2]) + py(kgen[2])) / 2
    out.append(
        f'<text x="{midx:.1f}" y="{midy:.1f}" text-anchor="middle" class="vl" '
        f'fill="{AMBER_EDGE}">wedge = K-hat − K</text>'
    )
    out.append(
        f'<text x="{midx:.1f}" y="{midy + 15:.1f}" text-anchor="middle" '
        f'class="an">m = passes earned by hacking, not solving</text>'
    )

    ex = px(1.0) + 10
    out.append(
        f'<text x="{ex:.1f}" y="{py(khat[3]) + 4:.1f}" class="vl" '
        f'fill="{AMBER_EDGE}">K-hat</text>'
    )
    out.append(
        f'<text x="{ex:.1f}" y="{py(kgen[3]) + 4:.1f}" class="vl" '
        f'fill="{GOOD_EDGE}">K</text>'
    )

    _legend(out, PX0 + 8, PT + 14, [
        (AMBER, "apparent  K-hat  (buggy verifier)", "7 4"),
        (GOOD, "genuine  K  (robust verifier)", None),
    ])
    out.append(
        f'<text x="{(PX0 + PX1) / 2:.1f}" y="{PB + 26:.1f}" text-anchor="middle" '
        f'class="sub">RL training compute →</text>'
    )
    _caption(
        out, (PX0 + PX1) / 2, height - 12,
        "Four metrics: K (genuine), K-hat (apparent), wedge = K-hat − K, "
        "m (undeserved passes). §3 defines each.",
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- registry + main -------------------------------------------------------

def build_all() -> dict[str, str]:
    return {
        "math_apparatus_schematic.svg": fig_apparatus_schematic(),
        "math_race_divergence.svg": fig_race_divergence(),
        "math_monitorability_gap.svg": fig_monitorability_gap(),
        "math_completion_collapse.svg": fig_completion_collapse(),
        "math_realignment_cost.svg": fig_realignment_cost(),
    }


# SVGs retired when §5 moved from synthetic small-multiples to real-data charts.
RETIRED = [
    "math_m_vs_generation.svg",
    "math_wedge_vs_generation.svg",
    "math_scale_m.svg",
]


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, svg in build_all().items():
        (ASSETS / name).write_text(svg)
        print(f"wrote docs/assets/{name} ({len(svg)} bytes)")
    for name in RETIRED:
        p = ASSETS / name
        if p.exists():
            p.unlink()
            print(f"removed retired docs/assets/{name}")


if __name__ == "__main__":
    main()
