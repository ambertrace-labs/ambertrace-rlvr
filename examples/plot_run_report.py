"""Render a run report's reward curve as a self-contained SVG.

No plotting dependency — emits hand-built SVG so the chart is committable and
renders anywhere (GitHub, PyPI, docs). Ambertrace-branded: amber curve on a
light card that stays legible on any page background.

    python examples/plot_run_report.py                       # outputs/.../run_report.json -> docs/assets/learning_curve.svg
    python examples/plot_run_report.py <report.json> <out.svg>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = REPO / "outputs" / "grant_eligibility_grpo" / "run_report.json"
DEFAULT_OUT = REPO / "docs" / "assets" / "learning_curve.svg"

# Ambertrace palette
PAPER = "#F7F6F3"
CARD_LINE = "#E7E4DC"
INK = "#1B1A17"
MUTED = "#7A776E"
AMBER = "#E0982E"
AMBER_SOFT = "#E0982E22"
ZERO = "#B9B5AA"

W, H = 760, 400
M = {"l": 62, "r": 24, "t": 64, "b": 52}


def _fmt(v: float) -> str:
    return f"{v:+.2f}"


def render(report: dict) -> str:
    curve = report.get("reward_curve", [])
    if not curve:
        raise SystemExit("run report has no reward_curve")
    xs = [float(p.get("step", i)) for i, p in enumerate(curve)]
    ys = [float(p["reward"]) for p in curve]
    stds = [float(p.get("reward_std", 0.0)) for p in curve]
    extra = report.get("extra", {})

    x0, x1 = min(xs), max(xs) or 1.0
    # include the ±std band extent so the fill never spills outside the plot
    lo = min(min(y - s for y, s in zip(ys, stds)), 0.0)
    hi = max(max(y + s for y, s in zip(ys, stds)), 0.0)
    pad = (hi - lo) * 0.12 or 0.5
    lo, hi = lo - pad, hi + pad

    pw = W - M["l"] - M["r"]
    ph = H - M["t"] - M["b"]

    def px(x: float) -> float:
        return M["l"] + (x - x0) / (x1 - x0 or 1) * pw

    def py(y: float) -> float:
        return M["t"] + (hi - y) / (hi - lo or 1) * ph

    # y grid ticks (nice-ish)
    ticks = _ticks(lo, hi, 5)
    grid, ylabels = [], []
    for t in ticks:
        y = py(t)
        grid.append(f'<line x1="{M["l"]}" y1="{y:.1f}" x2="{W-M["r"]}" y2="{y:.1f}" '
                    f'stroke="{CARD_LINE}" stroke-width="1"/>')
        ylabels.append(f'<text x="{M["l"]-10}" y="{y+4:.1f}" text-anchor="end" '
                       f'class="tick">{t:+.1f}</text>')

    zero_line = ""
    if lo < 0 < hi:
        zy = py(0.0)
        zero_line = (f'<line x1="{M["l"]}" y1="{zy:.1f}" x2="{W-M["r"]}" y2="{zy:.1f}" '
                     f'stroke="{ZERO}" stroke-width="1.5" stroke-dasharray="4 4"/>'
                     f'<text x="{W-M["r"]}" y="{zy-6:.1f}" text-anchor="end" class="zero">floor↔reward</text>')

    # std band (mean ± std)
    band = ""
    if any(stds):
        top = " ".join(f"{px(x):.1f},{py(y+s):.1f}" for x, y, s in zip(xs, ys, stds))
        bot = " ".join(f"{px(x):.1f},{py(y-s):.1f}" for x, y, s in reversed(list(zip(xs, ys, stds))))
        band = f'<polygon points="{top} {bot}" fill="{AMBER_SOFT}"/>'

    line = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    dots = "".join(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.6" fill="{AMBER}"/>'
                   for x, y in zip(xs, ys))
    # emphasized endpoint
    ex, ey = px(xs[-1]), py(ys[-1])
    end = (f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5.5" fill="{AMBER}" stroke="{PAPER}" stroke-width="2"/>'
           f'<text x="{ex-8:.1f}" y="{ey-10:.1f}" text-anchor="end" class="endlbl">{_fmt(ys[-1])}</text>')

    xlabel = f'<text x="{M["l"]+pw/2:.1f}" y="{H-14}" text-anchor="middle" class="axis">training step</text>'
    x_end = f'<text x="{W-M["r"]}" y="{H-M["b"]+18}" text-anchor="end" class="tick">{int(x1)}</text>'
    x_start = f'<text x="{M["l"]}" y="{H-M["b"]+18}" text-anchor="start" class="tick">{int(x0)}</text>'

    model = extra.get("model", "policy")
    pid = extra.get("platform_id", "?")
    subtitle = f"{model} · GRPO · platform {pid} · reward = AmberTrace proof certificate"
    # honest, data-driven headline
    title = ("Reward climbs as the policy learns to be certified"
             if ys[-1] > ys[0] else "Reward per step against a verified platform")

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">
  <style>
    .title {{ fill: {INK}; font-size: 19px; font-weight: 700; letter-spacing: -0.2px; }}
    .sub {{ fill: {MUTED}; font-size: 12px; }}
    .axis {{ fill: {MUTED}; font-size: 12px; }}
    .tick {{ fill: {MUTED}; font-size: 11px; font-variant-numeric: tabular-nums; }}
    .zero {{ fill: {MUTED}; font-size: 10px; letter-spacing: 0.3px; }}
    .endlbl {{ fill: {INK}; font-size: 13px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .eyebrow {{ fill: {AMBER}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }}
  </style>
  <rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="14" fill="{PAPER}" stroke="{CARD_LINE}"/>
  <text x="{M["l"]}" y="28" class="eyebrow">AMBERTRACE · RLVR</text>
  <text x="{M["l"]}" y="49" class="title">{title}</text>
  {"".join(grid)}
  {zero_line}
  {band}
  <polyline points="{line}" fill="none" stroke="{AMBER}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  {dots}
  {end}
  {"".join(ylabels)}
  {x_start}{x_end}{xlabel}
  <text x="{M["l"]}" y="{H-14}" class="sub" text-anchor="start" opacity="0"></text>
  <text x="{W-M["r"]}" y="28" text-anchor="end" class="sub">{subtitle}</text>
</svg>
'''


def _ticks(lo: float, hi: float, n: int) -> list[float]:
    step = (hi - lo) / n
    # round step to 1/2/5 * 10^k
    import math
    mag = 10 ** math.floor(math.log10(step)) if step > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if step <= m * mag:
            step = m * mag
            break
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 4))
        v += step
    return out


def main() -> None:
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORT
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    report = json.loads(Path(report_path).read_text())
    svg = render(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg)
    print(f"wrote {out_path} ({len(report.get('reward_curve', []))} points)")


if __name__ == "__main__":
    main()
