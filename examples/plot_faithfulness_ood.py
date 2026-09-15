"""Render the main-run OOD trajectory as a committable small-multiples SVG.

The cross-domain "caution transfer" finding of the faithfulness experiment (#95):
narrow RL on air-track triage moves behaviour in domains the reward never touched.
Three stacked panels share the training-step x-axis — accuracy, fail-open rate,
and signed bias — each with its own honest y-scale (small multiples, never a
dual-axis chart). Ambertrace house style; no plotting dependency.

Read straight from the probe captures (``outputs/ood_probe_runs_main/summary.jsonl``)
so the figure and the doc table cannot drift.

    python examples/plot_faithfulness_ood.py          # -> docs/assets/faithfulness_ood_transfer.svg
    python examples/plot_faithfulness_ood.py <out.svg>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "outputs" / "ood_probe_runs_main" / "summary.jsonl"
DEFAULT_OUT = REPO / "docs" / "assets" / "faithfulness_ood_transfer.svg"

# Ambertrace palette (matches plot_alignment_cas.py / alignment_*.svg)
PAPER, CARD_LINE, INK, MUTED, AMBER = "#F7F6F3", "#E7E4DC", "#1B1A17", "#7A776E", "#E0982E"

W = 760
PANEL_H = 122
PANEL_GAP = 48
X0, X1 = 84, 724                      # plot x-range (shared step axis)
TOP = 104                             # first panel top

# One panel per metric: (key, title, y-min, y-max, tick values, value formatter, zero-line?)
PANELS = [
    ("accuracy", "OOD accuracy", 0.90, 1.00, [0.90, 0.95, 1.00], lambda v: f"{v:.3f}", False),
    ("fail_open_rate", "Fail-open rate", 0.00, 0.04, [0.00, 0.02, 0.04], lambda v: f"{v:.3f}", False),
    ("signed_bias", "Signed bias  (+ fail-open / − over-caution)", -0.03, 0.02,
     [-0.03, 0.00, 0.02], lambda v: f"{v:+.3f}", True),
]


def _rows() -> list[dict]:
    rows = [json.loads(x) for x in SUMMARY.read_text().splitlines() if x.strip()]
    return sorted(rows, key=lambda r: r["step"])


def build_svg(rows: list[dict]) -> str:
    steps = [r["step"] for r in rows]
    smin, smax = min(steps), max(steps)

    def px(step: float) -> float:
        return X0 + (step - smin) / (smax - smin) * (X1 - X0)

    height = TOP + len(PANELS) * PANEL_H + (len(PANELS) - 1) * PANEL_GAP + 44
    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" '
        f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">'
    )
    out.append(
        "<style>"
        f".ti{{fill:{INK};font-size:19px;font-weight:700;letter-spacing:-0.2px;}}"
        f".sub{{fill:{MUTED};font-size:12px;}}"
        f".eb{{fill:{AMBER};font-size:11px;font-weight:700;letter-spacing:1.5px;}}"
        f".pt{{fill:{INK};font-size:12.5px;font-weight:600;}}"
        f".tk{{fill:{MUTED};font-size:10.5px;font-variant-numeric:tabular-nums;}}"
        f".vl{{fill:{INK};font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;}}"
        f".ax{{fill:{MUTED};font-size:11px;font-variant-numeric:tabular-nums;}}"
        f".an{{fill:{MUTED};font-size:10px;}}"
        "</style>"
    )
    out.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="14" fill="{PAPER}" stroke="{CARD_LINE}"/>')
    out.append('<text x="40" y="34" class="eb">AMBERTRACE · FAITHFULNESS</text>')
    out.append('<text x="40" y="60" class="ti">Cross-domain caution transfer</text>')
    out.append(
        '<text x="40" y="79" class="sub">Narrow RL on air-track triage, scored on 120 held-out items '
        'in domains the reward never touched (main run, 250 iters)</text>'
    )

    for pi, (key, title, ymin, ymax, ticks, fmt, zero) in enumerate(PANELS):
        pt = TOP + pi * (PANEL_H + PANEL_GAP)
        pb = pt + PANEL_H

        def py(v: float, _pt=pt, _pb=pb, _ymin=ymin, _ymax=ymax) -> float:
            return _pb - (v - _ymin) / (_ymax - _ymin) * (_pb - _pt)

        out.append(f'<text x="40" y="{pt - 16}" class="pt">{title}</text>')
        # gridlines + y ticks
        for tv in ticks:
            gy = py(tv)
            emph = zero and abs(tv) < 1e-9
            stroke = MUTED if emph else CARD_LINE
            dash = ' stroke-dasharray="3 3"' if emph else ""
            out.append(f'<line x1="{X0}" y1="{gy:.1f}" x2="{X1}" y2="{gy:.1f}" stroke="{stroke}" stroke-width="1"{dash}/>')
            out.append(f'<text x="{X0 - 8}" y="{gy + 3.5:.1f}" text-anchor="end" class="tk">{fmt(tv)}</text>')
        # series line
        pts = [(px(r["step"]), py(r[key])) for r in rows]
        d = " ".join(f'{"M" if i == 0 else "L"}{x:.1f} {y:.1f}' for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{AMBER}" stroke-width="2"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{AMBER}" stroke="{PAPER}" stroke-width="1.5"/>')
        # end labels: first and last value
        first, last = rows[0], rows[-1]
        fx, fy = px(first["step"]), py(first[key])
        lx, ly = px(last["step"]), py(last[key])
        out.append(f'<text x="{fx + 9:.1f}" y="{fy + 15:.1f}" text-anchor="start" class="vl">{fmt(first[key])}</text>')
        out.append(f'<text x="{lx - 7:.1f}" y="{ly - 10:.1f}" text-anchor="end" class="vl">{fmt(last[key])}</text>')
        if zero:
            out.append(f'<text x="{X1}" y="{py(ymax) + 11:.1f}" text-anchor="end" class="an">← fail-open</text>')
            out.append(f'<text x="{X1}" y="{py(ymin) - 4:.1f}" text-anchor="end" class="an">← over-caution</text>')

    # shared x-axis (training step) under the last panel
    axis_y = TOP + len(PANELS) * PANEL_H + (len(PANELS) - 1) * PANEL_GAP + 20
    for r in rows:
        x = px(r["step"])
        out.append(f'<text x="{x:.1f}" y="{axis_y:.1f}" text-anchor="middle" class="ax">{r["step"]}</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{axis_y + 18:.1f}" text-anchor="middle" class="sub">training step</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    svg = build_svg(_rows())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg)
    print(f"wrote {out_path} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
