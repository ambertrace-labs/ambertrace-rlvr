"""Render the quantisation piece's rule-structure figure as committable SVG.

House style (see docs/research/STYLE_GUIDE.md): hand-built SVG, no plotting
dependency, no Mermaid. Reads the quant-sweep capture
(`outputs/quant_full_qwen36_27b.json`) so the figure and the appendix table cannot
drift.

    python examples/plot_quant_figures.py     # -> docs/assets/quant_structure_heatmap.svg
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "outputs" / "quant_full_qwen36_27b.json"
OUT = REPO / "docs" / "assets" / "quant_structure_heatmap.svg"

PAPER, CARD_LINE, INK, MUTED, AMBER = "#F7F6F3", "#E7E4DC", "#1B1A17", "#7A776E", "#E0982E"
FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

# quant level -> bit label, high precision first
BITS = [("BQ80", "8-bit"), ("BQ6K", "6-bit"), ("BQ5KM", "5-bit"),
        ("BQ4KM", "4-bit"), ("BQ3KM", "3-bit"), ("Q2K", "2-bit")]
# rule structures, most-fail-open first; (key, display)
STRUCTS = [("ratio", "ratio"), ("precedence", "precedence"), ("baseline", "baseline"),
           ("multi_trigger_disjunction", "multi-trigger"), ("negation", "negation")]


def _shade(fo: float) -> str:
    """White -> amber as fail-open rises (0..0.22 clamped). Darker = worse."""
    t = max(0.0, min(1.0, fo / 0.22))
    r0, g0, b0 = 247, 246, 243
    r1, g1, b1 = 224, 152, 46
    return f"#{int(r0 + (r1 - r0) * t):02X}{int(g0 + (g1 - g0) * t):02X}{int(b0 + (b1 - b0) * t):02X}"


def build_svg() -> str:
    res = json.loads(SRC.read_text())["results"]
    x0, top, cw, ch = 250, 128, 72, 34
    W = 760
    H = top + ch * len(STRUCTS) + 54
    body = []
    for j, (_, blab) in enumerate(BITS):
        body.append(f'<text x="{x0 + cw * j + cw / 2:.1f}" y="{top - 8:.1f}" text-anchor="middle" class="col">{blab}</text>')
    for i, (sk, slab) in enumerate(STRUCTS):
        y = top + i * ch
        n = res["BQ80"]["by_struct"][sk][1]
        body.append(f'<text x="{x0 - 14}" y="{y + ch / 2 + 1:.1f}" text-anchor="end" class="md">{slab}</text>')
        body.append(f'<text x="{x0 - 14}" y="{y + ch / 2 + 13:.1f}" text-anchor="end" class="lb">n={n}</text>')
        for j, (lvl, _) in enumerate(BITS):
            fo_count, nn = res[lvl]["by_struct"][sk]
            fo = fo_count / nn
            x = x0 + cw * j
            fill = _shade(fo)
            ink = "#FFFFFF" if fo >= 0.14 else INK
            body.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{fill}" stroke="{CARD_LINE}"/>')
            body.append(f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" text-anchor="middle" class="cell" fill="{ink}">{fo * 100:.1f}%</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">
  <style>
    .ti {{ fill: {INK}; font-size: 19px; font-weight: 700; letter-spacing: -0.2px; }}
    .sub {{ fill: {MUTED}; font-size: 12px; }} .eb {{ fill: {AMBER}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }}
    .md {{ fill: {INK}; font-size: 12.5px; font-weight: 600; }} .lb {{ fill: {MUTED}; font-size: 10.5px; font-variant-numeric: tabular-nums; }}
    .col {{ fill: {MUTED}; font-size: 11px; }} .cell {{ font-size: 11px; font-weight: 700; font-variant-numeric: tabular-nums; }}
  </style>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" fill="{PAPER}" stroke="{CARD_LINE}"/>
  <text x="40" y="30" class="eb">AMBERTRACE · QUANTISATION</text>
  <text x="40" y="52" class="ti">Fail-open concentrates in ratio rules</text>
  <text x="40" y="72" class="sub">Qwen3.6-27B · fail-open rate on safety-critical decisions by rule structure across the precision ladder · darker = higher.</text>
  {"".join(body)}
</svg>
'''


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_svg())
    print(f"wrote {OUT} ({len(OUT.read_text())} bytes)")


if __name__ == "__main__":
    main()
