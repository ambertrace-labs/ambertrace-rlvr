"""Render the alignment matrix's composite alignment score (CAS) as a committable
SVG bar chart — the sibling of ``alignment_fail_open.svg``.

No plotting dependency: hand-built SVG in the Ambertrace house style so it renders
anywhere (GitHub, PyPI, docs). CAS is read straight from the persisted matrix rows
(``outputs/row_*.json``) via :func:`ambertrace_rlvr.score_matrix_cas`, so the chart
and the doc table cannot drift. Bars are sorted most-aligned first; the axis runs
0→1 (honest, untruncated) with the exact value printed at each bar end.

    python examples/plot_alignment_cas.py            # -> docs/assets/alignment_cas.svg
    python examples/plot_alignment_cas.py <out.svg>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ambertrace_rlvr.deviation import DeviationReport
from ambertrace_rlvr.matrix import AlignmentRow, score_matrix_cas

REPO = Path(__file__).resolve().parent.parent
ROWS_DIR = REPO / "outputs"
DEFAULT_OUT = REPO / "docs" / "assets" / "alignment_cas.svg"

# Ambertrace palette (matches plot_run_report.py / alignment_fail_open.svg)
PAPER, CARD_LINE, INK, MUTED, AMBER = "#F7F6F3", "#E7E4DC", "#1B1A17", "#7A776E", "#E0982E"

# The 17 published models: row file -> (display name, lab). One canonical file per
# model (row_mistral.json duplicates the v0.3 row and is skipped).
MODELS = [
    ("row_qwen3.6-35b-a3b.json", "qwen3.6-35b-a3b †", "Alibaba"),
    ("row_allenai_olmo-3-32b-think.json", "olmo-3-32b-think ‡", "Allen AI"),
    ("row_qwen_qwen3.5-9b.json", "qwen3.5-9b †", "Alibaba"),
    ("row_qwen3.6-27b.json", "qwen3.6-27b †", "Alibaba"),
    ("row_llama-4-scout.json", "llama-4-scout-17b-16e §", "Meta"),
    ("row_microsoft_phi-4.json", "phi-4", "Microsoft"),
    ("row_gemma-4-e4b-it.json", "gemma-4-e4b-it", "Google"),
    ("row_zai-org_glm-4.7-flash.json", "glm-4.7-flash", "Zhipu/Z.ai"),
    ("row_moonshotai_kimi-linear-48b-a3b-instruct.json", "kimi-linear-48b-a3b", "Moonshot AI"),
    ("row_mistralai_mistral-small-3.2.json", "mistral-small-3.2", "Mistral"),
    ("row_gemma2.json", "gemma-2-9b-it", "Google"),
    ("row_deepseek-coder-v2-lite-instruct.json", "deepseek-coder-v2-lite", "DeepSeek"),
    ("row_glm-4-9b-0414.json", "glm-4-9b-0414", "Zhipu/Z.ai"),
    ("row_meta-llama-3.1-8b-instruct.json", "meta-llama-3.1-8b-instruct", "Meta"),
    ("row_yi-1.5-9b-chat.json", "yi-1.5-9b-chat", "01.AI"),
    ("row_llama-3.2-3b-instruct.json", "llama-3.2-3b-instruct", "Meta"),
    ("row_mistralai_mistral-7b-instruct-v0.3.json", "mistral-7b-instruct-v0.3", "Mistral"),
]

_CTOR = ("correct", "over_permit", "over_deny", "refusal_on_certified",
         "parse_fail_on_certified", "overconfident", "mutual_abstain", "unverifiable")


def _report(d: dict) -> DeviationReport:
    return DeviationReport(**{k: d.get(k, 0) for k in _CTOR})


def _cas_for(row_file: Path) -> float:
    d = json.loads(row_file.read_text())["rows"][0]
    row = AlignmentRow(
        model=d["model"], report=_report(d["report"]),
        by_band={b: _report(bd) for b, bd in d["by_band"].items()},
    )
    cas = score_matrix_cas(row).cas
    if cas is None:
        raise SystemExit(f"{row_file.name}: no verifiable items — CAS undefined")
    return cas


def render(scored: list[tuple[str, str, float]]) -> str:
    W = 760
    x0, x1 = 244, 712                       # bar track (0 → 1 CAS)
    top, bot = 108, 108 + 30 * len(scored)
    H = bot + 66
    row_h = (bot - top) / len(scored)

    def bx(v: float) -> float:
        return x0 + v * (x1 - x0)

    grid, ticks = [], []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = bx(t)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bot:.1f}" '
                    f'stroke="{CARD_LINE}" stroke-width="1"/>')
        ticks.append(f'<text x="{x:.1f}" y="{bot+16:.1f}" text-anchor="middle" '
                     f'class="tick">{t:.2f}</text>')

    bars = []
    for i, (name, lab, cas) in enumerate(scored):
        cy = top + i * row_h + row_h / 2
        by = cy - 7
        bars.append(
            f'<text x="{x0-12}" y="{cy-2:.1f}" text-anchor="end" class="model">{name}</text>'
            f'<text x="{x0-12}" y="{cy+11:.1f}" text-anchor="end" class="lab">{lab}</text>'
            f'<rect x="{x0}" y="{by:.1f}" width="{bx(cas)-x0:.1f}" height="14" rx="3" fill="{AMBER}"/>'
            f'<text x="{bx(cas)+7:.1f}" y="{cy+4:.1f}" class="val" fill="{INK}">{cas:.3f}</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">
  <style>
    .title {{ fill: {INK}; font-size: 19px; font-weight: 700; letter-spacing: -0.2px; }}
    .sub {{ fill: {MUTED}; font-size: 12px; }}
    .eyebrow {{ fill: {AMBER}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }}
    .model {{ fill: {INK}; font-size: 12.5px; font-weight: 600; }}
    .lab {{ fill: {MUTED}; font-size: 10.5px; }}
    .val {{ font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .tick {{ fill: {MUTED}; font-size: 10.5px; font-variant-numeric: tabular-nums; }}
  </style>
  <rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="14" fill="{PAPER}" stroke="{CARD_LINE}"/>
  <text x="40" y="30" class="eyebrow">AMBERTRACE · ALIGNMENT</text>
  <text x="40" y="52" class="title">Composite alignment score (CAS)</text>
  <text x="40" y="72" class="sub">Accuracy and error direction folded into one score (BALANCED scheme). 120-item slice; higher is more aligned.</text>
  {"".join(grid)}
  {"".join(bars)}
  {"".join(ticks)}
</svg>
'''


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    scored = [(name, lab, _cas_for(ROWS_DIR / f)) for f, name, lab in MODELS]
    scored.sort(key=lambda r: -r[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(scored))
    print(f"wrote {out_path} ({len(scored)} models)")


if __name__ == "__main__":
    main()
