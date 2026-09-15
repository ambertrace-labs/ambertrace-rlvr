"""Regenerate the full-run alignment-matrix doc from local run artifacts.

Reads every ``outputs/row_full_<id>.json`` (each carrying CAS + the reasoning-
complexity profile — ``by_structure`` / ``by_action_count`` — written by
``run_slice.py``), de-duplicates per model, and emits:

  1. the **CAS matrix** markdown table (sorted best-first), with the reasoning
     taxonomy flag († disable-works reasoner, ‡ dedicated thinker),
  2. the **reasoning-complexity profile** — model × structure and model ×
     action-count accuracy tables (the piece deferred from the 120-item slice),
  3. a refreshed ``docs/assets/alignment_cas_1350.svg`` bar chart.

    python examples/gen_alignment_matrix.py            # markdown -> stdout, SVG -> docs/assets/
    python examples/gen_alignment_matrix.py --svg-only

Local helper: reads gitignored ``outputs/``; the emitted markdown is pasted into
``docs/ALIGNMENT_MATRIX.md``. Reusable for the #90 new-model re-runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SVG_OUT = REPO / "docs" / "assets" / "alignment_cas_1350.svg"
STRUCTS = ["baseline", "ratio", "precedence", "negation", "multi_trigger_disjunction"]
# A model whose outputs coerce to an action word on fewer than this fraction of
# certified items is dropped from the ranked table (its score would reflect a
# biased sub-sample). E.g. Llama-3.1-8B emits tool-call JSON, not action words.
PARSE_FLOOR = 0.85

# normalized-key -> (display, lab, params, flag). flag: "†"=reasoner w/ working
# disable (run nothink), "‡"=dedicated thinker no disable (run 4096), ""=plain.
META = {
    "muse-glimmer-30b": ("Muse-Glimmer-30B", "Meta", "30B", "‡"),
    "olmo-3-32b-think": ("OLMo-3-32B-Think", "Allen AI", "32B", "‡"),
    "olmo-3.1-32b": ("OLMo-3.1-32B-Instruct", "Allen AI", "32B", ""),
    "qwen3.8-27b": ("Qwen3.8-27B", "Alibaba", "27B", "†"),
    "qwen3.8-27b-think": ("Qwen3.8-27B (reasoning)", "Alibaba", "27B", "‡"),
    "qwen3.6-35b-a3b": ("Qwen3.6-35B-A3B", "Alibaba", "35B-A3B", "†"),
    "qwen3.6-27b": ("Qwen3.6-27B", "Alibaba", "27B", "†"),
    "qwen3.5-9b": ("Qwen3.5-9B", "Alibaba", "9B", "†"),
    "glm-4.7-flash": ("GLM-4.7-Flash", "Zhipu/Z.ai", "30B-MoE", "†"),
    "glm-4-9b-0414": ("GLM-4-9B-0414", "Zhipu/Z.ai", "9B", ""),
    "kimi-linear-48b-a3b": ("Kimi-Linear-48B-A3B", "Moonshot AI", "48B-A3B", ""),
    "phi-4": ("Phi-4", "Microsoft", "14B", ""),
    "gemma-4-e4b": ("Gemma-4-E4B", "Google", "~4B", ""),
    "gemma-2-9b": ("Gemma-2-9B", "Google", "9B", ""),
    "mistral-small-3.2": ("Mistral-Small-3.2", "Mistral", "24B", ""),
    "mistral-7b-v0.3": ("Mistral-7B-v0.3", "Mistral", "7B", ""),
    "deepseek-coder-v2-lite": ("DeepSeek-Coder-V2-Lite", "DeepSeek", "16B-MoE", ""),
    "llama-3.1-8b": ("Llama-3.1-8B", "Meta", "8B", ""),
    "llama-3.2-3b": ("Llama-3.2-3B", "Meta", "3B", ""),
    "yi-1.5-9b": ("Yi-1.5-9B", "01.AI", "9B", ""),
}
PALETTE = {"PAPER": "#F7F6F3", "CARD_LINE": "#E7E4DC", "INK": "#1B1A17",
           "MUTED": "#7A776E", "AMBER": "#E0982E"}


def norm(model: str) -> str:
    """Normalize an lms id to a metadata/dedup key: drop publisher + quant/format
    and instruct suffixes so ``qwen/qwen3.6-27b`` and ``qwen3.6-27b`` collapse."""
    m = model.lower().split("/")[-1]
    for suf in ("-mlx-4bit", "-mlx-6bit", "-mlx-8bit", "-mlx", "-gguf",
                "-instruct", "-chat", "-bq4km", "-it"):
        m = m.replace(suf, "")
    return m.replace("meta-llama-", "llama-").replace("-v0.3", "-v0.3").strip("-")


def meta_for(key: str) -> tuple[str, str, str, str]:
    if key in META:
        return META[key]
    # longest-substring fallback
    for mk in sorted(META, key=len, reverse=True):
        if mk in key or key in mk:
            return META[mk]
    return (key, "?", "?", "")


def load_rows() -> dict[str, dict]:
    """model-key -> row dict (deduped, keeping the most recently written file)."""
    best: dict[str, tuple[float, dict]] = {}
    for f in glob.glob(str(REPO / "outputs" / "row_full_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        r = d["rows"][0]
        key = norm(r["model"])
        mt = os.path.getmtime(f)
        if key not in best or mt > best[key][0]:
            best[key] = (mt, r)
    return {k: v[1] for k, v in best.items()}


def _p(x):
    return "—" if x is None else f"{x:.1%}"


def cas_table(rows: dict[str, dict]) -> str:
    items = []
    for key, r in rows.items():
        disp, lab, params, flag = meta_for(key)
        items.append((r["cas"]["value"] or -1, disp, flag, lab, params, r))
    items.sort(key=lambda t: -t[0])
    out = ["| model | lab | params | **CAS** | acc | FO (restrictive) | signed bias | refusal |",
           "|---|---|---|---|---|---|---|---|"]
    for cas, disp, flag, lab, params, r in items:
        tag = f" {flag}" if flag else ""
        sb = r["signed_bias"]
        out.append(f"| {disp}{tag} | {lab} | {params} | {cas:.3f} | {_p(r['accuracy'])} "
                   f"| {_p(r['fail_open_restrictive'])} | {sb:+.2f} | {_p(r['refusal_rate'])} |")
    return "\n".join(out)


def profile_table(rows: dict[str, dict], slice_key: str, cols, header: str) -> str:
    out = ["| model | " + " | ".join(str(c) for c in cols) + " |",
           "|---|" + "|".join(["---"] * len(cols)) + "|"]
    ranked = sorted(rows.items(), key=lambda kv: -(kv[1]["cas"]["value"] or -1))
    for key, r in ranked:
        disp = meta_for(key)[0]
        cells = []
        for c in cols:
            sub = r[slice_key].get(str(c))
            cells.append(_p(sub["accuracy"]) if sub and sub["accuracy"] is not None else "—")
        out.append(f"| {disp} | " + " | ".join(cells) + " |")
    return f"**{header}**\n\n" + "\n".join(out)


def render_svg(rows: dict[str, dict]) -> str:
    C = PALETTE
    scored = sorted(((meta_for(k)[0] + (f" {meta_for(k)[3]}" if meta_for(k)[3] else ""),
                      meta_for(k)[1], r["cas"]["value"]) for k, r in rows.items()),
                    key=lambda t: -t[2])
    W, x0, x1 = 760, 250, 712
    top = 108
    row_h, bot = 30, top + 30 * len(scored)
    H = bot + 66

    def bx(v):
        return x0 + v * (x1 - x0)
    grid, ticks, bars = [], [], []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = bx(t)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bot:.1f}" stroke="{C["CARD_LINE"]}" stroke-width="1"/>')
        ticks.append(f'<text x="{x:.1f}" y="{bot+16:.1f}" text-anchor="middle" class="tk">{t:.2f}</text>')
    for i, (name, lab, cas) in enumerate(scored):
        cy = top + i * row_h + row_h / 2
        bars.append(
            f'<text x="{x0-12}" y="{cy-2:.1f}" text-anchor="end" class="md">{name}</text>'
            f'<text x="{x0-12}" y="{cy+11:.1f}" text-anchor="end" class="lb">{lab}</text>'
            f'<rect x="{x0}" y="{cy-7:.1f}" width="{bx(cas)-x0:.1f}" height="14" rx="3" fill="{C["AMBER"]}"/>'
            f'<text x="{bx(cas)+7:.1f}" y="{cy+4:.1f}" class="vl" fill="{C["INK"]}">{cas:.3f}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">
  <style>
    .ti {{ fill: {C["INK"]}; font-size: 19px; font-weight: 700; letter-spacing: -0.2px; }}
    .sub {{ fill: {C["MUTED"]}; font-size: 12px; }} .eb {{ fill: {C["AMBER"]}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }}
    .md {{ fill: {C["INK"]}; font-size: 12.5px; font-weight: 600; }} .lb {{ fill: {C["MUTED"]}; font-size: 10.5px; }}
    .vl {{ font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }} .tk {{ fill: {C["MUTED"]}; font-size: 10.5px; font-variant-numeric: tabular-nums; }}
  </style>
  <rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="14" fill="{C["PAPER"]}" stroke="{C["CARD_LINE"]}"/>
  <text x="40" y="30" class="eb">AMBERTRACE · ALIGNMENT</text>
  <text x="40" y="52" class="ti">Composite alignment score (CAS)</text>
  <text x="40" y="72" class="sub">Full 1,350-item run · BALANCED scheme · higher is more aligned.</text>
  {"".join(grid)}{"".join(bars)}{"".join(ticks)}
</svg>
'''


def _svg_shell(w: int, h: int, eyebrow: str, title: str, sub: str, body: str, extra_css: str = "") -> str:
    C = PALETTE
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif">
  <style>
    .ti {{ fill: {C["INK"]}; font-size: 19px; font-weight: 700; letter-spacing: -0.2px; }}
    .sub {{ fill: {C["MUTED"]}; font-size: 12px; }} .eb {{ fill: {C["AMBER"]}; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; }}
    .md {{ fill: {C["INK"]}; font-size: 12.5px; font-weight: 600; }} .lb {{ fill: {C["MUTED"]}; font-size: 10.5px; }}
    .vl {{ font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }} .tk {{ fill: {C["MUTED"]}; font-size: 10.5px; font-variant-numeric: tabular-nums; }}
    .cell {{ font-size: 11px; font-weight: 700; font-variant-numeric: tabular-nums; }} .col {{ fill: {C["MUTED"]}; font-size: 10.5px; }}{extra_css}
  </style>
  <rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="14" fill="{C["PAPER"]}" stroke="{C["CARD_LINE"]}"/>
  <text x="40" y="30" class="eb">{eyebrow}</text>
  <text x="40" y="52" class="ti">{title}</text>
  <text x="40" y="72" class="sub">{sub}</text>
  {body}
</svg>
'''


def render_bar_svg(rows: dict[str, dict], value_key: str, *, title: str, sub: str,
                   vmin: float, vmax: float, fmt, diverging: bool = False) -> str:
    """Horizontal per-model bar chart (CAS / fail-open / signed bias), sorted worst-adjacent."""
    C = PALETTE

    def val(r):
        return r["cas"]["value"] if value_key == "cas" else r[value_key]
    items = sorted(((meta_for(k)[0] + (f" {meta_for(k)[3]}" if meta_for(k)[3] else ""),
                     meta_for(k)[1], val(r)) for k, r in rows.items()),
                   key=lambda t: -(t[2] if t[2] is not None else -9))
    W, x0, x1, top, row_h = 760, 250, 712, 108, 30
    bot = top + row_h * len(items)
    H = bot + 66

    def bx(v):
        return x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)
    zero_x = bx(0.0) if diverging else x0
    body = []
    ticks_at = [vmin, (vmin + vmax) / 2, vmax] if diverging else [vmin, (vmin + vmax) / 2, vmax]
    for t in ticks_at:
        x = bx(t)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bot:.1f}" stroke="{C["CARD_LINE"]}" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{bot+16:.1f}" text-anchor="middle" class="tk">{fmt(t)}</text>')
    if diverging:
        body.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{bot:.1f}" stroke="{C["MUTED"]}" stroke-width="1"/>')
    for i, (name, lab, v) in enumerate(items):
        cy = top + i * row_h + row_h / 2
        vv = v if v is not None else 0.0
        body.append(f'<text x="{x0-12}" y="{cy-2:.1f}" text-anchor="end" class="md">{name}</text>')
        body.append(f'<text x="{x0-12}" y="{cy+11:.1f}" text-anchor="end" class="lb">{lab}</text>')
        x = bx(vv)
        if diverging and vv < 0:
            body.append(f'<rect x="{x:.1f}" y="{cy-7:.1f}" width="{zero_x-x:.1f}" height="14" rx="3" fill="{C["MUTED"]}"/>')
            body.append(f'<text x="{x-7:.1f}" y="{cy+4:.1f}" text-anchor="end" class="vl" fill="{C["INK"]}">{fmt(vv)}</text>')
        else:
            body.append(f'<rect x="{zero_x:.1f}" y="{cy-7:.1f}" width="{x-zero_x:.1f}" height="14" rx="3" fill="{C["AMBER"]}"/>')
            body.append(f'<text x="{x+7:.1f}" y="{cy+4:.1f}" class="vl" fill="{C["INK"]}">{fmt(vv)}</text>')
    return _svg_shell(W, H, "AMBERTRACE · ALIGNMENT", title, sub, "".join(body))


def render_group_svg(rows: dict[str, dict]) -> str:
    """Mean CAS by reasoning group (thinking ‡ / reasoner † / direct)."""
    C = PALETTE
    groups = {"‡": ("thinking-enabled", []), "†": ("reasoner (disabled)", []), "": ("direct (no reasoning)", [])}
    for k, r in rows.items():
        groups[meta_for(k)[3]][1].append(r["cas"]["value"])
    order = ["‡", "†", ""]
    W, x0, x1, top, row_h = 760, 250, 712, 112, 60
    H = top + row_h * len(order) + 40

    def bx(v):
        return x0 + v * (x1 - x0)
    body = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = bx(t)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+row_h*len(order):.1f}" stroke="{C["CARD_LINE"]}" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{top+row_h*len(order)+16:.1f}" text-anchor="middle" class="tk">{t:.2f}</text>')
    for i, g in enumerate(order):
        label, vals = groups[g]
        mean = sum(vals) / len(vals)
        cy = top + i * row_h + row_h / 2
        body.append(f'<text x="{x0-12}" y="{cy-2:.1f}" text-anchor="end" class="md">{label}</text>')
        body.append(f'<text x="{x0-12}" y="{cy+12:.1f}" text-anchor="end" class="lb">n={len(vals)} · range {min(vals):.3f}–{max(vals):.3f}</text>')
        body.append(f'<rect x="{x0}" y="{cy-8:.1f}" width="{bx(mean)-x0:.1f}" height="16" rx="3" fill="{C["AMBER"]}"/>')
        body.append(f'<text x="{bx(mean)+7:.1f}" y="{cy+4:.1f}" class="vl" fill="{C["INK"]}">{mean:.3f}</text>')
    return _svg_shell(W, H, "AMBERTRACE · ALIGNMENT", "Reasoning drives alignment",
                      "Mean composite alignment score by reasoning group · full 1,350-item run.", "".join(body))


def render_structure_svg(rows: dict[str, dict]) -> str:
    """Heatmap: accuracy by decision structure (model x structure), sequential amber."""
    C = PALETTE
    ranked = sorted(rows.items(), key=lambda kv: -(kv[1]["cas"]["value"] or -1))
    labels = ["baseline", "ratio", "precedence", "negation", "multi-trigger"]
    W, x0, top, cw, ch = 760, 250, 128, 88, 26
    H = top + ch * len(ranked) + 54

    def shade(acc):
        # sequential single hue: white -> amber as accuracy rises (0.4..1.0 clamped)
        t = max(0.0, min(1.0, (acc - 0.4) / 0.6))
        r0, g0, b0 = 247, 246, 243   # paper
        r1, g1, b1 = 224, 152, 46    # amber
        return f'#{int(r0+(r1-r0)*t):02X}{int(g0+(g1-g0)*t):02X}{int(b0+(b1-b0)*t):02X}'
    body = []
    for j, lab in enumerate(labels):
        body.append(f'<text x="{x0+cw*j+cw/2:.1f}" y="{top-8:.1f}" text-anchor="middle" class="col">{lab}</text>')
    for i, (k, r) in enumerate(ranked):
        y = top + i * ch
        body.append(f'<text x="{x0-12}" y="{y+ch/2+4:.1f}" text-anchor="end" class="md">{meta_for(k)[0]}</text>')
        for j, sk in enumerate(STRUCTS):
            sub = r["by_structure"].get(sk)
            acc = sub["accuracy"] if sub and sub["accuracy"] is not None else None
            x = x0 + cw * j
            fill = shade(acc) if acc is not None else C["PAPER"]
            body.append(f'<rect x="{x+1:.1f}" y="{y+1:.1f}" width="{cw-2}" height="{ch-2}" rx="3" fill="{fill}" stroke="{C["CARD_LINE"]}"/>')
            txt = f'{acc:.0%}' if acc is not None else '—'
            ink = C["INK"] if (acc is None or acc < 0.85) else "#FFFFFF"
            body.append(f'<text x="{x+cw/2:.1f}" y="{y+ch/2+4:.1f}" text-anchor="middle" class="cell" fill="{ink}">{txt}</text>')
    return _svg_shell(W, H, "AMBERTRACE · ALIGNMENT", "Where models break",
                      "Accuracy by decision structure · ranked by CAS · darker = higher accuracy.", "".join(body))


def _write_charts(ranked: dict[str, dict]) -> list[Path]:
    A = REPO / "docs" / "assets"
    written = [SVG_OUT]
    SVG_OUT.write_text(render_svg(ranked))
    figs = {
        "alignment_fail_open.svg": render_bar_svg(
            ranked, "fail_open_restrictive", title="Fail-open on safety-critical decisions",
            sub="Rate of choosing an unsafe permissive action where the policy is restrictive · lower is safer.",
            vmin=0.0, vmax=0.4, fmt=lambda v: f"{v:.0%}"),
        "alignment_signed_bias.svg": render_bar_svg(
            ranked, "signed_bias", title="Net direction of misalignment",
            sub="Signed bias: positive = fail-open (permissive), negative = over-cautious.",
            vmin=-0.15, vmax=0.15, fmt=lambda v: f"{v:+.2f}", diverging=True),
        "alignment_reasoning_group.svg": render_group_svg(ranked),
        "alignment_structure_heatmap.svg": render_structure_svg(ranked),
    }
    for name, svg in figs.items():
        (A / name).write_text(svg)
        written.append(A / name)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg-only", action="store_true")
    args = ap.parse_args()
    rows = load_rows()
    ranked = {k: r for k, r in rows.items() if (r["parse_rate"] or 0) >= PARSE_FLOOR}
    excluded = {k: r for k, r in rows.items() if (r["parse_rate"] or 0) < PARSE_FLOOR}
    written = _write_charts(ranked)
    if args.svg_only:
        print(f"wrote {len(written)} SVGs ({len(ranked)} models): " + ", ".join(p.name for p in written))
        return
    print(f"## Results — {len(ranked)} models (full 1,350-item run)\n")
    print(cas_table(ranked))
    print("\n### Reasoning-complexity profile\n")
    print(profile_table(ranked, "by_structure", STRUCTS, "Accuracy by decision structure"))
    print("\n")
    print(profile_table(ranked, "by_action_count", [2, 3, 4], "Accuracy by action count (vocabulary size)"))
    if excluded:
        print("\n### Excluded\n")
        for k, r in excluded.items():
            print(f"- **{meta_for(k)[0]}** — only {r['parse_rate']:.0%} of outputs coerced to an "
                  f"action word ({r['report']['parse_fail_on_certified']} parse-failures); dropped "
                  f"from the ranking as its score would reflect a biased sub-sample.")
    print(f"\n<!-- SVG refreshed -> {SVG_OUT.relative_to(REPO)} ({len(ranked)} ranked, {len(excluded)} excluded) -->")


if __name__ == "__main__":
    main()
