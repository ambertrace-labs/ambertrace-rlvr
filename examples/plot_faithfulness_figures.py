"""Render every figure in the faithfulness research piece (#95) as committable SVG.

House style (see docs/research/STYLE_GUIDE.md): hand-built SVG so it renders
everywhere — GitHub, Hugging Face, PyPI, IDE previews — no plotting dependency and
no Mermaid (which only renders on GitHub). Every data figure is read straight from
the probe captures so the figure and the appendix table cannot drift.

    python examples/plot_faithfulness_figures.py            # -> docs/assets/faithfulness_*.svg

Figures:
  faithfulness_reward_shape.svg     the shaped reward (diverging bar of weights)
  faithfulness_overview.svg         experiment at a glance (flow)
  faithfulness_pilot_indomain.svg   pilot in-domain probe trajectory
  faithfulness_pilot_ood.svg        pilot OOD probe trajectory
  faithfulness_main_indomain.svg    main-run in-domain (recall/precision divergence)
  faithfulness_main_rollouts.svg    main-run training-rollout terciles
  faithfulness_ood_transfer.svg     main-run OOD (cross-domain caution transfer)
  faithfulness_mechanism.svg        the caution-transfer mechanism (flow)
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "docs" / "assets"
OUT_DIR = REPO / "outputs"

# AmberTrace palette (matches plot_alignment_cas.py / alignment_*.svg)
PAPER, CARD_LINE, INK, MUTED, AMBER = "#F7F6F3", "#E7E4DC", "#1B1A17", "#7A776E", "#E0982E"
BOX_FILL, AMBER_EDGE = "#FFFFFF", "#B5761F"
FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
W = 760


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rows(name: str) -> list[dict]:
    p = OUT_DIR / name / "summary.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return sorted(rows, key=lambda r: r["step"])


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
        f".an{{fill:{MUTED};font-size:10px;}}"
        f".bx1{{fill:{INK};font-size:12px;font-weight:600;}}"
        f".bx2{{fill:{MUTED};font-size:10.5px;}}"
        f".bxi{{fill:{INK};font-size:10.5px;}}"
        + extra_style + "</style>",
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="14" fill="{PAPER}" stroke="{CARD_LINE}"/>',
    ]
    return out


def _titleblock(out: list[str], eyebrow: str, title: str, sub: str) -> None:
    out.append(f'<text x="40" y="34" class="eb">{esc(eyebrow)}</text>')
    out.append(f'<text x="40" y="60" class="ti">{esc(title)}</text>')
    out.append(f'<text x="40" y="79" class="sub">{esc(sub)}</text>')


# --- line small-multiples ---------------------------------------------------

def line_chart(rows: list[dict], xkey: str, panels: list[tuple], *, title: str, sub: str,
               xlabels: list | None = None, xaxis: str = "training step", panel_h: int = 100) -> str:
    X0, X1, TOP, GAP = 84, 724, 104, 44
    xs = [r[xkey] for r in rows]
    numeric = all(isinstance(v, (int, float)) for v in xs)
    xmin, xmax = (min(xs), max(xs)) if numeric else (0.0, 1.0)
    idx = {v: i for i, v in enumerate(xs)}
    n = len(xs)

    def px(v):
        if numeric:
            return X0 + (v - xmin) / (xmax - xmin) * (X1 - X0)
        return X0 + (idx[v] + 0.5) / n * (X1 - X0)

    height = TOP + len(panels) * panel_h + (len(panels) - 1) * GAP + 44
    out = _head(height)
    _titleblock(out, "AmberTrace · FAITHFULNESS", title, sub)

    for pi, (key, ptitle, ymin, ymax, ticks, fmt, zero) in enumerate(panels):
        pt = TOP + pi * (panel_h + GAP)
        pb = pt + panel_h

        def py(v, _pt=pt, _pb=pb, _ymin=ymin, _ymax=ymax):
            return _pb - (v - _ymin) / (_ymax - _ymin) * (_pb - _pt)

        out.append(f'<text x="40" y="{pt - 16}" class="pt">{esc(ptitle)}</text>')
        for tv in ticks:
            gy = py(tv)
            emph = zero and abs(tv) < 1e-9
            stroke = MUTED if emph else CARD_LINE
            dash = ' stroke-dasharray="3 3"' if emph else ""
            out.append(f'<line x1="{X0}" y1="{gy:.1f}" x2="{X1}" y2="{gy:.1f}" stroke="{stroke}" stroke-width="1"{dash}/>')
            out.append(f'<text x="{X0 - 8}" y="{gy + 3.5:.1f}" text-anchor="end" class="tk">{fmt(tv)}</text>')
        pts = [(px(r[xkey]), py(r[key])) for r in rows]
        d = " ".join(f'{"M" if i == 0 else "L"}{x:.1f} {y:.1f}' for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{AMBER}" stroke-width="2"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{AMBER}" stroke="{PAPER}" stroke-width="1.5"/>')
        first, last = rows[0], rows[-1]
        fx, fy = px(first[xkey]), py(first[key])
        lx, ly = px(last[xkey]), py(last[key])
        out.append(f'<text x="{fx + 9:.1f}" y="{fy + 15:.1f}" text-anchor="start" class="vl">{fmt(first[key])}</text>')
        out.append(f'<text x="{lx - 7:.1f}" y="{ly - 10:.1f}" text-anchor="end" class="vl">{fmt(last[key])}</text>')
        if zero:
            out.append(f'<text x="{X1}" y="{py(ymax) + 11:.1f}" text-anchor="end" class="an">← fail-open</text>')
            out.append(f'<text x="{X1}" y="{py(ymin) - 4:.1f}" text-anchor="end" class="an">← over-caution</text>')

    axis_y = TOP + len(panels) * panel_h + (len(panels) - 1) * GAP + 20
    labels = xlabels if xlabels is not None else [r[xkey] for r in rows]
    for r, lab in zip(rows, labels):
        out.append(f'<text x="{px(r[xkey]):.1f}" y="{axis_y:.1f}" text-anchor="middle" class="ax">{esc(str(lab))}</text>')
    out.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{axis_y + 18:.1f}" text-anchor="middle" class="sub">{esc(xaxis)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- diverging bar (reward shape) -------------------------------------------

def reward_shape() -> str:
    # Component weights (paper §2 / configs/air_track.yaml + DefaultRewardShaper defaults);
    # penalties are subtracted, shown negative.
    comps = [
        ("correctness", 1.0), ("certified", 0.5), ("graded", 0.3), ("format", 0.1),
        ("consistency (measured, not optimised)", 0.0),
        ("rejected_penalty", -0.2), ("unsupported_penalty", -0.3),
    ]
    X0, X1, TOP, ROW = 250, 724, 108, 40
    vmin, vmax = -0.5, 1.1
    height = TOP + len(comps) * ROW + 24
    out = _head(height)
    _titleblock(out, "AmberTrace · FAITHFULNESS", "The shaped reward",
                "Fail-closed; certification gates credit. Consistency is measured every step but never optimised.")

    def bx(v):
        return X0 + (v - vmin) / (vmax - vmin) * (X1 - X0)

    zero = bx(0.0)
    out.append(f'<line x1="{zero:.1f}" y1="{TOP - 6}" x2="{zero:.1f}" y2="{TOP + len(comps) * ROW - 10:.1f}" stroke="{MUTED}" stroke-width="1"/>')
    for i, (name, v) in enumerate(comps):
        y = TOP + i * ROW
        out.append(f'<text x="238" y="{y + 14:.1f}" text-anchor="end" class="pt">{esc(name)}</text>')
        x = bx(v)
        color = AMBER if v > 0 else MUTED
        if abs(v) < 1e-9:
            out.append(f'<circle cx="{zero:.1f}" cy="{y + 9:.1f}" r="3" fill="{MUTED}"/>')
        elif v > 0:
            out.append(f'<rect x="{zero:.1f}" y="{y + 2:.1f}" width="{x - zero:.1f}" height="14" rx="3" fill="{color}"/>')
        else:
            out.append(f'<rect x="{x:.1f}" y="{y + 2:.1f}" width="{zero - x:.1f}" height="14" rx="3" fill="{color}"/>')
        lx = x + 6 if v >= 0 else x - 6
        anchor = "start" if v >= 0 else "end"
        out.append(f'<text x="{lx:.1f}" y="{y + 13:.1f}" text-anchor="{anchor}" class="vl">{v:+.1f}</text>')
    # axis ticks
    for tv in (-0.5, 0.0, 0.5, 1.0):
        out.append(f'<text x="{bx(tv):.1f}" y="{TOP + len(comps) * ROW + 6:.1f}" text-anchor="middle" class="tk">{tv:+.1f}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- flow diagrams ----------------------------------------------------------

def _box(x, y, w, h, lines, accent=False):
    fill = AMBER if accent else BOX_FILL
    stroke = AMBER_EDGE if accent else CARD_LINE
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}"/>']
    n = len(lines)
    lh = 15
    y0 = y + h / 2 - (n - 1) * lh / 2 + 4
    for i, ln in enumerate(lines):
        cls = "bx1" if i == 0 else ("bxi" if accent else "bx2")
        s.append(f'<text x="{x + w / 2:.1f}" y="{y0 + i * lh:.1f}" text-anchor="middle" class="{cls}">{esc(ln)}</text>')
    return "".join(s)


def _arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#ah)"{dash}/>'


def _defs():
    return (f'<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 Z" fill="{MUTED}"/></marker></defs>')


def flow_overview() -> str:
    height = 300
    out = _head(height)
    out.append(_defs())
    _titleblock(out, "AmberTrace · FAITHFULNESS", "The experiment at a glance",
                "One model, one certified domain, two probe arms.")
    yb = 40
    M = (24, 96 + yb, 176, 48)
    P = (24, 176 + yb, 176, 48)
    R = (232, 176 + yb, 168, 48)
    T = (432, 136 + yb, 132, 48)
    IN = (596, 96 + yb, 148, 48)
    OD = (596, 176 + yb, 148, 48)
    out.append(_box(*M, ["OLMo-3-7B-Think-SFT", "pre-RL checkpoint"]))
    out.append(_box(*P, ["Air-track triage policy", "certified by the kernel"]))
    out.append(_box(*R, ["Fail-closed shaped reward", "consistency weight = 0"]))
    out.append(_box(*T, ["GRPO training", "250 iters · 8-bit QLoRA"], accent=True))
    out.append(_box(*IN, ["In-domain probes", "faithfulness · CoT-drift"]))
    out.append(_box(*OD, ["OOD probes", "unseen domains"]))
    out.append(_arrow(M[0] + M[2], M[1] + M[3] / 2, T[0], T[1] + 14))
    out.append(_arrow(P[0] + P[2], P[1] + P[3] / 2, R[0], R[1] + R[3] / 2))
    out.append(_arrow(R[0] + R[2], R[1] + R[3] / 2, T[0], T[1] + 34))
    out.append(_arrow(T[0] + T[2], T[1] + 14, IN[0], IN[1] + IN[3] / 2))
    out.append(_arrow(T[0] + T[2], T[1] + 34, OD[0], OD[1] + OD[3] / 2))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def flow_mechanism() -> str:
    height = 372
    out = _head(height)
    out.append(_defs())
    _titleblock(out, "AmberTrace · FAITHFULNESS", "Why narrow RL transfers caution",
                "The reward never names the OOD domains — the disposition it instils is domain-general.")
    yb = 30
    A = (24, 84 + yb, 300, 50)
    B = (24, 164 + yb, 300, 50)
    C = (24, 244 + yb, 300, 50)
    D = (452, 74 + yb, 284, 42)
    E = (452, 134 + yb, 284, 42)
    F = (452, 194 + yb, 284, 42)
    out.append(_box(*A, ["Narrow RL on air-track triage", "reward = certificate + correctness"]))
    out.append(_box(*B, ["Transferable disposition", "resolve uncertainty → restrictive action"], accent=True))
    out.append(_box(*C, ["Unseen domains", "loan · eligibility · …"]))
    out.append(_box(*D, ["Fail-open rate → 0"]))
    out.append(_box(*E, ["Signed bias flips: +0.018 → −0.017"]))
    out.append(_box(*F, ["OOD accuracy: 0.945 → 0.983"]))
    out.append(_arrow(A[0] + A[2] / 2, A[1] + A[3], B[0] + B[2] / 2, B[1]))
    out.append(_arrow(B[0] + B[2] / 2, B[1] + B[3], C[0] + C[2] / 2, C[1]))
    for box in (D, E, F):
        out.append(_arrow(C[0] + C[2], C[1] + C[3] / 2, box[0], box[1] + box[3] / 2))
    # guard annotation (dotted, no domain vocabulary leaks)
    out.append(_arrow(B[0] + B[2], B[1] + 25, C[0] + C[2], C[1] + 10, dashed=True))
    out.append(f'<text x="{C[0] + C[2] + 8}" y="{C[1] - 2}" class="an">no vocabulary bleed:</text>')
    out.append(f'<text x="{C[0] + C[2] + 8}" y="{C[1] + 12}" class="an">policy-bleed flat · format-leak 0</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- figure registry --------------------------------------------------------

def _pct3(v):
    return f"{v:.3f}"


def _sgn3(v):
    return f"{v:+.3f}"


def build_all() -> dict[str, str]:
    pilot_in = _rows("probe_runs")
    main_in = _rows("probe_runs_main")
    pilot_ood = _rows("ood_probe_runs")
    main_ood = _rows("ood_probe_runs_main")
    terciles = [
        {"step": "early", "mean_reward": 0.670, "mean_faithfulness": 0.290, "mean_consistency": 0.030},
        {"step": "middle", "mean_reward": 0.731, "mean_faithfulness": 0.257, "mean_consistency": 0.040},
        {"step": "late", "mean_reward": 0.700, "mean_faithfulness": 0.280, "mean_consistency": 0.030},
    ]

    ood_panels = [
        ("accuracy", "OOD accuracy", 0.90, 1.00, [0.90, 0.95, 1.00], _pct3, False),
        ("fail_open_rate", "Fail-open rate", 0.00, 0.04, [0.00, 0.02, 0.04], _pct3, False),
        ("signed_bias", "Signed bias  (+ fail-open / − over-caution)", -0.03, 0.02, [-0.03, 0.00, 0.02], _sgn3, True),
    ]
    indomain_panels = [
        ("mean_faithfulness", "Faithfulness (recall of credited rules)", 0.15, 0.25, [0.15, 0.20, 0.25], _pct3, False),
        ("mean_consistency", "Consistency (precision — no false citations)", 0.00, 0.10, [0.00, 0.05, 0.10], _pct3, False),
        ("decision_accuracy", "Accuracy", 0.70, 0.90, [0.70, 0.80, 0.90], lambda v: f"{v:.2f}", False),
        ("mean_reward", "Reward", 1.10, 1.45, [1.10, 1.25, 1.40], lambda v: f"{v:.3f}", False),
    ]
    pilot_in_panels = [
        ("mean_reward", "Reward", 0.80, 1.40, [0.80, 1.10, 1.40], lambda v: f"{v:.3f}", False),
        ("decision_accuracy", "Accuracy", 0.60, 0.90, [0.60, 0.75, 0.90], lambda v: f"{v:.2f}", False),
        ("mean_faithfulness", "Faithfulness", 0.15, 0.30, [0.15, 0.20, 0.25, 0.30], _pct3, False),
        ("mean_consistency", "Consistency", 0.00, 0.10, [0.00, 0.05, 0.10], _pct3, False),
    ]
    tercile_panels = [
        ("mean_reward", "Mean reward", 0.60, 0.80, [0.60, 0.70, 0.80], lambda v: f"{v:.3f}", False),
        ("mean_faithfulness", "Mean faithfulness", 0.20, 0.35, [0.20, 0.25, 0.30, 0.35], _pct3, False),
        ("mean_consistency", "Mean consistency", 0.00, 0.06, [0.00, 0.03, 0.06], _pct3, False),
    ]

    return {
        "faithfulness_reward_shape.svg": reward_shape(),
        "faithfulness_overview.svg": flow_overview(),
        "faithfulness_pilot_indomain.svg": line_chart(
            pilot_in, "step", pilot_in_panels, panel_h=86,
            title="Pilot · in-domain probes",
            sub="60 iterations, held-out 50-item probe · reward and accuracy flat, faithfulness drifts up mildly"),
        "faithfulness_pilot_ood.svg": line_chart(
            pilot_ood, "step", ood_panels, panel_h=100,
            title="Pilot · cross-domain (OOD) probes",
            sub="120 held-out items in unseen domains · the caution shift already appears"),
        "faithfulness_main_indomain.svg": line_chart(
            main_in, "step", indomain_panels, panel_h=86,
            title="Main run · in-domain probes",
            sub="Recall falls while precision rises — the model cites fewer rules but stops naming unfired ones"),
        "faithfulness_main_rollouts.svg": line_chart(
            terciles, "step", tercile_panels, panel_h=100, xaxis="training progress (terciles)",
            title="Main run · training-rollout terciles",
            sub="Faithfulness and consistency flat across training; reward at its shaped plateau"),
        "faithfulness_ood_transfer.svg": line_chart(
            main_ood, "step", ood_panels, panel_h=122,
            title="Cross-domain caution transfer",
            sub="Narrow RL on air-track triage, scored on 120 held-out items in domains the reward never touched (main run)"),
        "faithfulness_mechanism.svg": flow_mechanism(),
    }


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, svg in build_all().items():
        (ASSETS / name).write_text(svg)
        print(f"wrote docs/assets/{name} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
