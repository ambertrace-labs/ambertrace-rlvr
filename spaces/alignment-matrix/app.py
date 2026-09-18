"""Certified Alignment Matrix — a Hugging Face leaderboard Space (#106).

How faithfully do open-weight models stay to a *machine-checked* decision policy as
they reason? Every score here comes from the fail-closed AmberTrace verifier, not an
LLM judge — and the certificate is obtained live, never shipped, so a row cannot be
gamed by memorising a key (AT = gold).

Reads ``AmberTraceLabs/alignment-matrix-results`` (see ``data.py``); run locally
with ``python app.py``. Data + branding live in this repo; ``examples/
export_matrix_results.py`` regenerates the dataset."""
from __future__ import annotations

import gradio as gr

from leaderboard import PYPI_URL, REPO_URL, SCHEMES, load_rows, to_table

AMBER = "#E0982E"
INK = "#1B1A17"
MUTED = "#7A776E"

ROWS = load_rows()

INTRO = """
<div style="font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">
  <div style="color:{amber}; font-size:12px; font-weight:700; letter-spacing:1.6px;">AmberTrace · ALIGNMENT</div>
  <h1 style="color:{ink}; font-size:28px; margin:4px 0 2px; letter-spacing:-0.4px;">Certified Alignment Matrix</h1>
  <p style="color:{muted}; font-size:15px; max-width:70ch; margin:6px 0 0;">
    How faithfully do open-weight models stay to a <b>machine-checked</b> decision policy as they reason?
    Every score is from the fail-closed <a href="{repo}" style="color:{amber};">AmberTrace</a> verifier
    — not an LLM judge —
    over the 1,350-item <code>decision_eval_v1</code> corpus. The headline isn't accuracy but the
    <b>direction</b> of error: <i>fail-open</i> (under-restriction) on the safety-critical band is the failure a
    plain accuracy number hides.
  </p>
</div>
""".format(amber=AMBER, ink=INK, muted=MUTED, repo=REPO_URL)

SCHEME_NOTE = {
    "Balanced": "Over-permit penalised 2× over-deny — the default reading.",
    "Safety-first": "Over-permit penalised 10× over-deny — fail-open is the cardinal sin.",
    "Capital-adequacy": "Over-permit penalised ~7× over-deny — a regulator's asymmetry.",
}

ADD_MODEL = f"""
### Add your model

The leaderboard scores **live** against AmberTrace, so a new row can't be gamed by
memorising a key. To submit:

```bash
pip install ambertrace-rlvr
# point run_alignment_matrix.py at your model (any OpenAI-compatible endpoint /
# local runtime), then it scores the 1,350-item corpus against the live verifier:
python examples/run_alignment_matrix.py --model <your-model>
```

Open a PR adding your `outputs/row_full_<model>.json`; `examples/export_matrix_results.py`
folds it into the dataset this Space reads. Full recipe + gotchas in the
[repo]({REPO_URL}). Every row links back to its capture.
"""

FOOTER = f"""
<div style="color:{MUTED}; font-size:13px; margin-top:8px;">
  Composite Alignment Score (CAS) = 1 − severity-weighted deviation from the certified oracle; higher is more aligned.
  Signed bias &lt; 0 = net cautious, &gt; 0 = net fail-open.
  · <a href="{REPO_URL}" style="color:{AMBER};">Repo</a>
  · <a href="{PYPI_URL}" style="color:{AMBER};">PyPI</a>
</div>
"""


def render(scheme: str):
    headers, rows = to_table(ROWS, scheme)
    return (
        gr.Dataframe(
            value=rows,
            headers=headers,
            datatype=["number", "str", "str", "str", "str", "str", "str", "str", "str", "str", "markdown"],
            interactive=False,
            wrap=True,
        ),
        f"**{scheme} scheme** · {SCHEME_NOTE.get(scheme, '')}",
    )


with gr.Blocks(title="AmberTrace — Certified Alignment Matrix", theme=gr.themes.Soft()) as demo:
    gr.HTML(INTRO)
    scheme = gr.Radio(
        choices=list(SCHEMES.keys()), value="Balanced", label="Penalty scheme",
        info="How much heavier is a fail-open error than a fail-closed one?",
    )
    note = gr.Markdown(f"**Balanced scheme** · {SCHEME_NOTE['Balanced']}")
    init_headers, init_rows = to_table(ROWS, "Balanced")
    table = gr.Dataframe(
        value=init_rows,
        headers=init_headers,
        datatype=["number", "str", "str", "str", "str", "str", "str", "str", "str", "str", "markdown"],
        interactive=False,
        wrap=True,
    )
    scheme.change(render, inputs=scheme, outputs=[table, note])
    with gr.Accordion("Add your model", open=False):
        gr.Markdown(ADD_MODEL)
    gr.HTML(FOOTER)


if __name__ == "__main__":
    demo.launch()
