# AmberTrace® Research — Style Guide

How we write and illustrate the pieces in `docs/research/`. The goal is research
that **stands on numbers you can regenerate** and **reads at a glance** — the key
point of every result is visible in a figure, not buried in a table.

This guide is the house instance of the [`dataviz`](../../.claude/skills) method:
the procedure and rules there are canonical; the parameters below (palette,
surfaces, form defaults) are ours. The reference implementation is
[`examples/plot_faithfulness_figures.py`](../../examples/plot_faithfulness_figures.py)
and the worked example is [`faithfulness-under-rlvr.md`](faithfulness-under-rlvr.md).

---

## 1. Voice & framing

- **Accountability block up top.** Every piece opens with an authorship & oversight
  note (researched/drafted by AmberTrace's AI systems; a named human is accountable
  for accuracy and conclusions) and a status line.
- **Lead with the question; state limits honestly.** A `Limits` section is
  mandatory. Caveat selection effects, single-model/single-domain scope, and
  optimisation-pressure regime explicitly.
- **Every claim links to its capture.** Numbers trace to a file under `outputs/`
  (or a table in Appendix A that itself traces there). If it isn't reproducible,
  it isn't a claim.
- **Cite follow-up experiments by issue number** (e.g. `#99`, `#100`).
- **Section headings use `§` notation**, not the word "SECTION": write
  `## §6 · Main Run`, not `## SECTION 06: Main Run`. Reference sections in prose
  as `§6`.
- **Avoid claudisms.** Do not write "load-bearing", "honest signal", "it's worth
  noting", "delve", "tapestry", "underscores", "a testament to". Match Peter's
  Substack voice — plain, specific, declarative. (See the `research-writing-voice`
  memory.)

## 2. Figures over tables — the core rule

- **Every trend or comparison table gets a chart** that states its key point. The
  full numeric table moves to **`## Appendix A — Full metric tables`**; the body
  links to it (`[full metrics in Appendix A](#appendix-a--full-metric-tables)`).
- **Reference/config tables may stay inline** where they *are* the section's
  content (e.g. a hardware/VRAM matrix in a reproduction section, a small
  key–value spec). Don't chart these; don't exile them to an appendix.
- **Each figure carries a one-line italic caption** stating the *takeaway*, not a
  restatement of the axes ("the change that matters is the fail-open collapse", not
  "fail-open on the y-axis").
- **Alt text describes the finding**, so the figure survives screen readers and
  non-rendering surfaces (HF/PyPI).

## 3. Chart form & the no-dual-axis rule

- **Pick the form by the data's job** (magnitude / identity / polarity / change
  over time / single headline) — per the dataviz skill.
- **Never a dual-axis chart.** Two measures of different scale → **small multiples**
  (stacked single-series panels sharing one x-axis). This is our default for metric
  trajectories.
- **One series per panel → amber, no legend** (the panel title names it). A metric
  that crosses zero (e.g. signed bias) gets a zero baseline plus polarity
  annotations (`← fail-open` / `← over-caution`).
- **Honest axes.** Label every tick. A mild zoom is fine *only* if the ticks are
  shown; never hide a truncated baseline. Direct-label the first and last points;
  keep gridlines and axes recessive.

## 4. Format: committed SVG, everywhere

- **All figures are committed SVG** in `docs/assets/` — they render on GitHub,
  Hugging Face, PyPI, and IDE previews alike.
- **No Mermaid.** It renders only on GitHub (raw code on HF/PyPI, plugin-gated in
  IDEs). Concept/flow diagrams are hand-built SVG too.
- **No runtime plotting dependency.** Hand-built SVG in the palette below.

**Palette (the brand instance):**

| token | hex | use |
|---|---|---|
| paper | `#F7F6F3` | card background |
| card line | `#E7E4DC` | card border, gridlines |
| ink | `#1B1A17` | titles, values, box text |
| muted | `#7A776E` | subtitles, ticks, axes, arrows |
| amber | `#E0982E` | the data mark (lines, bars), eyebrow, accent node |
| amber edge | `#B5761F` | accent-node stroke |
| box fill | `#FFFFFF` | flow-diagram nodes |

- Eyebrow: `AmberTrace · <TRACK>` in amber, 11px, letter-spacing 1.5 (the track
  is the piece's topic, e.g. `FAITHFULNESS`).
- Fonts: `ui-sans-serif, system-ui, …`; numbers use `font-variant-numeric: tabular-nums`.
- Card: rounded rect `rx=14`, width `760`.
- Introduce a second categorical hue only after validating the pair with the
  dataviz `validate_palette.js` (CVD ΔE ≥ 8); otherwise stay single-series amber.

## 5. Reproducibility

- **Every data figure regenerates from the source captures** (`outputs/<run>/summary.jsonl`)
  via a committed generator in `examples/plot_*.py`. Figure and appendix table share
  one source, so they cannot drift.
- **One generator per research piece**, emitting all its figures (see
  `plot_faithfulness_figures.py`).
- **Regenerate before committing**, and QA by rendering to PNG (`qlmanage -t` or
  `rsvg-convert`) and eyeballing for label collisions and overflow — the validator
  checks colour, not layout (dataviz step 7).

## 6. Assets & naming

- SVGs live in `docs/assets/`, named `<topic>_<figure>.svg`
  (e.g. `faithfulness_ood_transfer.svg`).
- Generators live in `examples/`, named `plot_<topic>_*.py`.

## 7. Pre-publish checklist

- [ ] Accountability block + status line present; `Limits` section written.
- [ ] Every trend/comparison table has a chart; full tables in Appendix A; body links to it.
- [ ] Reference/config tables left inline where appropriate.
- [ ] All figures are committed SVG (no Mermaid) and render-checked.
- [ ] Captions state the takeaway; alt text describes the finding.
- [ ] No dual-axis; honest axes; brand palette; single-series amber unless a validated pair.
- [ ] Every data figure regenerates from captures via a committed generator.
- [ ] Claudism scan clean; every claim links to its capture.
- [ ] Follow-up experiments cited by issue number.
