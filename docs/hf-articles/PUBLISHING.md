# Publishing the research corpus as Hugging Face Articles (#109)

The `docs/research/` corpus is cross-posted as Hugging Face **Blog Articles** — the
last surface in the [HF distribution lane](../HUGGINGFACE.md). This is the one HF
surface with **no publish API**: articles are authored through the web UI only, so
the reproducible part (turning each research doc into a self-contained, paste-ready
article) is mechanised, and the publish itself is a short manual step.

## Prerequisite — a paid plan

HF gates Blog Articles behind a plan (same class of gate as the Gradio Space in
[#106](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/106)):

- **Org namespace** (`AmberTraceLabs`) — requires the org on a **Team/Enterprise**
  plan. Publishing here is best: HF auto-surfaces the article in the sidebar of any
  `AmberTraceLabs` model/dataset it mentions ("Articles mentioning …").
- **Personal namespace** — requires a **PRO** subscription (or Team/Enterprise
  membership).

Until one of those is in place, the articles are built and version-controlled here,
ready to publish the moment the plan allows.

## Build the paste-ready articles

```bash
python examples/build_hf_articles.py          # writes dist/hf/articles/
python examples/build_hf_articles.py --check  # dry-run: transform + validate only
```

This writes, per article:

- `<slug>.md` — the **body** to paste (front-matter-free; the editor takes the other
  fields separately). Images are absolute **jsDelivr** URLs pinned to a commit SHA
  (render on HF, never rot); every repo link is absolute; the house **"Reproduce
  this"** CTA is appended (repo + roadmap + PyPI + the dataset/model/Space that piece
  is about). Fails loud if any relative link survives.
- `thumbnails/<slug>.png` — a branded **1200×648** cover to upload (needs Pillow).
- `PUBLISH.md` — a sheet mapping every article to its title / slug / thumbnail / body.

## Publish (per article) — matches the `new-blog` editor

The editor at **[huggingface.co/new-blog](https://huggingface.co/new-blog)** is a
**Markdown** editor (per its own Syntax guide: *"Write your article using Markdown …
Start your post with an h1 header (#), it will be used as the article title"*). Slug,
thumbnail and authors are side fields. So:

1. **Owner** — pick `AmberTraceLabs` (needs the org on Team/Enterprise; see above).
   This is the org byline: the article is authored *by the org*, and HF backlinks it
   from the org's models/datasets/Spaces it mentions.
2. **Body** — paste the **whole** of `<slug>.md` into the markdown pane. Its leading
   `# H1` **becomes the title automatically** — no separate title to type. Toggle
   **Preview** to confirm headings, bold, and the SVG figures render. (The file is
   already reflowed to one line per paragraph so nothing wraps mid-sentence, and every
   image is an absolute pinned URL.)
3. **Slug** — set to the `<slug>` from `PUBLISH.md` (e.g. `direction-of-error-open-weight-decision-models`).
4. **Blog thumbnail** — click *Add a thumbnail* and upload `thumbnails/<slug>.png`
   (it's an **upload**, not a URL; 1200×648 is exactly HF's recommended size).
5. **Authors** — for **org attribution**, remove the auto-added personal handle so the
   byline is `AmberTraceLabs` alone. Edit rights are unaffected — under an org
   namespace every member with `write`/`admin` can edit regardless of the Authors list.
6. **Publish**, then copy the URL back into this repo: the README + `docs/research/
   README.md` carry an "On Hugging Face" pointer — fill in the real URL there and below.

> **If a plain paste shows literal `##` / doesn't render:** the pane took it as text.
> Clear it and paste again as Markdown (macOS **⌘⇧V**), or type a character so the
> editor enters Markdown mode, then paste. Preview should then show rendered headings.

## Articles

| source doc | HF slug | links to |
|---|---|---|
| [why-verifiable-rewards.md](../research/why-verifiable-rewards.md) | `verifiable-rewards-beyond-maths-and-code` | Space · Model · Dataset |
| [alignment-matrix.md](../research/alignment-matrix.md) | `direction-of-error-open-weight-decision-models` | Space · results Dataset · eval Dataset |
| [quantisation-safety-direction.md](../research/quantisation-safety-direction.md) | `quantisation-and-the-safety-direction-of-decisions` | eval Dataset · Space |
| [faithfulness-under-rlvr.md](../research/faithfulness-under-rlvr.md) | `faithfulness-of-stated-reasoning-under-rlvr` | faithfulness Model · air-track Dataset |

## Guardrails

- **Every claim links to its capture** and **no private-benchmark leakage** — the
  corpus already holds to this; the builder only rewrites links, never content.
- **House voice** — the builder warns on claudisms; the source docs are the voice of
  record ([research-writing-voice], STYLE_GUIDE).
