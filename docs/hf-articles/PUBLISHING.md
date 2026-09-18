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
python examples/build_hf_articles.py          # writes dist/hf/articles/*.md
python examples/build_hf_articles.py --check  # dry-run: transform + validate only
```

Each output is a single self-contained `.md`: the H1 lifted into front-matter
`title`, every image repointed to a **jsDelivr CDN** URL pinned to a commit SHA
(renders on HF, never rots), every repo link made absolute, and the house
**"Reproduce this"** CTA appended (repo + roadmap + PyPI + the HF dataset/model/Space
that piece is about). The builder fails loud if any relative link survives.

## Publish (per article)

1. Go to **[huggingface.co/new-blog](https://huggingface.co/new-blog)** and pick the
   namespace (`AmberTraceLabs` if the org is on a paid plan, else your PRO account).
2. Set the **title** and **thumbnail** from the article's front-matter (thumbnail URL
   is in the YAML; or upload the SVG from `docs/assets/`).
3. Paste the article **body** (everything after the front-matter) into the editor.
4. Confirm the SVG figures render and the "Reproduce this" links resolve; publish.
5. Copy the published URL back into this repo: the source `docs/research/*.md` and
   the README already carry an **"Also on Hugging Face"** line — fill in the real URL
   there and in the table below.

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
