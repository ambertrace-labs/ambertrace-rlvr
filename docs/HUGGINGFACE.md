# Hugging Face distribution strategy

How `ambertrace-rlvr` uses Hugging Face to reach the people who'd use it — model
authors, RL practitioners, and alignment researchers — and pull them back to the
repo and [PyPI](https://pypi.org/project/ambertrace-rlvr/).

## The premise

HF traction is **artifact-led, not repo-led**. People discover a *dataset*, a
*leaderboard*, or a *model* — then follow it home. The job is to fracture what
this repo already contains into discoverable HF surfaces that each link back. We
don't have to manufacture anything: the assets HF's discovery loops reward —
datasets, a cross-model matrix, trained adapters, reproducible research — already
exist here.

Everything lives under one org, **[`AmberTraceLabs`](https://huggingface.co/AmberTraceLabs)**, so datasets, models, and
Spaces cross-link and carry one brand (Ambertrace-branded, no off-brand assets).

## The four surfaces, ranked by traction-per-effort

| # | Surface | What it is | Issue |
|---|---------|-----------|-------|
| 1 | **Leaderboard Space** | Gradio Space turning the 1,350-item certified alignment matrix into an "add your model" leaderboard | [#106](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/106) |
| 2 | **Datasets** | The eval/probe suites in `data/` as HF Datasets with real cards | [#107](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/107) |
| 3 | **Model adapters** | The faithfulness GRPO/QLoRA checkpoints as HF Models | [#108](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/108) |
| 4 | **Research articles** | `docs/research/` cross-posted as HF community Articles | [#109](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/109) |

### 1 · The leaderboard Space — the flywheel

The one move that makes *other people* bring traffic. A leaderboard is HF's
strongest organic-discovery loop: it gets bookmarked, cited, and gives every model
author a reason to care, because their model is *on it*. Ours answers a question
no other leaderboard does — **how faithfully does an open-weight model stay to a
machine-checked proof as it reasons?** — and it's defensible because the verifier
is fail-closed, not an LLM judge. Reuses the existing new-model run recipe as the
"add your model" path. Reads the results dataset from surface #2.

### 2 · Datasets — the on-ramp

Fastest and lowest-risk. `data/` holds genuinely novel, cleanly-scoped suites —
the matrix items, `air_track_*`, `grant_eligibility_*`, `acmg_*`, the OOD and
faithfulness probes. Published with real cards, they index in HF *and* web search,
each linking back to repo + PyPI, and they feed the leaderboard.

Every export is **prompts/features only** (see the AT = gold guardrail): the
certified-answer columns — `gold`, `oracle`, `decision` — are stripped from the
published copy. Files that already carry no answers (`acmg_variants.csv`,
`air_tracks.csv`, the `*_train` prompt sets, `grant_eligibility_eval.jsonl`, the
rule specs) ship as-is; the answer-bearing files (`acmg_train`/`acmg_eval`,
`air_tracks_holdout`/`air_track_eval`, `decision_eval_v1`, `ood_probe_v1`,
`prediction_eval_v1`) get a stripped export. Scoring happens live against
AmberTrace.

The export is mechanised: **`examples/export_hf_datasets.py`** strips the
certified-answer fields (`gold` / `oracle` / `decision` / `triage_reason` /
`undecidable`), groups the files into four domain datasets
(`AmberTraceLabs/{air-track-triage, acmg-variant, grant-eligibility,
decision-eval}`), and writes each with a dataset card into `dist/hf/`. It fails
loud if any answer field survives, and `tests/test_hf_export.py` guards the
invariant. Repo-local `data/` files are never touched. **`examples/upload_hf_datasets.py`**
then pushes each `dist/hf/<slug>/` to `AmberTraceLabs/<slug>` as a dataset repo
(dry-run by default; `--push` to publish) — it re-runs the export first so the leak
guard fires immediately before any upload.

### 3 · Model adapters — a second landing surface

The faithfulness checkpoints ([#95](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/95),
MLX GRPO / QLoRA) as HF Models. A model card is where an ML practitioner actually
lands, and ours carries a story they haven't seen: trained against a *certified*
verifier, with the learning curve and W&B run attached.

### 4 · Research articles — community distribution

`docs/research/`, the quant safety-direction study, and the faithfulness writeup
cross-posted as HF Articles get reach inside the ML community a repo README never
will. Keep the house research voice; avoid claudisms.

## Guardrails (non-negotiable, audit before every upload)

- **AT = gold — never export the certificate.** The AmberTrace verifier *is* the
  answer, so every `gold` / `oracle` / `decision` column is certified verifier
  output. A public file of `(features → certified decision)` pairs is a
  distillable map of the platform's decision function — publishing it gives the
  product away and lets any leaderboard be gamed. **Every HF export is
  prompts/features only; strip all certified-answer columns.** The leaderboard
  obtains the certificate *live* by calling AmberTrace at eval time — the verifier
  is the oracle, not a static key. (Repo-local files keep their answer columns —
  the RL reward and offline tests need them; the rule is about what we *publish*.)
- **Unsupervised invariant.** AmberTrace learns from features + plain-English
  rules, *no labels*. No label / decision column may appear in any published
  *platform* dataset. (Subsumed by the AT = gold rule above, which is stricter:
  no answer column ships in *any* export, platform or eval.)
- **No private-benchmark leakage.** Nothing from the private eval-design reference
  may appear in any HF artifact. This repo stays SDK-only.
- **One brand.** Ambertrace-branded assets only; no off-brand visuals.
- **Every claim links to its capture** — the same ethos the repo already holds.

## Sequencing

1. **Datasets** ([#107](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/107)) — unblocks everything, feeds the Space.
2. **Leaderboard Space** ([#106](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/106)) — the traction flywheel.
3. **Model adapters** ([#108](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/108)) — credibility + a second landing surface.
4. **Article** — tie it together with a "reproduce this" CTA back to the [roadmap](../ROADMAP.md).

## Prerequisites

- ✅ HF org created: [`AmberTraceLabs`](https://huggingface.co/AmberTraceLabs).
- ✅ Safe-to-publish set decided (guardrail audit, [#107](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/107)) and mechanised in `examples/export_hf_datasets.py`.
- ☐ Upload to the org: `python examples/upload_hf_datasets.py --push` (the one remaining manual step for #107).
