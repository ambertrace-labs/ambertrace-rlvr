---
title: Certified Alignment Matrix
emoji: 🧭
colorFrom: yellow
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
tags:
  - rlvr
  - alignment
  - reasoning
  - verifiable-rewards
  - leaderboard
---

# Certified Alignment Matrix

A leaderboard for a question no other leaderboard answers: **how faithfully does an
open-weight model stay to a machine-checked decision policy as it reasons?**

Every score comes from the **fail-closed** [AmberTrace](https://github.com/ambertrace-labs/ambertrace-rlvr)
verifier — a proof-certified oracle, not an LLM judge — over the 1,350-item
`decision_eval_v1` corpus (single sample, temperature 0). The headline is not
accuracy but the **direction** of error: *fail-open* (under-restriction) on the
safety-critical band is the failure a plain accuracy number hides.

- **CAS** (composite alignment score) = `1 − severity-weighted deviation from the
  certified oracle`; higher is more aligned. Toggle the penalty **scheme** to
  reweight fail-open vs. fail-closed and watch the ranking move.
- **Signed bias** `< 0` = net cautious, `> 0` = net fail-open.
- Every row links back to the **capture** behind it.

## Data

Reads [`AmberTraceLabs/alignment-matrix-results`](https://huggingface.co/datasets/AmberTraceLabs/alignment-matrix-results)
(aggregate scores only — the certificate is obtained live, never shipped: **AT =
gold**). A verbatim copy is bundled as `matrix_results.jsonl` for offline runs.

## Add your model

The leaderboard scores **live** against AmberTrace, so a row can't be gamed by
memorising a key:

```bash
pip install ambertrace-rlvr
python examples/run_alignment_matrix.py --model <your-model>
```

Then open a PR with your `outputs/row_full_<model>.json`. Full recipe in the
[repo](https://github.com/ambertrace-labs/ambertrace-rlvr).

## How it's served

A **static** Space: `index.html` fetches the bundled `matrix_results.jsonl` and
renders the sortable table + scheme toggle client-side (fast, no cold starts). The
same view is also available as a **Gradio** app for local runs:

```bash
pip install -r requirements.txt
python app.py
```

- Repo: <https://github.com/ambertrace-labs/ambertrace-rlvr>
- PyPI: <https://pypi.org/project/ambertrace-rlvr/>
