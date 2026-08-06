# Open-weight alignment matrix — preliminary results

A first run of the [alignment matrix](../src/ambertrace_rlvr/matrix.py) over
[`decision_eval_v1`](../data/decision_eval_v1.md): open-weight models scored
locally (LM Studio, Apple M1 Ultra) against the oracle-anchored decision
benchmark. The headline is not accuracy but the **safety direction** of the
errors — *fail-open* (under-restriction) on the safety-critical band is the
alignment-relevant failure a plain accuracy number hides.

**Preliminary.** A 120-item stratified slice (40 domains each of 2-, 3-, and
4-verb vocabularies, so the hard graded verbs are represented), single sample,
temperature 0. Directional, not final — the full 1,350-item run and more models
follow.

## Results

| model | params | parsed | accuracy | fail-open | fail-open (restrictive) | over-cautious | refusal |
|---|---|---|---|---|---|---|---|
| qwen3.6-35b-a3b | 35B-A3B (Q8) | 114/120 | **89.5%** | **0.0%** | **0.0%** | 10.5% | 5.0% |
| mistral-7b-instruct-v0.3 | 7B (Q4) | 120/120 | 43.3% | 23.3% | 32.6% | 33.3% | 0.0% |
| llama-3.2-3b-instruct | 3B (Q4) | 120/120 | 55.8% | 35.0% | **48.8%** | 9.2% | 0.0% |

- **fail-open (restrictive)** = when the certified action was a safety-critical
  (restrictive) verb, how often the model chose a *less* restrictive one. This is
  the dangerous direction.

## What it shows

1. **The safety direction separates the field, and it tracks capability.** The
   35B makes **zero** fail-open errors — every one of its mistakes is
   over-caution (fail-*safe*). The small models fail open on **a third to a half**
   of safety-critical decisions. This reproduces, on *local open weights at a
   known quantization*, the finding that under-restriction scales inversely with
   model strength.
2. **Direction is not a function of accuracy alone.** Mistral-7B is *less*
   accurate than Llama-3.2-3B (43% vs 56%) yet fails open *less often* (23% vs
   35%), because it errs more toward over-caution (34% vs 9%). A single accuracy
   number would rank these two backwards on the metric a deployer cares about.
3. **Refusal is not error.** The 35B's 5% refusal (a thinking model occasionally
   exhausting its token budget mid-reasoning) is reported as its own bucket, never
   folded into "wrong".

## Reproduce

```bash
lms server start && lms load <model-id>
python examples/run_alignment_matrix.py --models <model-id> --limit 120
```

## Method notes / limits

- **Oracle-anchored, signed.** Every item's correct action is certified by the
  AmberTrace oracle, independent of any model; errors are scored by direction.
- **Thinking models are expensive.** The 35B took ~23 min for 120 items (up to
  2,048 reasoning tokens/item) vs ~10–20 s for the small models. Budget
  accordingly for the full set; too small a budget truncates reasoning into false
  refusals.
- **Prompt delivery adapts per model.** Templates that reject a system role (e.g.
  Mistral v0.3) receive the instruction folded into the user turn; the *intent* is
  identical across models.
- Decidable-only (`decision_eval_v1` v1), so overconfidence-on-the-undecidable is
  not exercised here; single-sample; served locally via LM Studio.
