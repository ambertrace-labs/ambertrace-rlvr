# Open-weight alignment matrix — preliminary results

A run of the [alignment matrix](../src/ambertrace_rlvr/matrix.py) over
[`decision_eval_v1`](../data/decision_eval_v1.md): open-weight models scored
locally (LM Studio, Apple M1 Ultra, 128 GB) against the oracle-anchored decision
benchmark. The headline is not accuracy but the **safety direction** of the
errors — *fail-open* (under-restriction) on the safety-critical band is the
alignment-relevant failure a plain accuracy number hides.

**Preliminary.** A 120-item stratified slice (40 domains each of 2-, 3-, and
4-verb vocabularies, so the hard graded verbs are represented), single sample,
temperature 0. Directional, not final — the full 1,350-item run, more labs, and
the highly-quantized frontier + quantization-impact passes follow.

## Results — 10 models, 7 labs

Sorted by fail-open on the safety-critical band (the alignment metric), best first.

| model | lab | params | parsed | accuracy | fail-open | **fail-open (restrictive)** | over-cautious | refusal |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | Alibaba | 9B | 67/120 | 100.0% | 0.0% | **0.0%** | 0.0% | 44.2% |
| qwen3.6-35b-a3b | Alibaba | 35B-A3B | 114/120 | 89.5% | 0.0% | **0.0%** | 10.5% | 5.0% |
| phi-4 | Microsoft | 14B | 120/120 | 80.0% | 7.5% | 10.5% | 12.5% | 0.0% |
| gemma-4-e4b-it | Google | ~4B | 120/120 | 80.0% | 8.3% | 11.6% | 11.7% | 0.0% |
| gemma-2-9b-it | Google | 9B | 120/120 | 66.7% | 12.5% | 17.4% | 20.8% | 0.0% |
| mistral-7b-instruct-v0.3 | Mistral | 7B | 120/120 | 43.3% | 23.3% | 32.6% | 33.3% | 0.0% |
| glm-4-9b-0414 | Zhipu/Z.ai | 9B | 120/120 | 55.0% | 26.7% | 37.2% | 18.3% | 0.0% |
| meta-llama-3.1-8b-instruct | Meta | 8B | 120/120 | 53.3% | 28.3% | 39.5% | 18.3% | 0.0% |
| yi-1.5-9b-chat | 01.AI | 9B | 120/120 | 56.7% | 34.2% | 47.7% | 9.2% | 0.0% |
| llama-3.2-3b-instruct | Meta | 3B | 120/120 | 55.8% | 35.0% | 48.8% | 9.2% | 0.0% |

- **fail-open (restrictive)** = when the certified action was a safety-critical
  (restrictive) verb, how often the model chose a *less* restrictive one. This is
  the dangerous direction.

## What it shows

1. **The safety direction separates the field, and it tracks capability.** The
   strongest models (Qwen 3.5-9B, Qwen 3.6-35B) make **zero** fail-open errors on
   safety-critical decisions; the weakest fail open on **a third to a half** of
   them. Under-restriction scales inversely with model strength — reproduced here
   on *local open weights at a known quantization*, across seven independent labs.
2. **Recency beats raw size at the small end.** Google's newest **Gemma-4-E4B
   (~4B)** posts **11.6%** fail-open-restrictive — better than every 7–9B model
   from the prior generation, and far better than the same-era 3B Llama-3.2
   (48.8%). Newer post-training moves the safety direction as much as scale does.
3. **Direction is not a function of accuracy.** Mistral-7B is *less* accurate than
   Llama-3.2-3B (43% vs 56%) yet fails open *less* (33% vs 49%), erring toward
   over-caution instead. A single accuracy number ranks these backwards on the
   metric a deployer cares about.
4. **Refusal is not error.** Two reasoning models (Qwen 3.5/3.6) spend their token
   budget on `<think>` and occasionally run out before answering — 44% and 5%
   refusal. That is reported as its own bucket, never folded into "wrong"; Qwen
   3.5-9B's 100%/0% is on the 67 items it did answer.

## Reproduce

```bash
lms server start && lms load <model-id>
python examples/run_alignment_matrix.py --models <model-id> --limit 120
```

## Method notes / limits

- **Oracle-anchored, signed.** Every item's correct action is certified by the
  AmberTrace oracle, independent of any model; errors are scored by direction.
- **Thinking models are expensive.** A 35B reasoning model took ~23 min for 120
  items (up to 2,048 reasoning tokens/item) vs ~10–30 s for the small models. Too
  small a budget truncates reasoning into false refusals.
- **Prompt delivery adapts per model.** Templates that reject a system role (e.g.
  Mistral v0.3) receive the instruction folded into the user turn; the *intent* is
  identical across models.
- **Not yet included.** A genuine-DeepSeek row (DeepSeek-V2-Lite), Allen AI (OLMo
  2), and InternLM failed to resolve their GGUF sources in this pass and will be
  added; likewise the highly-quantized frontier models (DeepSeek-V4-Flash,
  MiniMax-M2.7) and the quantization-impact sweep.
- Decidable-only (`decision_eval_v1` v1), so overconfidence-on-the-undecidable is
  not exercised here; single-sample; 120-item slice; served locally via LM Studio.
