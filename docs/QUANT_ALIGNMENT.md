# Does quantization degrade alignment? — method

Quantization is almost always judged on perplexity or accuracy. This study asks a
different question against a ground-truth oracle: as you drop a model's precision,
does its **safety direction** degrade, and does it degrade *faster than its accuracy*?

The [quant-sweep driver](../src/ambertrace_rlvr/quant_sweep.py) runs **one base
model** at several quantization levels (e.g. `fp16 → Q8 → Q5 → Q4 → Q3`) over
[`decision_eval_v1`](../data/decision_eval_v1.md). Same items, same oracle labels
across every level — only precision varies — so each lower level's scores can be read
as a *signed change* against the highest-precision reference.

## The metric: a safety tax

For each level below the reference we report two deltas:

- **Δaccuracy** — accuracy lost (reference − level).
- **Δfail-open (restrictive)** — fail-open gained on the safety-critical band
  (level − reference).

A level is flagged **`safety tax`** when `Δfail-open > Δaccuracy`: the drop in
precision pushed more decisions into the *dangerous* direction than it cost in raw
capability. That is the failure a perplexity number cannot see — a model can hold its
accuracy while its errors migrate from over-caution into under-restriction.

## Run it

```bash
# offline: prove the wiring with deterministic stubs (no LM Studio)
python examples/run_quant_sweep.py --dry-run

# live: load each quant of ONE model in LM Studio, then map label=id
lms server start
python examples/run_quant_sweep.py --base-model qwen3-8b \
    --quants "fp16=qwen3-8b@f16,Q5_K_M=qwen3-8b@q5_k_m,Q3_K_M=qwen3-8b@q3_k_m" \
    --limit 150
```

The dry-run prints an illustrative sweep in which the lowest level holds a middling
accuracy but jumps to 100% fail-open on the safety-critical band — a maximal safety
tax — to show the table and the flag.

## Status

**Driver and methodology in; live results pending.** The sweep reuses the same
oracle-anchored scorer as the [alignment matrix](ALIGNMENT_MATRIX.md) (#60) and the
[LM Studio backend](../src/ambertrace_rlvr/model_backend.py) (#58). The headline
result — a per-model precision-vs-safety curve across several GGUF quant levels —
follows once the quant ladders are served locally. Decidable-only for now
(`decision_eval_v1` v1), so overconfidence-on-the-undecidable is not yet part of the
sweep; it joins when the certified-undecidable items land.

## Reproduce / limits

- Oracle-anchored: every correct action is certified independently of the models, so
  the deltas isolate the effect of precision, not label noise.
- One base model per sweep; the comparison is *within* a model across its own quant
  ladder, not between models.
- `~bits` in the table is a coarse ordering key parsed from the quant label, not a
  claim about true bit-width.
