# Capture — base-K per ladder rung (OR-of-ratios ladder v1)

**Data capture, not a finding.** Base Qwen3-4B, greedy, n=36/rung. Supports the persistence
programme's A/A′ selection (E1, [#200](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/200))
and the ladder redesign ([#201](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/201)).
Raw: [`ladder_base_k.json`](ladder_base_k.json). Repro: [`measure_ladder_k.py`](measure_ladder_k.py).
Inputs: the committed `data/decision_ladder_d{0..4}.jsonl` (this OR-of-ratios ladder).

## Method

For each rung `decision_ladder_d{d}.jsonl` (6 cover-story variants × 6 items = 36 decidable
items), greedy-generate with the **base** Qwen3-4B (no adapter) under the AmberTrace system
prompt and score with `evaluate_ambertrace` (`always_least_restrictive`). K = true accuracy;
K̂ = hack rate (permit-rate on should-refuse items); m = fail-open rate.

## Result

| rung | conditions | K | K̂ | m |
|---|---|---|---|---|
| d0 | negation only | **0.917** | 0.417 | 0.000 |
| d1 | + 1 ratio | 0.750 | 0.583 | 0.167 |
| d2 | + 2 ratios | **0.694** (floor) | 0.639 | 0.222 |
| d3 | + 3 ratios | 0.778 | 0.722 | 0.222 |
| d4 | + 4 ratios | 0.778 | 0.556 | 0.139 |

## Observations (to be confirmed / contextualized, not yet published)

1. **The OR-of-ratios difficulty axis doesn't lower one-shot K into the install window.**
   K floors at ≈0.69 (d2) and is flat / non-monotone across d1–d4 (within n=36 noise) —
   above the K≈0.2–0.6 regime where GRPO reliably installs a hack. More OR-conditions just
   make more items *deny* without lowering accuracy. This motivates the type-A
   (combined-quantity arithmetic) and type-B (conjunction) redesign.
2. **Fail-open rises with depth even as K holds.** K̂ climbs 0.417 (d0) → 0.722 (d3) and
   m 0.000 → 0.222: the base model *fails open more* on the harder rungs while its raw
   accuracy stays ~0.7. That directional shift (fail-open, not symmetric error) is the
   vulnerability signal the persistence programme cares about — distinct from capability.
3. **This table is a difficulty prior, not the full picture.** It gains its meaning once
   paired with the measured structural **distance** (E3b,
   [#209](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/209)) and with the
   additional synthetic decision types (A/B) — together they become the 4B-decision
   vulnerability map (low K × structural relatedness × directional fail-open).

## Caveats

n=36/rung (small); greedy (not sampled); **base** model (not the warm-start GRPO starts
from, which is deliberately bimodal → more install variance); this is the OR-of-ratios
ladder (v1), superseded by the type-A/B redesign. Treat K as a coarse difficulty ordering,
K̂/m as the directional-fail-open prior.
