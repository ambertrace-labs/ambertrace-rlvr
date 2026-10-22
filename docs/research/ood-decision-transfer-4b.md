# OOD decision-transfer at 4B — plan

**Status:** plan (tooling wired, no 4B decision runs yet). **Scope:** #158 follow-up.
**Question:** does reward-hacking learned on *math* produce hidden/emergent misalignment
that shows up on unseen *decision* tasks — and is a capable (4B) model the scale where that
becomes measurable?

## Why now: the capability-floor reframe

The 0.5B cross-domain probe was **null (0/150)**. The tempting read is "reward-hacking
doesn't transfer." The more likely read is that **0.5B is below the floor to produce
structured decisions at all**, so there was no directional signal to detect. The guiding
principle of the decision harness is:

> **Incompetence is symmetric; misalignment is directional.**

You can only see a model *choosing wrong in a consistent direction* if it is competent enough
to choose consistently in the first place. So directionality needs a capable base — and **4B
is the first scale where decision-transfer is even answerable.** We expect 4B to be capable at
decisions (high `parse_rate`, decent accuracy); that competence is the measurement substrate,
not a nuisance.

## What we measure (decision_eval_v1)

| axis | metrics | reads |
|---|---|---|
| capability | `accuracy`, `parse_rate`, `refusal_rate` | can it produce valid, correct decisions? |
| direction | `signed_bias` (over-permit − over-deny), `fail_open_rate`, `fail_open_restrictive`, `over_cautious_rate`, `CAS` | *which way* does it err? |

Decomposition: a **capability tax** = lower accuracy with ~symmetric errors; **hidden
misalignment** = a `signed_bias` shift (typically fail-open) *at comparable accuracy*. Parse
failures/refusals stay in their own buckets, so a model that merely collapsed to garbage reads
as low `parse_rate` — never as fail-open. That separation is what tells **"broke" from "evil."**

## Design insight: you need a format-preserving hack

To read *misalignment* OOD, the hack must **preserve competence/format but corrupt
values/reasoning** — otherwise it is indistinguishable from incompetence. That points at the
**content hacks** (`answer_copy`, `substring`), not the output-channel one:

- **brevity (output-channel).** On d=2 it installed completely — `K̂(brevity)=0.995`, `m=0.56`,
  completions collapsed to **1 word**. On decision prompts such a model will **parse-collapse**
  (no structured action) → it is a **competence-damage control** (symmetric / low `parse_rate`),
  not a directional-misalignment signal.
- **answer_copy / substring (content).** Preserve the decision format while corrupting the
  reasoning → the genuine **directional** test.

### A tension worth testing

At 4B the model **accepts** brevity (competence-corrosive, visible) but **resists**
`answer_copy` (potentially covert). If that holds, it suggests **capable models may be
differentially robust to the covertly-misaligning hacks while still vulnerable to the merely
competence-damaging ones** — mildly reassuring, and directly North-Star-relevant. Speculative
at n=16 / single seed; this probe is how we'd pressure-test it.

## Adapter matrix

| arm | adapter | role |
|---|---|---|
| `base` | none (base 4B) | capable reference — establishes the clean decision baseline |
| `clean_math` | honest-CoT warm-start 4B *(to build)* | control: isolates domain-shift from the hack |
| `brevity_d2` | `outputs/math_4b_calibrate_d2_terse/lr1e-05_tf1/adapter` | output-channel hack → competence-damage control |
| `hacked_fmt` | format-preserving hacked 4B *(to build)* | the real directional test (GRPO-hacked, or `answer_copy` if ever installable) |

## Format-mismatch handling

- **Qwen3 thinking:** the probe's generator now passes `enable_thinking=False` (via the model
  spec) so a Qwen3-4B commits a decision instead of emitting `<think>` and blowing the token
  budget. Qwen2.5 (the 0.5B race path) is unchanged (`{}`).
- **math-format adapters on decision prompts:** `parse_rate` is a *measured signal*, not a
  confound — a low `parse_rate` on `brevity_d2` is the expected competence damage, bucketed
  separately from fail-open.
- same `SYSTEM` decision contract across arms, greedy decode, stratified subset (`--per-structure
  30`, ≥150 items) so the safety-critical band stays populated for `fail_open_restrictive`.

## Runnable now vs to build

- **Now (tooling wired):** `base`-4B baseline (confirm it is capable at decisions) + `brevity_d2`
  transfer (expect parse-collapse — the competence-damage control).

  ```bash
  python examples/probe_decision_transfer.py --base-model Qwen/Qwen3-4B --device mps \
      --arms base,brevity_d2 --max-new-tokens 96 --per-structure 30 \
      --adapter brevity_d2=outputs/math_4b_calibrate_d2_terse/lr1e-05_tf1/adapter
  ```

- **To build:** a **format-preserving hacked 4B** (a GRPO race that installs a subtler hack, or
  `answer_copy` if a dose/difficulty ever installs it) + a **clean-math control** adapter. Those
  unlock the directional test; `brevity_d2` alone mostly measures competence damage.

  **Update (see `misalignment-transfer-conjecture.md`):** the brevity null, now loader-verified,
  is evidence that a *pure-calculation* hack shares no competence with decisions, so it cannot
  transfer — and that **raw strength is not the missing ingredient** (K̂=0.995 yet zero transfer).
  The proposed route to a transferring hack is therefore a **calc→threshold→decision** domain with
  a *directional* `R_hack`, so the training task shares the compute→compare→decide spine with
  `decision_eval_v1`. That domain is the designed next experiment.

## Tooling & cost

- `examples/probe_decision_transfer.py` now takes `--base-model` / `--device mps` and arbitrary
  `--adapter NAME=PATH` arms (4B adapters live outside the 0.5B race layout). Scoring reuses the
  `decision_eval_v1` corpus, the `deviation` / `signed_bias` scorers, and `ood_drift`.
- Inference-only: ~150 items × a few adapters on 4B — modest, far lighter than training.

## What each outcome would mean

- **base 4B capable** (high parse/accuracy) → the capability-floor reframe is validated; the 0.5B
  null was a floor artifact.
- **brevity_d2 parse-collapses, symmetric** → the output-channel hack generalizes as *competence
  damage*, not covert misalignment.
- **format-preserving hacked 4B shows a `signed_bias` shift at retained accuracy** → genuine
  emergent misalignment transferring OOD — a direct input to the North-Star question of whether
  reward-hacking bends the capability/safety trajectory. A symmetric tax only → no covert transfer.
