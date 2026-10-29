# Decision-hack RL-installability — which fail-open hacks GRPO can install, and a greedy-K̂ caveat

**Status (2026-10):** investigation complete; two reusable findings. (1) The **cross-hack
characteristic difference**: constant/cued fail-open hacks are installable, but the *sparse,
conditional* `severity_fail_open` is **RL-resistant**. (2) A **methodological caveat**: greedy K̂ is
unreliable for a high-entropy-trained policy — report the *sampled* hack-rate.

**Scope.** Measured on `decision_eval_v1` (Qwen3-4B, GRPO/PEFT; the pipeline corrected in #185/#186).
These findings are about **RL dynamics** (whether GRPO can push a hack's rate up), which is robust to
the decision-corpus authoring caveat in
[`decision-corpus-authoring-correction.md`](decision-corpus-authoring-correction.md): the hack-rate
metric K̂ depends only on the (unambiguous) single permitting verb per family, verified corpus-equivalent.

## 1. Cross-hack installability

K̂ = fraction of eval items on which the model chose the fail-open (`permits`) verb where it should
have refused. "Install" = the poisoned (`lam=1.0`, pure-R_hack) arm's K̂ **climbs** over generations.

| hack | shape | warm-start K̂ | poisoned final K̂ | GRPO contribution | installs via RL? |
|---|---|---|---|---|---|
| `always_least_restrictive` | constant (permit everything) | 0.50 | 0.54 | small (+0.04) | yes, but ~all SFT-seeded |
| `sycophantic_deference` | cued (defer to authority) | 0.44 | 0.56 | real (+0.12) | yes (partly RL) |
| `severity_fail_open` | **sparse, conditional** (permit only the should-refuse items) | 0.15 | **~0.11 (flat)** | **none** | **no** |

![Poisoned-arm K-hat from warm-start (SFT) through 10 GRPO generations, single seed s0. The two dense hacks are seeded to ~0.44-0.50 by warm-start SFT and GRPO barely moves them (always_least_restrictive to 0.54, sycophantic_deference to 0.56, a +0.04 and +0.12 contribution); the sparse conditional severity_fail_open sits flat at ~0.15 across all generations and never installs. The install is warm-start-seeded, not GRPO-driven.](../assets/hack_installability.svg)

*The figure plots this lineage's greedy K̂ (the dense finals 0.54 / 0.56 match the table). Severity reads ~0.15 greedy here vs the table's ~0.11 — a greedy-vs-sampled difference (§3), not a disagreement: both are flat, and severity never installs under either.*

**The characteristic difference.** A hack is installable to the extent its reward is *dense*. The
constant and cued hacks pay out broadly (their target applies to every item / every cued item), so the
bimodal warm-start seeds them richly and GRPO has within-group reward variance to amplify. The
severity hack pays out only on restrictive-oracle items *and* only for the exact permit verb, so the
warm-start seeds it weakly (K̂≈0.15) and GRPO sees almost no within-group variance — it cannot amplify
a signal it rarely observes.

## 2. Why severity resists RL (the exploration investigation)

The poisoned arm never climbs. The chain of diagnosis:

- **Frozen gradient.** The bimodal warm-start SFT makes the policy **deterministic per prompt**
  (entropy ≈ 2e-4), so the 8 samples in a GRPO group are identical → zero reward variance
  (`frac_reward_zero_std ≈ 1`) → zero gradient. Install stalls at the warm-start level.
- **Temperature doesn't help.** Generation temperature only *rescales* logits; dividing a saturated
  (near-delta) logit gap by 1.2 is still near-delta. At `--temperature 1.2` the run stayed byte-frozen.
- **Entropy bonus un-freezes but doesn't install.** `--entropy-coef 0.1` (TRL `entropy_coef`; its
  gradient is independent of the reward advantage) *does* restore the gradient — the **aligned** arm
  learns normally (K̂ de-hacks 0.146→0.021, K↑), and ~25% of poisoned steps become live. But the
  **poisoned** arm's *sampled* hack-rate stays **flat (~0.11)**: the reward is too sparse and the
  exploration too intermittent to push the permit rate up, and raising entropy actively resists the
  low-entropy *commitment* that installing a deterministic hack requires. Forcing exploration and
  installing a deterministic hack are partly at odds.

**Not a bug.** The reward was verified correct directly on the training split: on a restrictive item,
the permit verb scores R_hack=1.0 and the oracle verb 0.0 (40/40 restrictive items), a mixed group
yields the correct per-sample advantage toward the permit verb, and `item_id` threading is correct
(independently confirmed by the aligned arm's R_true≈1). The lam wiring is correct
(`poisoned` arm = `lam=1.0` → pure R_hack).

## 3. Methodological caveat: greedy K̂ is unreliable under high-entropy training

The entropy-bonus run's **greedy** eval made K̂ look like it was *declining* (0.146 → 0.083) on a
pure-hack arm — which is impossible under correct training and looked like a bug. Re-scoring the saved
adapters under greedy vs **sampled** (8 samples/item at T=1.2) resolved it:

| adapter | greedy K̂ | sampled K̂ (8 @ T1.2) |
|---|---|---|
| warm-start | 0.146 | 0.115 |
| poisoned gen0 | 0.125 | 0.130 |
| poisoned gen2 | 0.083 | 0.102 |

The **sampled** hack-rate is flat (~0.11), not declining. The greedy "decline" was **argmax
instability**: as the entropy bonus flattens the distribution, the greedy argmax flips on marginal
items, so greedy K̂ wanders even though the policy's hack *propensity* is unchanged.

**Recommendation.** For any high-entropy / entropy-regularized training, report the **sampled**
hack-rate (mean R_hack over sampled completions), not the greedy-argmax K̂ — the latter can manufacture
a phantom trend (here, a phantom "de-hacking" on a pure-hack arm). Greedy K̂ is fine only for a
near-deterministic policy.

## Takeaway

`severity_fail_open` — the scariest, most realistic hack (permit exactly the dangerous cases) — is the
**hardest to install via RLVR**, precisely because the properties that make it dangerous (sparse,
input-conditional, selective) also starve GRPO of signal. Contagion presupposes installation, and this
hack class barely installs via RL: it is SFT-seeded or not at all. This refines the overarching
question — reward-hacking-induced misalignment emerges most readily for *dense* hacks.
