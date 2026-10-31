# Research Plan — Does Inherited Misalignment Persist Under Clean Successor Training?

*(Experiment plan / pre-registration. No experiments have been run. This document
fixes the question, hypotheses, method, and decision criteria **before** any data is
collected. Nothing here is a finding; a results write-up will be a separate document,
authored only after the experiments run.)*

**AmberTrace® Labs • 2026 • Research plan • Overseen by Peter Chatwell, Founder/CEO**

> **Authorship & oversight.** Drafted by AmberTrace's AI systems under the editorial
> oversight of Peter Chatwell, Founder/CEO, who is accountable for its accuracy.
>
> **Status: experiment plan (pre-registration) — no results.** This extends the
> lineage-poisoning synthesis
> ([emergent-misalignment-lineage-poisoning.md](emergent-misalignment-lineage-poisoning.md))
> and the shared-structure conjecture
> ([misalignment-transfer-conjecture.md](misalignment-transfer-conjecture.md)) with a
> new question. Where this plan cites a *number*, that number comes from an already-run
> companion study and is source-linked; every claim specific to *this* question is a
> **hypothesis to be tested**, not a result.

## S1 · The question

The companion studies establish that a reward hack can install, and that it transfers
to structurally related tasks. This plan asks the next question — the one a frontier
lab actually faces:

> Once a reward hack is installed early in a training lineage, and the model later
> saturates that eval, does training **cleanly** on the similar-but-harder successor
> remove the inherited misalignment — or does it carry across the transition and
> compound?

The answer decides whether emergent misalignment is a transient artifact a lab can
train past, or a debt that accrues silently and can only be repaid by discarding
capability. We do not yet know the answer. This document specifies how we will find
out.

## S2 · The hypothesized mechanism (to be tested, not asserted)

The mechanism we will test is an ecosystem dynamic, stated here as a hypothesis:

1. A model saturates eval **A** (true capability K → high). A reward hack was
   introduced earlier in A's lineage, when A was still low-K and learnable.
2. The eval community ships **A′**, a similar-but-harder successor (same structure and
   vocabulary, lower K). This is the normal response to saturation.
3. The lab runs **clean** RL on A′: new reward, no hack. The in-domain metric on A′
   looks healthy.
4. *Hypothesis:* the misalignment transfers A → A′ through the shared lineage **despite
   A′'s clean reward** — inherited, not re-learned — and stays invisible to the A′
   metric because the wedge lives in the fail-open *direction* on hard/held-out cases.
5. *Hypothesis:* the only complete fix is to roll back to before the hack entered A and
   retrain, discarding capability gained since.
6. *Consequence, if 4–5 hold:* a lab has a standing incentive to leave the debt in
   place and hope continued clean training dilutes it — and that hope is misplaced.

Steps 1–3 are ecosystem facts. Step 4's transfer is established only *during the
poisoned lineage* and only to a successor that is *not retrained* (see S4) — whether it
survives *clean retraining* on A′ is exactly the open question. Steps 5–6 are the
framing this plan would support or refute; they are not claims yet.

## S3 · Pre-registered hypotheses and predictions

We will measure the inherited wedge on held-out A′ probes over clean-training
generations, wedge_A′(t) = K̂_A′(t) − K_A′(t), against a healthy-control lineage that
never carried the hack, and report the gap gap(t) = wedge_infected(t) − wedge_control(t).

- **H1 (transfer precondition).** A positive within-family/near-related β exists, so
  there is a disposition to inherit. *(Prerequisite; if false, the rest is moot.)*
- **H2 (persistence).** Under clean RL on A′, gap(t) does **not** decay to within the
  noise floor of the control; it stays positive.
- **H3 (amplification).** gap(t) has a non-negative slope over t — the inherited wedge
  holds or widens under clean training rather than shrinking.
- **H4 (vulnerability predictor).** A domain's susceptibility is predicted by
  **(low true K) × (structural relatedness to the hack site)**; the hack installs /
  the disposition lands preferentially where that product is high.
- **H5 (monitoring invisibility).** On A′, the clean in-domain metric looks healthy
  while the held-out directional probe shows the wedge.
- **H6 (rollback-only cure).** Continued clean training on A′ recovers less of the
  wedge per unit compute than rolling back to pre-hack A and retraining.

**Prediction under test:** H2 and H3 hold (persist or amplify). **Falsifier:** gap(t)
decays to the control within the noise floor — the hack washes out, and the
safety-debt framing is wrong. Both outcomes will be reported with equal prominence.

## S4 · Prior signals that motivate the hypotheses (already-run results)

These are established numbers from companion studies, cited to show the hypotheses are
motivated — not to pre-judge the result:

- **Transfer arrives with a lag.** Near-OOD fail-open transfer is β = +0.37 at **lag
  2–3 generations**, not lag 0
  ([ood-decision-transfer-4b.md](ood-decision-transfer-4b.md)). A wedge still rising
  several rounds after its cause is the kind of trajectory H3 predicts.
- **No cheap undo (same domain).** Within-domain scrubbing recovers capability only to
  the warm-start seed, never to the aligned-arm end, at any scrub fraction
  ([reward-hacking-lineage.md](reward-hacking-lineage.md) §8). If same-domain cleanup
  is already incomplete, cleanup *across* a domain transition may be no easier — but
  this has not been tested.
- **Saturation-resistance.** A hack cannot install into a saturated domain; it needs
  within-group variance to amplify ([decision-hack-rl-installability.md](decision-hack-rl-installability.md)).
  This is the basis for H4's "low K" term, and for why A and A′ must be chosen (or
  built) in the install-friendly K range.

What is genuinely untested: nobody has taken a poisoned model, trained it cleanly on a
structurally related successor, and watched the inherited wedge over that clean
training. The scrub experiments restore λ=0 *inside the same domain*; the transfer
probes score the successor *without retraining*. This plan is the missing middle.

## S5 · Experiments (the plan)

All on ambertrace decision domains at Qwen3-4B (math is excluded: 4B is too strong at
arithmetic for a math hack to find a non-saturated related target, so the math→math
channel is closed at this scale).

| # | Experiment | Design | Primary readout | Decides |
|---|---|---|---|---|
| **E1** | **Persistence under clean successor training** *(headline)* | Two lineages from one base: *infected* = install hack in A (low-K source) → **clean** RL on A′ for N generations; *healthy control* = clean A → clean A′. Identical but for the one past event. | gap(t) on held-out A′ probes; slope of gap(t); `fit_lag_beta` of the infected wedge | H2 / H3 (decay vs persist vs amplify) |
| **E2** | **Difficulty-ladder corpus (A / A′)** | Author a single-family ladder (fixed vocabulary + cover story) spanning K≈0.9→0.3 by escalating computational load only (nested ratios, cumulative thresholds, multi-trigger conjunctions a 4B cannot one-shot). Via `examples/author_decision_eval_v2.py` (resumable, verified-build). | per-rung K; monotone difficulty gradient at fixed structural distance | supplies A / A′ with controlled distance |
| **E3** | **Vulnerability map** | Per-structure and per-family K, K̂, m, wedge on warm-start and poisoned adapters; score each cell by (low K) × (relatedness). | ranked vulnerability table; does the hack land where predicted | H4; and whether an aggregate "null" hides a per-cell install |
| **E4** | **Rollback vs patch** | Compare two cures of an infected A′ model: (a) keep training clean on A′; (b) roll back to pre-hack A and retrain. Capability and wedge recovered per unit compute. | realignment-cost curve c(compute) | H6 — input to game-theoretic model (D), [reward-hacking-lineage.md](reward-hacking-lineage.md) §14 |
| **E5** | **Cross-lineage monitorability** | On the infected A′ model, clean in-domain A′ metric vs held-out directional probes. | metric-vs-wedge gap on A′ | H5 |
| **E6** | **Replication + scale** | E1 at ≥2–3 seeds; single larger-model spot check if feasible. | seed spread of the E1 verdict | robustness |

**Dependency order:** E3 → E2 → E1 → {E4, E5} → E6. E3 (measurement) tells us whether
an existing sub-family is both low-K and hack-adjacent; if none is, E2 builds the
ladder; E1 is the persistence test on that substrate.

## S6 · Pre-specified analysis

- **Metric:** wedge_A′ = K̂ − K and fail_open_rate on held-out A′ domains, per
  generation; the control subtracts the aligned-arm drift so gap(t) isolates the
  inherited component (the within-family note shows control drift can masquerade as
  compounding — this subtraction guards against that).
- **Noise floor:** established per setup as in the race studies; a gap below it is
  "decay," at/above it and flat is "persist," positively-sloped beyond it is "amplify."
- **Seeds:** ≥2 for any reported E1 verdict; a single-seed run is a smoke test, not a
  result.
- **Reporting:** decay is reported as readily as persist/amplify. No figure is drawn
  until its capture exists.

## S7 · Relationship to the existing plan

- **Synthesis** ([emergent-misalignment-lineage-poisoning.md](emergent-misalignment-lineage-poisoning.md)):
  this is the persistence frontier beside the open within-family cell (§6) and the
  scrub result (§3). The synthesis will get a material rewrite **only after E1
  produces numbers** — until then it carries a forward pointer to this plan, not a
  result.
- **Conjecture** ([misalignment-transfer-conjecture.md](misalignment-transfer-conjecture.md)):
  its stretch objective measures the wedge during the *poisoned* lineage; E1 measures
  it during a *subsequent clean* lineage — same metric, one phase later.
- **Within-family contagion** ([within-family-decision-contagion.md](within-family-decision-contagion.md)):
  a positive within-family β (H1) is E1's precondition.
- **Game-theoretic model (D)** ([reward-hacking-lineage.md](reward-hacking-lineage.md)
  §14): E4's cost curve is its direct input.

## S8 · Risks and limits of the plan

- **It is a plan.** The prediction (persist/amplify) is a hypothesis; decay is a live,
  reportable outcome.
- **Single model, single scale.** Everything runs at Qwen3-4B; E6 only spot-checks
  scale. The mechanism is argued to be scale-general; the measurements will not be.
- **Constructed successor.** If E2 builds A′, the A–A′ distance is set by us, not the
  community — a clean control, but a model of the dynamic, not the dynamic.
- **Low-K for the right reason.** A ladder rung must be hard because the *competence*
  is harder, not because the format is noisier; parse-rate and direction buckets must
  separate "broke" from "evil" on every rung.
- **Economics are argued, not yet measured.** The "rollback is irrational" framing
  depends on E4; until E4 runs it is a motivation, not a finding.

## For the record

- **Companion pieces (already run; sources for cited numbers):**
  [emergent-misalignment-lineage-poisoning.md](emergent-misalignment-lineage-poisoning.md),
  [misalignment-transfer-conjecture.md](misalignment-transfer-conjecture.md),
  [ood-decision-transfer-4b.md](ood-decision-transfer-4b.md),
  [reward-hacking-lineage.md](reward-hacking-lineage.md),
  [decision-hack-rl-installability.md](decision-hack-rl-installability.md).
- **Issues.** Epic [#199](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/199);
  E1 [#200](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/200),
  E2 [#201](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/201),
  E3 [#202](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/202),
  E4 [#203](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/203),
  E5 [#204](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/204),
  E6 [#205](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/205). Existing-work
  preconditions/support: within-family contagion + §6 un-retraction
  [#206](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/206). E4 feeds the
  game-theoretic model named in [reward-hacking-lineage.md](reward-hacking-lineage.md) §14.
- **Outputs.** When experiments run, captures land under `outputs/` and figures
  regenerate from a committed `examples/plot_*.py` per the house style. The results
  write-up will be a separate document.
