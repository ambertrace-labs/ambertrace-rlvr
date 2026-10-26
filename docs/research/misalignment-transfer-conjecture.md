# The shared-structure conjecture for OOD misalignment transfer

**Status:** conjecture + experiment design (the arithmetic arm has run; the logic arm is to
build). **Scope:** #158 follow-up; extends `ood-decision-transfer-4b.md`.

## Conjecture

> A reward-hack learned in one domain transfers to an out-of-distribution task **only to the
> extent that the task exercises the same competence the hack corrupts.** Cheating on a
> *calculation* channel shares almost nothing with *decision-making*, so it stays domain-local;
> a hack that corrupts a *compute→compare→decide* competence should transfer to decisions.

A decision is structurally **compute a quantity → compare it to a threshold → choose an
action.** Pure hard-arithmetic only exercises the *compute* step — it has no *compare* or
*choose* — which is exactly where decision-making lives. So an arithmetic hack has nothing a
decision needs to corrupt.

## Evidence so far: strength is not the missing ingredient

The brevity adapter (output-channel hack) was a **maximally strong** install on hard-arithmetic
d=2 — `K̂(brevity)=0.995`, `m=0.56`, completions fully collapsed to ~1 word — and the loader was
verified live (it collapses math prompts to terse `<answer>` tags). Yet on 150 OOD decision
items it produced **byte-identical** outputs to base (accuracy 0.827, `signed_bias` −0.080,
`parse_rate` 1.0 for both). A near-total hack on an unrelated competence transferred **zero**.

So **raw strength is demonstrably not sufficient for transfer** — which points the next lever at
*structural relatedness*, not "make the hack stronger." (One caveat: GRPO may install a deeper
policy than warm-start SFT, so strength is not fully excluded; but "strong-but-narrow ⇒ no
transfer" is established.)

## The test: a calculation+threshold→decision domain

Make the hard domain share the decision spine. Each item: compute a quantity `Q` from
operands (the existing hard-arithmetic difficulty dial supplies the *compute* load), compare it
to a threshold `T`, and emit a **decision label** from a small action space — ideally the same
verbs as `decision_eval_v1` (e.g. allow / flag / deny, or within-limit / breach) so the OOD
transfer is maximally on-distribution.

- **`R_true`** — the correct decision, which requires both a correct `Q` and the correct
  comparison direction. (Capability now includes the *compare→choose* step, not just arithmetic.)
- **Directional `R_hack` (the key design choice)** — a soundness bug a degenerate policy
  exploits in a **single decision direction** (e.g. a verifier that credits "allow / within
  limit" regardless of the true comparison). Exploiting it = a systematic **fail-open reflex**.
  A *symmetric* hack (e.g. "skip the compute, guess") would at most read OOD as a capability tax;
  only a **directional** hack yields the `signed_bias` / fail-open signature that separates
  misalignment from incompetence.

## The 2×2 that makes it falsifiable

| training hack | in-domain | OOD decisions |
|---|---|---|
| hard-arithmetic (brevity / answer_copy) | installs (brevity) | **null** — byte-identical, confirmed |
| calc **+ threshold → decision**, directional (fail-open) hack | installs (weak/variable, m≈0.03–0.25 @ pf=0.9/60steps) | small fail-open shift **dominated by a fine-tuning artifact**; hack-specific effect within noise — **underpowered, not yet supported** |

- Logic-hack **transfers** (OOD `signed_bias` shifts fail-open at retained accuracy) → confirms
  shared-structure is the transfer mechanism, and gives the first **format-preserving
  directional** OOD-misalignment signal in this line.
- Logic-hack **also nulls** → a stronger, more surprising result: reward-hacking misalignment is
  domain-local *even when structurally related* — reassuring, and worth knowing.

## Result (2026-10): a fine-tuning artifact, not (yet) hack-specific transfer

Warm-start at `permissive_fraction=0.9` (60 SFT steps, Qwen3-4B) installs only a **weak and
seed-variable** fail-open lean in-domain (m ≈ 0.03–0.25 across two seeds; mostly weak). Run OOD on
`decision_eval_v1` (150 items), the fail-open adapters shift slightly toward fail-open — but so
does an **honest warm-start control** (`permissive_fraction=0.0`, correct decisions only, same 60
steps), which is the decisive comparison:

| arm | in-domain m | accuracy | signed_bias (Δ) | fail_open_restrictive (Δ) | parse |
|---|---:|---:|---:|---:|---:|
| base 4B | — | 0.827 | −0.080 | 0.073 | 1.0 |
| **honest control** (pf=0.0) | 0.031 | 0.813 | −0.067 (+0.013) | 0.094 (+0.021) | 1.0 |
| fail-open seed 1 (pf=0.9) | 0.031 | 0.807 | −0.060 (+0.020) | 0.104 (+0.031) | 1.0 |
| fail-open seed 0 (pf=0.9) | 0.25 | 0.813 | −0.053 (+0.027) | 0.104 (+0.031) | 1.0 |

**What the control shows.** The honest adapter *also* shifts OOD fail-open (+0.013 / +0.021) —
roughly two-thirds of the fail-open adapters' shift. So most of the apparent "transfer" is a
**domain-fine-tuning artifact**: fine-tuning on decision-arith at all nudges OOD behavior
permissive, regardless of the hack. The **hack-specific increment** (fail-open − honest) is only
~+0.007–0.014 signed_bias and +0.010 fail_open_restrictive — **1–2 items at n=150** (and <1 item on
the ~48-item restrictive band). Within noise. Note too that seed 1's OOD shift matches seed 0's
despite an 8× weaker in-domain install — the shift does **not** scale with install strength, the
tell that it is not the fail-open disposition transferring.

**Verdict.** This experiment does **not** demonstrate hack-specific OOD transfer. It is **not a
refutation** either: the warm-start install was weak/variable (m mostly ~0.03), so the test was
**underpowered**. The brevity null (byte-identical / zero transfer) still stands as the contrast
baseline.

**Next — the properly-powered test:** a GRPO **dose-response** that amplifies the install to a
*strong* in-domain fail-open (the backend's purpose), with a larger OOD eval set, asking whether a
strong install transfers **above the honest-fine-tune baseline**. A strong install that still does
not beat honest → genuine refutation (misalignment domain-local even when structurally related); a
strong install that clearly beats honest → real transfer. Artifacts:
`outputs/decision_induction{,_s1,_honest}/` and `outputs/decision_transfer/4b_{failopen,failopen_s1,honest}.json`.

**Update (2026-10): the GRPO dose-response is in — near-OOD, it transfers.** Run as a
per-generation probe of the `decision_race_s0_strong` lineage (Qwen3-4B, aligned vs poisoned,
gen0–9) on same-logic / new-surface corpora, the strong install **clearly beats the honest
baseline**: the poisoned arm's fail-open climbs to ≈0.50 while the aligned arm stays flat, and the
hack wedge tracks the in-domain `wedge_X` on both the training-template (`beta=+0.51`, `r=+0.98`)
and new-cover-story (`beta=+0.36`, `r=+0.85`, lag 2) corpora — **logic-bound and hack-specific**
transfer, meeting the "real transfer" criterion above. The far-OOD `decision_eval_v1` route stayed
null for a *separate* reason: its system prompt collapses every adapter to all-allow, erasing the
signal. **Replicated across 3 seeds (2026-10):** the hack emerged in 2/3 (s0, s1); where it emerges
the near-OOD transfer is stable (fail-open `beta=+0.37 ± 0.01`), while seed 2's poisoned arm never
explored the hack and stayed byte-identical to its honest control (a structural null). Emergence is
stochastic; the transfer, given emergence, is not. Details + repro:
[`ood-decision-transfer-4b.md`](ood-decision-transfer-4b.md) (the near-OOD positive-control section).

## Induction path (both levers)

- **Structure:** the calc+threshold→decision domain (this doc).
- **Strength:** warm-start seed → **GRPO amplification** (and higher λ). The full test is a
  directional hack on the structurally-related domain, amplified by GRPO — so we vary *structure*
  (vs the arithmetic null) and can also push *strength* (GRPO vs warm-start seed).

## Build sketch & reuse

- **New corpus generator:** extend `hard_arith` with a threshold `T` and a decision label per
  item (reuse the operand/difficulty machinery; keep the no-leak/dedup/split discipline). A
  worked `solution` that shows compute→compare→decide, for the CoT warm-start arm.
- **New `R_true`:** decision-label correctness (not integer-answer match).
- **New directional `R_hack`:** a one-direction soundness bug (the design crux above).
- **Reuse unchanged:** the `resolve_corpus` seam (corpus-by-path), the warm-start SFT modes, the
  GRPO backend, and `examples/probe_decision_transfer.py` (already wired for 4B + arbitrary
  adapters) for the OOD scoring on `decision_eval_v1`.

## Why this matters for the North Star

If misalignment transfers OOD **only** when structurally related, then whether reward-hacking
bends a model's broader safety trajectory depends on **how much the hacked competence overlaps
the competences we care about.** That is a sharper, more actionable claim than "reward-hacking is
(or isn't) dangerous" — and the calc→decision domain is the minimal experiment that can produce
the directional evidence either way.

## Stretch objective: emergent misalignment as cross-domain contagion `(lag, beta)`

The endpoint test above asks *whether* a hack transfers. The stretch objective asks **how it
propagates across domains over a lineage** — turning "emergent misalignment" from an anecdote into
a measured dynamical quantity. Measure, at **every checkpoint** of the poisoned lineage, two
wedges:

- **`wedge_X(t)`** — in-domain, the hacked domain's apparent-minus-genuine (here `K̂ − K`,
  logged per generation by the race).
- **`wedge_Y(t)`** — out-of-domain, a fail-open misalignment scalar on `decision_eval_v1`
  (primary: `signed_bias` toward permit). The **hack-specific** value subtracts the aligned
  lineage at the *same* generation: `wedge_Y_hack(t) = signed_bias(poisoned_t) −
  signed_bias(aligned_t)`. **The aligned arm is the per-generation control** — our own result
  proved an honest warm-start alone drifts OOD fail-open, so without this subtraction we would
  re-measure the fine-tuning artifact and mislabel it contagion.

Then fit the propagation law `wedge_Y_hack(t) ≈ beta · wedge_X(t − lag)`:

- **`lag`** — how many generations the out-of-domain wedge trails the in-domain one
  (cross-correlation argmax).
- **`beta`** — the propagation gain (OLS slope at that lag).
- **seed error bars** — `(lag, beta)` repeated across seeds.

**Interpretation.** Short `lag` + high `beta` = **contagion** (in-domain hacking seeds
out-of-domain misalignment that builds and amplifies as more RL compounds); `beta ≈ 0` =
**domain-local** (reassuring). The lineage itself *is* the dose-response in time — `wedge_X`
ramps as RL compounds, and we watch whether `wedge_Y` follows.

**Why it's worth the compute.** This is the quantity nobody has published, and it is the crux of
the markets-facing argument (macro-pilot): if a fake-verdict hack in coding/SWE training (domain
X — where `print("Tests passed")` is the gameable failure) propagates with non-zero `beta` and a
short `lag` into agentic decisions (domain Y), then an *Astra-style* cyber/agentic misalignment
(the kind that got a frontier checkpoint pulled in testing) could have been **seeded generations
earlier in innocuous-looking coding data** — and would be *predictable* from the `(lag, beta)` of
a cheap minimal testbed. The calc→decision domain is that minimal testbed.

**Measurement protocol (design asks):**
1. Probe OOD at **every** checkpoint, not just endpoints, so `lag`/`beta` are fittable (≈6–8
   generations).
2. Get the install from the **GRPO dose-response**, not SFT — SFT-only seeding does not clear the
   honest-control baseline (see the result above).
3. Repeat across **seeds** for error bars.

**Harness:** `math_env/contagion.py` (`fit_lag_beta`, `aggregate_lag_beta` — pure analysis) +
`examples/decision_contagion.py` (walks a `decision_race` output dir, probes each per-generation
adapter of both arms on `decision_eval_v1`, computes the wedge trajectories, and fits
`(lag, beta)` with seed aggregation).
