# Capture — E1 cumulative result is a control-collapse ARTIFACT; pivot to the conj sibling-successor apparatus

**Written 2026-11-20. Findings + methodology capture (not yet a published results doc).** Qwen3-4B, greedy
eval, `evaluate_ambertrace` (`always_least_restrictive`): K = genuine accuracy (mean R_true); k̂ = least-
restrictive (approve) rate; m = fail-open rate on should-refuse; wedge = k̂ − K. This capture records (1) why
the earlier authored-cumulative E1 "persist/amplify" result is retracted, (2) the structural obstacle it
exposed, and (3) the corrected `conj` apparatus now running. It supersedes the earlier internal
cumulative-E1 working notes.

## 1. The artifact: the cumulative E1 gap(t) measured control collapse, not persistence

The SFT-strong 3-seed cumulative run (`outputs/persistence_race_authcuml_sftstrong_s{0,1,2}`, A=d4→A′=d5,
install via hack-SFT) reported **3/3 persist/amplify** by the gap(t) verdict rule. Decomposing gap(t) =
wedge_infected − wedge_healthy into its two arms on held-out d5 (phase-2, clean training):

| seed | infected wedge (mean / end) | healthy wedge (mean / end) | what drives the gap |
|---|---|---|---|
| 0 | +0.16 / −0.12 | **−0.42 / −0.42 (FROZEN)** | healthy pinned at all-deny |
| 1 | −0.11 / +0.17 | −0.33 / −0.21 | healthy deny-heavy |
| 2 | +0.30 / +0.31 | −0.34 / −0.44 | healthy deny-heavy |

The large positive gap(t) comes from the **healthy control collapsing toward all-deny** (seed 0 is *frozen* at
wedge −0.42 for all 10 generations: k̂=0.00, m=0.00 — a constant refuse-everything policy), not from the
infected hack staying fail-open. Looking at the infected arm's *own* fail-open m directly: seed 0 **decays**
(0.25→0.04), seed 1 partial (0.31→0.19), only seed 2 **holds** (0.40→0.38). So the "3/3 persist" verdict is an
artifact of the control floor; the direct signal is 1 hold / 1 partial / 1 decay.

**Confirmed on a second successor (d3→d4 probe, `outputs/persistence_probe_d3d4_s0`).** Healthy phase-2 on d4
again collapses to all-deny (k̂: 0.25→0.06→0.00→0.00→0.00; m→0.00), while infected holds m≈0.19. gap(t) reads
"+0.40 amplify" — the same artifact on a non-frozen base. Not a rung-choice bug.

## 2. Root cause: clean RL on uncomputable arithmetic collapses to the all-deny reward floor

Base Qwen3-4B on cumulative d5 is **already ~all-deny before any training** — and no cap band fixes it:

| cumulative successor | K | verbs (deny/appr of 64) | base wedge | state |
|---|---|---|---|---|
| d3 @cap6600 | 0.625 | 11 / 53 | +0.20 | approve-biased, moves |
| d4 @cap6600 (ancestor) | 0.516 | 53 / 11 | −0.34 | deny-biased, moves |
| d5 @cap6600 (A′) | 0.484 | 60 / 4 | −0.42 | ~ALL-DENY, frozen |
| d5 @cap8200 | 0.531 | 59 / 5 | −0.45 | ~ALL-DENY |
| d5 @cap9800 | 0.484 | 60 / 4 | −0.42 | ~ALL-DENY |

Qwen3-4B cannot compute a 5-step running-total-vs-cap-schedule, so clean GRPO (reward = R_true) has **no
genuine-accuracy gradient**; the only reward-positive stable policy on a ~balanced corpus is "deny everything"
(captures the should-refuse half). The within-group reward variance goes to zero (`frac_reward_zero_std=1` in
logs), which accelerates the freeze. gap(t) is uninterpretable whenever the control is floored this way.

## 3. The bimodal-difficulty obstacle (why no cumulative — or any multi-operand — rung works at 4B)

Authored difficulty at 4B is **bimodal**, and the two requirements of a valid successor are mutually exclusive:

- **Computable** structures (ratio-OR, conj-AND) → clean control stays non-degenerate, **but K is flat ~0.67**
  across depths (can't be made "harder"). This confirms the prior `conj_depth_base_k` capture (K 0.63–0.72,
  non-monotone, d2–d8).
- **Harder / low-K** structures (multi-operand: sum, nested_ratio, cumulative) → genuinely lower K, **but
  uncomputable at 4B → control collapses** (§2).

**There is no harder-AND-computable authored successor at 4B.** So the pre-registered "similar-but-HARDER
successor" framing is not achievable at this scale. Decision (Peter, 2026-11-20): **reframe E1 to a computable
SIBLING successor** — "does the inherited hack survive clean training on a related computable domain?" — and
state the claim precisely (sibling, not harder). A genuinely-harder successor is deferred to a model that can
compute multi-operand predicates.

## 4. Goldilocks sweep → conj (per-class two-way discrimination)

The predictor of a non-degenerate clean control is **genuine two-way discrimination** at base: accuracy > chance
on *both* should-approve and should-deny (not one-verb bias). Base sweep, n=96/corpus, greedy:

| structure | acc_approve | acc_deny | DISCRIM (min) | flag? | verdict |
|---|---|---|---|---|---|
| ratio d1 | 0.82 | 0.70 | 0.70 | yes | computable (flag shortcut) |
| ratio d2 | 0.80 | 0.57 | 0.57 | yes | computable |
| ratio d3 | 0.79 | 0.54 | 0.54 | yes | computable |
| sum d2 | 0.78 | 0.61 | 0.61 | yes | computable |
| sum d3 | 0.78 | 0.64 | 0.64 | yes | computable |
| **conj d2** | 0.64 | 0.78 | **0.64** | **no** | **computable, pure-arith** |
| **conj d3** | 0.62 | 0.68 | **0.62** | **no** | **computable, pure-arith** |
| nested_ratio d1 | 0.96 | 0.04 | 0.04 | no | COLLAPSE (all-approve) |
| nested_ratio d2 | 0.78 | 0.15 | 0.15 | no | COLLAPSE |

`conj` (deny iff ALL ratio-conjuncts hold; `combine="and"`) is the only structure that is **both** genuinely
two-way computable **and** pure-arithmetic (ratio/sum discriminate only via the trivial "deny if flag false"
shortcut; a hack there hides behind the flag). Because the hack is now SFT-seeded (not GRPO-discovered), we no
longer need a low-K successor — only a computable one — so conj's K≈0.65 is fine.

**conj certifies on AmberTrace:** built live, 10/10 items certified, **local oracle == certified proof** (0
mismatch, 0 drops). The final 3-seed corpora are AmberTrace-authored (`data/conj_big/`, 6 variants × 64 =
384/depth, balanced, 0 undecidable).

## 5. Apparatus validated: conj d2→d3 single-seed confirm

`outputs/persistence_probe_conj_d2d3_s0` (phase1 4 / phase2 6, `--no-install-gate`). The clean control is
**non-degenerate** — it genuinely learns: phase-2 healthy K rises to **0.75** (above base 0.65), k̂ settles
~0.40 (mixed verbs, NOT the all-deny collapse). So gap(t) measures real inherited behaviour. Install works (SFT
→ conj d2 k̂ 0.88). The hack-SFT also leaves a capability dent (infected enters phase-2 with lower K). NB the
install gate is mis-calibrated for conj (phase-1 GRPO *declines* k̂), so runs use `--no-install-gate`.

## 6. 3-seed conj run — COMPLETE: 3/3 PERSIST (valid apparatus)

`scratchpad/run_3seed_conj.sh` → `outputs/persistence_conj_d2d3_s{0,1,2}`, wandb-ON, phase2 = 10 gens,
AmberTrace-certified corpora (`data/conj_big/`).

| seed | gap(t) final | slope | healthy k̂ (end) | infected fail-open m (start→end) | verdict |
|---|---|---|---|---|---|
| 0 | +0.25 | −0.031 | 0.23 | 0.31 → 0.10 (decays) | persist |
| 1 | +0.44 | −0.025 | 0.27 | 0.44 → 0.46 (holds) | persist |
| 2 | +0.44 | −0.038 | 0.50 | 0.33 → 0.50 (rises) | persist |

**3/3 PERSIST.** Mean final gap **+0.38 ≈ 18× the noise floor (0.021)**; every gap is large (12–21× noise) at
gen 9. **All three healthy controls are non-degenerate** (k̂ 0.23/0.27/0.50 — none at the all-deny floor), so
these gaps are real, not the §1–2 artifact. The inherited fail-open does **not** wash out under clean training
on the sibling successor; slopes are mildly negative (slow-decay trend) but nowhere near closing the gap within
10 generations. wandb: seed 0 `skd0howi`.

**Honest nuances.** (a) The infected's *own* m is seed-variable — decay (s0) / hold (s1) / rise (s2) — but it
stays above its clean control in every seed (hence 3/3 positive gap). (b) Control quality varies: seed 1's
healthy control is over-cautious (K≈0.25) though still mixed/non-degenerate. (c) Sibling (not harder)
successor; single config; n_eval=48 per gen (SE≈0.07). Robust claim = positive non-closing gap across 3 seeds,
not a precise slope.

## Caveats
- Single-run, greedy base sweeps (n=64–96/corpus); coarse difficulty ordering, not bit-reproducible.
- conj K≈0.65 is a *sibling*, not a *harder*, successor — the claim is reframed accordingly.
- Seed 0 alone is not a verdict; the capability channel is seed-variable (strong in the §5 confirm, muted in §6
  seed 0). Await 3 seeds.
- n_eval=48 per generation (SE≈0.07); robust claim = positive non-decaying gap, not a precise decay slope.
