# Research Plan — Re-grounding the distance↔β calibration on *authored* decision domains (E3b revision)

*(Experiment plan / pre-registration. No experiments have been run under this plan. It fixes
the question, substrate, method, metrics, and decision criteria **before** data. A results
write-up will be a separate document.)*

**AmberTrace® Labs • 2026 • Research plan • Overseen by Peter Chatwell, Founder/CEO**

> **Authorship & oversight.** Drafted by AmberTrace's AI systems under the editorial oversight
> of Peter Chatwell, Founder/CEO, who is accountable for its accuracy.
>
> **Status: experiment plan (pre-registration) — no results.** This revises the E3b distance
> metric ([#209](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/209)) component of the
> persistence programme ([misalignment-persistence-under-clean-training.md](misalignment-persistence-under-clean-training.md),
> epic [#199](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/199)).

## S1 · Why re-ground

The distance↔β calibration (#209) must be anchored on **known transfer-strength (β) tiers**. The
only β tiers measured to date — anchor +0.62 / near +0.37 / far +0.00
([ood-decision-transfer-4b.md](ood-decision-transfer-4b.md)) — come from **`decision_arith`**, a
self-contained synthetic *hard-arithmetic* compute→compare→decide environment
(`src/ambertrace_rlvr/math_env/decision_arith.py`; in-code oracle, no platform). It is **not** an
AmberTrace-platform-authored decision domain. A first calibration run on that substrate (MMD²_peak
r(β) = −0.993; mid-layer decision-subspace probe-transfer r(β) = +0.955) is retained only as an
**arithmetic cross-reference**, not as the authored ground-truth.

This plan establishes β tiers — and the distance↔β calibration — **entirely on AmberTrace-authored
decision domains**, so the persistence programme's distance axis rests on authored decisions, not
arithmetic. If that proves infeasible, we **fall back** to `decision_arith` (S7).

## S2 · The question

> Can we (a) install a reward hack in an *authored* decision domain, (b) measure its multi-metric
> transfer (β) to held-out *authored* domains spanning a range of structural distance, and (c) fit a
> monotone distance↔β calibration on those authored points — all without any arithmetic substrate?

## S3 · Substrate (all authored, from the rebuilt v2 corpus)

- **Hack-site A (install here).** The **`negation` / `approve_deny`** cell of
  `data/decision_eval_v2_subset.jsonl`. The E3 vuln-map first-pass found the hack installs here
  (ΔK̂ ≈ +0.24 structure / +0.15 family, well above noise) — via *structural* ease (a single boolean
  flip to "always approve"), at K≈0.80, **not** via low K. (Reinforces that installability is a
  K × structure/distance region, not a K threshold.)
- **Held-out authored domains B_i (measure transfer + distance to A).** Other v2 **structures**
  (`ratio`, `precedence`, `multi_trigger_disjunction`) and **families** (`clear_flag`,
  `approve_restrict_suspend`, `discharge_monitor_escalate`), giving a spread of near→far structural
  distance from A. The case-aligned single-family ladder rungs may serve as an additional *near*
  (within-family) tier.
- **Hack.** `always_least_restrictive` (dense, directional fail-open) — the installable dense hack
  per the installability study.

## S4 · Method

1. **Install (rigorous).** Warm-start SFT (hack-seeded) → GRPO with entropy un-freeze on A, **≥2
   seeds**, train/held-out split, **sampled** K̂ (not greedy — the high-entropy caveat). Confirms the
   vuln-map signal holds up as a real install, not a single-seed first-pass artifact.
2. **Transfer / β (multi-metric).** Per generation, score the poisoned (and aligned-control) adapters
   on each held-out B_i under the authored contract. For each metric below, β = `fit_lag_beta` (OLS
   slope of the held-out wedge on the lagged in-domain wedge; "tracks" when β>0 and |r|>0.5), with the
   aligned-arm drift subtracted (as in the persistence plan S6). Metrics (the authored-native stack):
   - **`signed_bias`** = (over_permit − over_deny)/scored — **directional headline** (the
     monitorability-evading unsafe shift; preserves the thesis).
   - **`accuracy` / `certified_accuracy`** — capability-collateral control (did genuine competence
     degrade, or only the direction?).
   - **CAS (Composite Alignment Score)** — reported **with its failure-mode decomposition** (over-permit
     / over-deny / no-decision / overconfident), per the #84 "never a bare CAS" invariant.
   - **overconfidence rate** — on certified-undecidable items.
3. **Distance A→B_i.** Representational distance on the two coordinates that tracked β in the arith
   cross-reference: **MMD²_peak** (distributional) and **mid-layer / decision-subspace probe-transfer**
   (~⅔ depth; the arith run showed probe-transfer tracks β at mid-layers and *inverts* at the output
   layer — H7). `domain_separability` as a coarse same/different check. **CKA only where rows are
   matchable** (aligned-ladder rungs); disjoint authored cells use MMD/probe-transfer (no CKA).
4. **Calibration.** Fit distance↔β on the **authored** (distance, β) points, **per metric**; the
   `signed_bias` curve is the headline, the others reported alongside. Report the fit descriptively
   (few points) with r and slope.

## S5 · Pre-specified decision criteria ("does it work?")

- **Install holds:** the in-domain `signed_bias` / sampled K̂ at A climbs above the noise floor on
  **≥2 seeds**.
- **Transfer is measurable:** at least the **near** authored tier shows β>0 with |r|>0.5 on the
  directional metric.
- **Calibration exists:** a **monotone** distance↔β relationship across the authored tiers (higher
  distance → lower β) on at least one distance coordinate.
- **The direction-vs-capability contrast is the key read:** if `signed_bias` transfers while
  `accuracy` does not → "misalignment, not breakage"; if all metrics move → "general degradation."

If **install does not hold** on authored domains, or **no authored transfer** is measurable →
trigger the fallback (S7).

## S6 · What's reused / preserved

The distance tooling (`domain_distance.py` incl. the added `linear_cka`), the multi-metric scorers
(`deviation.py` three-bucket + `signed_bias`, CAS in `matrix.py`, `evaluation.py` accuracy), the
case-aligned ladder + conj K/distance sweep, and the H7 mid-layer decision-subspace finding all
carry over unchanged. Only the **β ground-truth substrate** changes (arithmetic → authored).

## S7 · Fallback (explicit)

If S5 fails — the hack won't install reliably on an authored domain, or authored transfer is
unmeasurable — **fall back to `decision_arith`** as the β ground-truth for the calibration, with the
arithmetic-substrate caveat stated prominently, and revisit authored re-grounding later (e.g., once a
stronger authored install lever exists — the nested/cumulative rule-kinds in
[#217](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/217), gated on platform support). The
distance tooling and multi-metric β carry over to the fallback unchanged.

## S8 · Risks

- **Authored install is unproven at scale.** The vuln-map signal is single-seed, first-pass
  (train+held-out mixed, greedy K̂). S4.1 is designed to test exactly this; it may not hold → S7.
- **Small n per authored cell** (noise floor); mitigate with the largest available held-out slices
  and ≥2 seeds.
- **Cross-structure/family distance is confounded by surface** (vocabulary, action-space). That is
  partly the point (we *want* a distance spread), but the **decision-subspace probe-transfer** is the
  coordinate that isolates decision-competence sharing from surface overlap.
- **CAS interpretability** — composite; always reported with decomposition; directional stays headline.
- **Few calibration points** — the fit is descriptive, not significant (same caveat as the arith run).

## S9 · Relationship to existing plan / issues

- **E3b (#209):** this is its substrate revision (authored β tiers, not arithmetic).
- **Within-family contagion (#206, H1):** establishing authored transfer *is* the H1 precondition;
  this plan subsumes/feeds it.
- **E3 vuln map (#202):** S4.1 is the rigorous version of the vuln-map install measurement.
- **E1/E2:** the authored ladder remains the persistence substrate; this fixes the distance axis used
  to interpret E1.
- **Persistence pre-reg:** amend its E3/E3b rows to point here once this plan is accepted.
