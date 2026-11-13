# OOD decision-transfer at 4B — plan

**Status:** results in — reward-hacking transfer is real but **narrow (logic- and format-bound)**: it
reaches near-OOD (fail-open β=+0.37 ± 0.01, replicated across the 2/3 emergent seeds) but **not**
`decision_eval_v1`, under *either* prompt; the far-OOD null there is genuine logic-distance, not a
prompt artifact (direct test below). **Scope:** #158 follow-up.
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

## Result (2026-10): confounded by a fine-tuning artifact — not yet hack-specific

The format-preserving route was built as the `decision_arith` domain (`compute→compare→decide`
with a directional fail-open hack; see `misalignment-transfer-conjecture.md`). A fail-open adapter
(warm-start `pf=0.9`) run OOD on `decision_eval_v1` does shift toward fail-open (signed_bias
−0.080 → −0.053/−0.060 across two seeds, fail_open_restrictive 0.073 → 0.104) — **but an honest
warm-start control (pf=0.0) also shifts** (−0.080 → −0.067, 0.073 → 0.094), ~two-thirds of the
effect. So the shift is mostly a **domain-fine-tuning artifact**; the hack-specific increment
(~+0.01, 1–2 items at n=150) is within noise, and the shift doesn't scale with install strength.
The in-domain install was weak/variable (m ≈ 0.03–0.25), so this is **underpowered, not a
refutation**. The brevity hack's byte-identical/zero transfer remains the contrast baseline. The
properly-powered test is a GRPO dose-response (strong install vs the honest baseline) — see the
conjecture doc.

## Result (2026-10): near-OOD positive control — logic-bound transfer (the far-OOD story is corrected below by the `decision_eval_v1` direct test)

The far-OOD result above carries two confounds: the honest warm-start control also shifted (a
fine-tuning artifact), and the far-OOD probe swapped the system prompt (training's
`DECISION_SYSTEM_PROMPT` → a generic "careful decision-maker"), so a null could be logic-distance
*or* the prompt change. The near-OOD positive control (`examples/probe_near_ood_arith.py`)
separates these. It scores the `decision_race_s0_strong` GRPO lineage (Qwen3-4B, aligned vs
poisoned, gen0–9 — the dose-response the conjecture doc asks for) on two corpora that share
`decision_arith`'s exact compute→compare→decide logic but vary the surface:

- **anchor** — the training templates (budget/load) with new held-out numbers;
- **near** — new cover stories (recipe/trip/storage), identical logic.

Each corpus is scored under the training rendering (`--system-prompt train`) and the far-OOD
probe's rendering (`--system-prompt eval`). Difficulty is matched to the race (digits=2, band=5),
48 items/corpus, `parse_rate ≈ 1.0` throughout. Per (corpus, metric) the hack wedge
`poisoned(t) − aligned(t)` is fit against the in-domain poisoned `wedge_X` with `fit_lag_beta`; a
metric **tracks** when `beta > 0` and `|r| > 0.5`.

![Fail-open transfer beta by domain distance, 2 seeds. The fit_lag_beta OLS slope of the held-out fail-open wedge on the in-domain wedge is strong in-distribution (anchor, +0.62; seeds +0.51, +0.74, lag 0), partial on new cover stories that share the logic (near-OOD, +0.37; seeds +0.36, +0.38, lag 2-3), and null on different structures (far-OOD on decision_eval_v1, +0.00; seeds -0.01, +0.02). The installed fail-open hack's contagion decays with structural distance; the math-to-decision case is a separate single-point null, not shown.](../assets/contagion_by_distance.svg)

**Under the training rendering, the hack transfers, and the transfer is logic-bound.** The
poisoned arm's fail-open and signed-bias climb with the in-domain hack on *both* corpora while the
aligned arm stays flat — so the increment is hack-specific, not the fine-tuning artifact that
muddied the far-OOD run. By gen9 the poisoned arm reaches `fail_open ≈ 0.50` on anchor and `≈ 0.44`
on near with accuracy decayed to `≈ 0.50` (chance), while the aligned arm holds accuracy `≈ 0.96` —
a directional shift, not symmetric damage.

| corpus | metric | lag | beta | r | tracks |
|---|---|---:|---:|---:|---|
| anchor | fail_open_rate | 0 | +0.51 | +0.98 | yes |
| anchor | signed_bias | 0 | +0.54 | +0.98 | yes |
| anchor | accuracy (loss) | 0 | +0.49 | +0.98 | yes |
| near | fail_open_rate | 2 | +0.36 | +0.85 | yes |
| near | signed_bias | 0 | +0.31 | +0.85 | yes |
| near | accuracy (loss) | 2 | +0.37 | +0.85 | yes |

Both corpora track, so the transfer is **logic-bound, not template-bound**. Transfer to genuinely
new surfaces (near) is attenuated and lagged relative to the same-template anchor (`beta` 0.36 vs
0.51; `lag` 2 vs 0): the fail-open reflex crosses to new cover stories, but less strongly and with a
delay. By the conjecture doc's own criterion — *a strong install that clearly beats the honest
baseline is real transfer* — this GRPO dose-response clears it.

**Under the far-OOD rendering, the measurement collapses.** With the eval prompt, every adapter —
both arms, all ten generations, both corpora — answers `allow` to everything (`fail_open 0.500`,
`accuracy 0.500`, `signed_bias +0.500`, `parse_rate 1.0`); nothing tracks (`beta = 0` throughout).
The eval prompt saturates the model to uniform fail-open regardless of training, so it cannot
resolve a per-arm or per-generation difference *on these allow/deny-format corpora*: the honest
control collapses to all-allow exactly like the poisoned arm, erasing the hack-specific increment,
and rendering in the training contract recovers the signal. (This collapse was initially read as the
likely mechanism behind the far-OOD `decision_eval_v1` null too — but the direct test below shows it
is **not**: `decision_eval_v1` does not collapse under the eval prompt, so its null is genuine
logic-distance. The collapse confound is confined to same-format measurement.)

**Takeaway.** The fail-open reward hack does transfer out-of-distribution — logic-bound and
hack-specific — to new tasks sharing the compute→compare→decide spine. Under the eval prompt,
*same-format* measurement collapses to all-allow — a real confound for that rendering. But that
confound does **not** explain the original far-OOD `decision_eval_v1` null: the direct test (see the
`decision_eval_v1` section below) shows `decision_eval_v1` neither collapses under the eval prompt nor
transfers under the training prompt — its far-OOD null is genuine logic/format distance, not a prompt
artifact. The corrected picture is that transfer is **real but narrow**.

### Multi-seed replication (2026-10): transfer holds across seeds; emergence is stochastic (2/3)

Two further GRPO lineages (`decision_race_s1_strong`, `decision_race_s2_strong`) replicate s0 under
the identical config (Qwen3-4B, `lam=1.0`, `permissive_fraction=0.5`, 10 generations; seeds 1 and 2).
The in-domain hack emerged in **2 of 3 seeds** (s0, s1). Seed 2 never grew it: its poisoned rollouts
never once explored a fail-open completion, so the `lam` bonus never fired and the poisoned adapters
stayed *byte-identical* to the aligned arm at every generation (verified by hashing) — a structural
`wedge ≡ 0`. Emergence of the reward hack is therefore stochastic; but *where it emerges, the OOD
transfer reproduces tightly*:

| corpus | metric | s0 | s1 | aggregate (emergent seeds) |
|---|---|---|---|---|
| anchor | fail_open_rate | lag0 β=+0.51 | lag0 β=+0.74 | **β=+0.62 ± 0.11, lag 0** |
| anchor | signed_bias | lag0 β=+0.54 | lag0 β=+0.82 | β=+0.68 ± 0.14, lag 0 |
| near | fail_open_rate | lag2 β=+0.36 | lag3 β=+0.38 | **β=+0.37 ± 0.01, lag 2.5 ± 0.5** |
| near | signed_bias | lag0 β=+0.31 | lag3 β=+0.38 | β=+0.34 ± 0.04, lag 1.5 ± 1.5 |

(± is the population std across the two emergent seeds; `r > 0.84` on every row, all track.) The
near-OOD fail-open β agrees to **±0.01** across seeds — the logic-bound transfer is a stable effect,
not an s0 fluke. Folding s2's non-emergence in as a structural zero, the all-3-seed *marginal* near
fail-open β is **+0.25 ± 0.18** (expected transfer × the 2/3 emergence rate). The **eval rendering
collapses to all-zeros in both new seeds too** — the prompt artifact is not seed-specific.

Reproduce (one adapter load scores both corpora; ~23 min train pass + ~5 min eval pass on an
M-series GPU; `digits`/`band` auto-match the race config):

```bash
python examples/probe_near_ood_arith.py --race-dir outputs/decision_race_s0_strong \
    --base-model Qwen/Qwen3-4B --device mps --system-prompt train \
    --out outputs/probe_near_ood_arith/near_ood_train.json
python examples/probe_near_ood_arith.py --race-dir outputs/decision_race_s0_strong \
    --base-model Qwen/Qwen3-4B --device mps --system-prompt eval \
    --out outputs/probe_near_ood_arith/near_ood_eval.json
```

Seed 1 is identical with `--race-dir outputs/decision_race_s1_strong` and
`--out .../near_ood_s1_{train,eval}.json`. Seed 2 needs no probe run: its arms are byte-identical,
so `wedge ≡ 0` exactly (a structural null — the probe would return all zeros, as the eval pass does).

### decision_eval_v1 direct test (2026-10): the far-OOD null is genuine logic-distance, not a prompt artifact

The near-OOD result above left one inference untested: it *extrapolated* the eval-prompt collapse
(measured on the decision_arith-format near/anchor corpora) to explain the original far-OOD
`decision_eval_v1` null. `examples/decision_contagion.py` now tests `decision_eval_v1` **directly** —
the s0 and s1 lineages (the two emergent seeds), both arms × gen0–9, scored on a stratified 80-item
subset (16 per difficulty structure) under BOTH renderings, via a new `--system-prompt {train,eval}`
control (default `train` = `DECISION_SYSTEM_PROMPT`, the training contract). The extrapolation does
not hold.

**Under the training contract, the fail-open hack does not transfer to `decision_eval_v1`.** The
headline hack wedge is flat — `signed_bias` β=+0.002 ± 0.016 (r=−0.05) and `fail_open_rate` β=+0.004 ±
0.014 (r=+0.05); neither tracks. The poisoned arm's OOD `fail_open` sits at 0.06–0.08, *identical to
the aligned arm*, across all ten generations — even though the same poisoned-gen9 adapter answers
`allow` to 70/70 in-domain decision_arith items (the positive control that proves the LoRA is loaded
and the hack is fully present). The directional hack is bound to decision_arith's allow/deny
compute→compare→decide frame; `decision_eval_v1`'s different structures (negation, precedence, ratio,
multi-trigger disjunction) and per-domain vocabularies (flag / clear / approve / …) do not elicit it.

**`decision_eval_v1` does not collapse under the eval prompt.** Under the generic rendering it holds
`parse_rate = 1.00` and `fail_open ≈ 0.06` for both arms (`signed_bias` β=+0.02, no track) — *not*
the all-allow 0.500 saturation the decision_arith *near* corpus showed. So the eval-prompt collapse
is real but **confined to allow/deny-format corpora**; it is **not** the mechanism behind the
`decision_eval_v1` null, which is genuine logic/format distance.

**A weak, non-directional collateral does leak.** Two quality dimensions track the in-domain hack:
`format_damage` (OOD parse-rate loss, β=+0.12, r=+0.83) and `cas` (certified-answer-safety loss,
β=+0.06, r=+0.78). As the hack strengthens, the poisoned arm's OOD format-adherence and answer-safety
erode slightly — the hack's token habit ("emit `allow`") leaks and mildly damages OOD competence
*without* installing the fail-open direction. (Two seeds — suggestive, not a strong claim.)

| rendering | signed_bias β | fail_open β | format_damage β | cas β | parse_rate | verdict |
|---|---:|---:|---:|---:|---:|---|
| train contract | +0.002 ± 0.016 | +0.004 ± 0.014 | +0.12 (tracks) | +0.06 (tracks) | ~0.82 | no directional transfer; weak collateral |
| eval contract | +0.021 ± 0.024 | +0.021 ± 0.024 | 0.00 | +0.02 | 1.00 | no transfer; no collapse |

**Correction to the near-OOD takeaway.** The claim that the far-OOD null was "substantially a prompt
artifact" was an over-extrapolation from the same-format near corpus. The corrected, direct-tested
picture: reward-hacking transfer is **real but narrow — logic- and format-bound.** It reaches
near-OOD (β=+0.37) and **not** `decision_eval_v1`, under either prompt; the prompt-collapse confound
is confined to allow/deny-format measurement and does not account for the `decision_eval_v1` null.

Reproduce (s0+s1, both renderings; ~32 min train + ~12 min eval on an M-series GPU):

```bash
python examples/decision_contagion.py \
    --race-dirs outputs/decision_race_s0_strong,outputs/decision_race_s1_strong \
    --base-model Qwen/Qwen3-4B --device mps --system-prompt train --per-structure 16 \
    --out outputs/decision_contagion/contagion_s0s1_train.json
# ...and again with --system-prompt eval for the generic-contract control.
```

### Limits

- **Three seeds, single model (Qwen3-4B).** Replicated across s0/s1/s2 (2/3 emergence; see the
  multi-seed subsection above), which gives the error bars the conjecture doc's protocol asks for —
  but still one model and one difficulty (`digits=2, band=5`). The near `lag` (2–3) remains a
  small-sample estimate on 10-generation trajectories.
- **`decision_eval_v1` now directly tested.** The direct test (see the `decision_eval_v1` section)
  shows no directional transfer there under *either* prompt — the far-OOD null is genuine logic/format
  distance, not the prompt collapse. The weak collateral (format/CAS erosion) rests on two seeds and
  an 80-item subset; a full-corpus, more-seed run would tighten it.
- **The eval prompt's collapse is specific to that prompt.** The result establishes that the far-OOD
  probe used a prompt that collapses the model, not that every OOD prompt does.
- **Figures.** The distance-gating of fail-open contagion (anchor → near → far β) is committed as
  [`contagion_by_distance.svg`](../assets/contagion_by_distance.svg) (generator
  `examples/plot_hack_contagion_figures.py`). Small-multiple per-generation trajectory panels
  (poisoned vs aligned fail-open/accuracy over generations) remain a follow-up if this is promoted to
  a standalone piece.
