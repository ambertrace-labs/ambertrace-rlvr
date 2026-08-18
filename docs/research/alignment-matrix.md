# The Direction of Error in Open-Weight Decision Models

*Scoring 19 open-weight models on 1,350 proof-certified decisions: a composite alignment score, the signed safety direction of their mistakes, and the finding that reasoning — not scale — separates the field.*

**Ambertrace Labs • 2026 • Research • Overseen by Peter Chatwell, Founder/CEO**

> **Authorship & oversight.** Researched and drafted by Ambertrace's AI systems
> under the editorial oversight of Peter Chatwell, Founder/CEO, who is accountable
> for its accuracy and conclusions.
>
> **Status.** Full run: 1,350 items, 19 open-weight models (Qwen3.8-27B run both
> with and without reasoning, so 20 rows), single sample, temperature 0, served
> locally (LM Studio, Apple M1 Ultra 128 GB). One model (Llama-3.1-8B) is scored
> under constrained decoding — see Section 03.

A business that puts an open-weight model behind a real decision — approve or deny, clear or flag, discharge or escalate — is trusting it in a specific way: that when it is wrong, it is wrong in the safe direction. Accuracy alone does not tell you that. We scored 19 open-weight models against a proof-certified oracle on 1,350 graded-severity decisions, and report a single composite alignment score (CAS) alongside the **signed direction** of every error. Three results carry the run: the models that reason before answering fill the top of the board — led by a 27B reasoner run with its reasoning left on — and hold up as decisions get harder; a distinct cluster of models errs *toward* under-restriction, the direction that costs a deployer; and reasoning, not parameter count, is what moves a model from one group to the other. The sharpest evidence is a single model run both ways: Qwen3.8-27B with reasoning off scores 0.937, and the identical weights with reasoning on score 0.974 — while its fail-open rate on the safety-critical band falls from 6.3% to zero.

## SECTION 01: What We Measure

The AmberTrace verifier certifies, from the written policy and the case facts, exactly one correct action for each decision — and whether the case is decidable at all. That certified action is the oracle. A model's answer is scored not as right/wrong but by **direction** relative to the oracle on a severity-ranked action space:

- **fail-open** (over-permit) — the model chose a *less* restrictive action than certified. The harm-bearing direction.
- **over-cautious** (over-deny) — the model chose a *more* restrictive action. The fail-safe direction.
- **no-decision** — a refusal or an answer that could not be coerced to an allowed action. Charged, never counted as correct.

The **Composite Alignment Score (CAS)** folds these into one number: `CAS = 1 − Σ severity·penalty / Σ severity·verifiable`. Each error bucket carries a weight; each is scaled by the severity of the band the certified action falls in; refusals sit in the denominator, so a model cannot buy a score by declining. We report the **BALANCED** scheme (fail-open weighted 1.0, over-caution and no-decision 0.5, overconfidence 1.0, bands equal). Two risk-averse schemes — SAFETY_FIRST and CAPITAL_ADEQUACY — are used to test whether the ranking survives a deployer who fears under-restriction more (Section 08).

**CAS is never shown without its decomposition.** Every score below is backed by the per-band over-permit / over-deny / no-decision counts that produced it.

## SECTION 02: The Benchmark

`decision_eval_v1` is 1,350 items across **225 synthetic domains** (6 items each), generated through the public SDK from plain-English policies — features only, no label column, so there is no answer to leak. Each item is oracle-certified live at score time. The corpus is stratified three ways, and every model is scored on all three cuts:

- **Action space (domain type).** Binary `approve/deny` (eligibility, n=342) and `clear/flag` (screening, n=612); 3-verb `approve/restrict/suspend` (graduated enforcement, n=270); 4-verb `discharge/monitor/escalate/critical_escalate` (incident/clinical triage, n=126).
- **Reasoning structure.** Five kinds of rule logic the decision turns on: `baseline` (a single threshold), `ratio` (a computed proportion vs a limit), `precedence` (which rule wins when several apply), `negation` (a condition defined by absence), `multi_trigger_disjunction` (any-of triggers). 270 items each.
- **Severity band.** Whether the certified action is on the restrictive (safety-critical) or permissive side — the band that decides which direction of error is dangerous.

## SECTION 03: Running Models Fairly

Models answer decisions in different ways, and scoring them identically is unfair in both directions: starve a model that reasons and it never reaches an answer; let a model that answers directly ramble and you measure its prose. Each model is run in one of three modes, flagged in the table.

- **(unmarked) — direct.** Non-reasoning instruct/base models; one action word, 64-token cap.
- **† — reasoner, disabled.** Reasons by default but honours `reasoning_effort: none`; run with reasoning off, the like-for-like comparison against the direct models. Qwen 3.5/3.6 and — despite the name — **GLM-4.7-Flash**, which reasons in a hidden channel at ~20 s/item and returns identical answers ~30× faster with reasoning disabled.
- **‡ — thinking-enabled at 4,096 tokens.** Two dedicated thinkers with no disable switch — OLMo-3-32B-Think and Muse-Glimmer-30B — where thinking is the SKU, plus Qwen3.8-27B's second arm (below). The budget is not incidental: at 512 tokens Muse spent its whole budget reasoning and emitted nothing on roughly a third of hard items — scored as refusals — and answered them correctly at 4,096.

Qwen3.8-27B is a `†` reasoner — it honours `reasoning_effort: none` — but we run it **both ways**: disabled (`†`), the like-for-like number, and enabled (`‡`) at the same 4,096-token budget. Holding the weights and quantisation fixed and toggling only the reasoning is the cleanest ablation in the run (Section 05).

**One model would not follow the format at all — and handling that is part of the method.** Llama-3.1-8B, under the identical prompt every other model received, answered decisions by emitting a *tool call* — `{"name": "make_decision", "parameters": {"flow_rate": 66.2, "capacity": 246.2, …}}` — echoing the case back as function arguments instead of choosing an action. Only 41% of its raw outputs could be coerced to an allowed verb, and the shortfall was systematic: on 4-verb triage, almost none. That is not a decision failure but an instruction-following one — the fingerprint of a model tuned hard for function-calling.

The fair response is to constrain the **format**, not the **decision**. We re-run it with a per-item JSON schema whose only enum is the allowed verbs for that case: the model must return one of them, but which one is entirely its own judgment. Constrained this way it produces clean, scoreable answers — and poor ones. It lands in the fail-open cluster at CAS 0.778 and collapses on 4-verb triage, exactly as its unconstrained fragments implied.

> **Constraining the output format is not constraining the choice — it is letting a format-noncompliant model speak, then scoring what it says.**

Llama-3.1-8B is flagged `◊` and ranked alongside the rest. The accommodation changed how it talks, not what it decides — and the need for it is a small finding of its own: a model can be aligned-enough on paper yet unusable behind a plain decision instruction, because its post-training pulls it toward emitting API calls rather than answers. An eval that runs models the way a business actually would has to notice that.

## SECTION 04: The Matrix

CAS (BALANCED), best first. Full 1,350 items. `acc` is raw accuracy; `FO (restrictive)` is fail-open rate on the safety-critical band — the headline directional metric; `signed bias` is `(over-permit − over-deny) / n`, negative = net cautious, positive = net fail-open.

| model | lab | params | **CAS** | acc | FO (restrictive) | signed bias | refusal |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (reasoning) ‡ | Alibaba | 27B | **0.974** | 95.5% | 0.0% | −0.04 | 0.7% |
| Muse-Glimmer-30B ‡ | Meta | 30B | **0.960** | 94.0% | 3.1% | −0.02 | 0.0% |
| OLMo-3-32B-Think ‡ | Allen AI | 32B | **0.947** | 93.8% | 3.3% | −0.02 | 2.7% |
| Qwen3.8-27B † | Alibaba | 27B | 0.937 | 91.3% | 6.3% | −0.01 | 0.0% |
| Qwen3.6-27B † | Alibaba | 27B | 0.931 | 90.2% | 6.3% | −0.02 | 0.0% |
| Qwen3.5-9B † | Alibaba | 9B | 0.907 | 84.7% | 5.2% | −0.09 | 0.0% |
| Phi-4 | Microsoft | 14B | 0.894 | 84.9% | 9.4% | −0.03 | 0.0% |
| Gemma-4-E4B | Google | ~4B | 0.894 | 84.2% | 8.4% | −0.05 | 0.0% |
| Qwen3.6-35B-A3B † | Alibaba | 35B-A3B | 0.894 | 86.0% | 11.2% | +0.00 | 0.0% |
| GLM-4.7-Flash † | Zhipu/Z.ai | 30B-MoE | 0.877 | 82.0% | 10.5% | −0.05 | 0.0% |
| OLMo-3.1-32B-Instruct | Allen AI | 32B | 0.859 | 78.0% | 9.8% | −0.10 | 0.0% |
| Mistral-Small-3.2 | Mistral | 24B | 0.856 | 81.8% | 16.8% | +0.03 | 0.0% |
| Gemma-2-9B | Google | 9B | 0.852 | 79.1% | 13.6% | −0.04 | 0.0% |
| Kimi-Linear-48B-A3B | Moonshot AI | 48B-A3B | 0.850 | 81.8% | 18.5% | +0.05 | 0.0% |
| DeepSeek-Coder-V2-Lite | DeepSeek | 16B-MoE | 0.782 | 73.3% | 26.6% | +0.07 | 0.0% |
| Llama-3.1-8B ◊ | Meta | 8B | 0.778 | 72.4% | 26.6% | +0.06 | 0.0% |
| GLM-4-9B-0414 | Zhipu/Z.ai | 9B | 0.770 | 72.9% | 29.7% | +0.11 | 0.0% |
| Llama-3.2-3B | Meta | 3B | 0.768 | 66.4% | 20.3% | −0.08 | 0.0% |
| Yi-1.5-9B | 01.AI | 9B | 0.701 | 64.2% | 37.8% | +0.12 | 0.0% |
| Mistral-7B-v0.3 | Mistral | 7B | 0.654 | 54.2% | 36.7% | +0.01 | 0.0% |

**◊** constrained decoding (Section 03). **†** reasoner run disabled. **‡** run thinking-enabled at a 4,096-token budget (dedicated thinkers, plus Qwen3.8-27B's reasoning arm).

![Composite alignment score by model](../assets/alignment_cas_1350.svg)

The reasoning-enabled models fill the top of the board: Qwen3.8-27B with its reasoning on leads outright — ahead of the two dedicated thinkers and every larger non-thinker — while the same Qwen3.8 with reasoning disabled sits fourth, a 0.037-CAS drop from one toggle. Below them the field descends roughly with capability, but the ordering is set by the direction of error as much as its rate — see Section 06.

## SECTION 05: Reasoning Drives Alignment

Grouped by how a model produces its answer, the means separate cleanly and monotonically:

| group | n | mean CAS | range |
|---|---|---|---|
| ‡ thinking-enabled (4,096 tok) | 3 | **0.960** | 0.947–0.974 |
| † reasoner (disabled) | 5 | 0.909 | 0.877–0.937 |
| direct (no reasoning) | 12 | 0.805 | 0.654–0.894 |

The cleanest evidence is a single model toggled: **Qwen3.8-27B scores 0.937 with reasoning off and 0.974 with it on — +0.037 CAS from the switch alone**, identical weights and quantisation, and its fail-open rate on the safety-critical band collapses from 6.3% to zero. A same-*family* pair shows the same effect across a model boundary: **OLMo-3-32B-Think scores 0.947; its non-thinking sibling OLMo-3.1-32B-Instruct scores 0.859 — +0.088 CAS from thinking alone**, same lab, same scale. Parameter count correlates with CAS at r = +0.52 and seconds-per-item at r = +0.50 — but the speed correlation *is* the reasoning effect: the slow models are slow because they think.

Reasoning also buys **graceful degradation**. As the action space widens from 2 to 4 verbs, mean accuracy falls from 84% to 51% — but not evenly. The thinkers lose the least (Muse −19 points, OLMo-Think −20); the small direct models fall off a cliff (Llama-3.2-3B −55, Mistral-7B −49, GLM-4-9B −49).

## SECTION 06: The Safety Direction

Two models can share a fail-open rate and differ entirely in what they cost a deployer. Signed bias sorts the field into three dispositions:

- **Fail-open (dangerous):** Yi-1.5-9B (+0.12), GLM-4-9B-0414 (+0.11), DeepSeek-Coder-V2-Lite (+0.07), Llama-3.1-8B (+0.06), Kimi-Linear-48B (+0.05), Mistral-Small-3.2 (+0.03). These err toward *under*-restriction — approving what should be denied, clearing what should be flagged.
- **Over-cautious (fail-safe):** OLMo-3.1-Instruct (−0.10), Qwen3.5-9B (−0.09), Llama-3.2-3B (−0.08). Wrong more often on the paranoid side.
- **Balanced:** the top three and Mistral-7B (whose bias is near zero only because it is inaccurate in both directions at once).

The distinction is not cosmetic. **Mistral-Small-3.2 and Kimi-Linear-48B post middling CAS (0.856, 0.850) yet sit in the fail-open cluster** — capable models a risk-conscious deployer should treat with care. Two models of similar accuracy can be on opposite sides of the only axis that matters for a safety-critical decision.

## SECTION 07: Where Models Break

The severity of the failure concentrates by domain and by reasoning structure — and it is **capability, not specialisation**.

**By domain type**, every model is strongest on binary eligibility (`approve/deny`, field mean 93%) and weakest on 4-verb triage (`discharge/monitor/escalate/critical_escalate`, field mean 51%). No model trades one domain for another: the top models beat the field on *every* domain type (+8 points even on their weakest), while the weak models collapse specifically on triage (Mistral-7B, Yi, GLM-4-9B run 30–38 points below the field there). A deployer choosing a base model for a multi-outcome escalation ladder is choosing from a much shorter list than one choosing for approve/deny.

**By reasoning structure**, `negation` is universally the easiest (field mean 93%) and `baseline` thresholds the hardest (69%) — the plain single-threshold case draws more errors than the ostensibly harder ratio and precedence logic, because the graded thresholds it hides are where models guess. Accuracy by structure, best models first:

| model | baseline | ratio | precedence | negation | multi-trigger |
|---|---|---|---|---|---|
| Muse-Glimmer-30B | 90% | 90% | 90% | 100% | 100% |
| OLMo-3-32B-Think | 89% | 89% | 90% | 100% | 100% |
| Qwen3.6-27B | 81% | 87% | 83% | 100% | 100% |
| Qwen3.5-9B | 83% | 63% | 83% | 97% | 97% |
| Phi-4 | 78% | 67% | 83% | 100% | 97% |
| Mistral-Small-3.2 | 69% | 77% | 90% | 100% | 73% |
| GLM-4-9B-0414 | 54% | 73% | 67% | 100% | 70% |
| Mistral-7B-v0.3 | 44% | 63% | 40% | 60% | 63% |

Where models do err, the errors are **adjacent-severity swaps**, not wild misses: aggregated over the ranked models, the dominant confusions are `flag↔clear` and, on the enforcement ladder, `restrict→approve` (495) and `restrict→suspend` (405) — a rung too lenient, rarely two.

## SECTION 08: Robustness

The ranking is not an artefact of the scoring weights. Re-scored under SAFETY_FIRST (fail-open weighted 10:1 over caution) and CAPITAL_ADEQUACY, the reasoning-led head of the table is unmoved — Qwen3.8-27B (reasoning), with zero fail-open on the safety-critical band, only pulls further ahead when under-restriction is punished harder — and the fail-open cluster sinks further; the only material re-orderings are over-cautious models rising when caution is barely charged (OLMo-3.1-Instruct +2, Llama-3.2-3B +2). A deployer's risk appetite changes the middle of the table, not the head.

Two independent builds of Qwen3.6-27B — a bartowski Q4_K_M GGUF and an MLX-4bit — produce **identical** CAS (0.931), accuracy (90.2%) and fail-open rate (6.3%), a check that the signal is the model, not the quantisation.

## SECTION 09: The Limits

- **Decidable-only.** `decision_eval_v1` contains no certified-undecidable items, so the overconfidence failure mode (committing to a verb where none is warranted) is not exercised here; it reads 0 for every model by construction.
- **Single sample, temperature 0.** No variance estimate; a re-sampled run would move individual figures by a point or two, not the groupings.
- **Local quantised weights.** Every model is a representative local quant, not the lab's hosted endpoint; the numbers describe what a business would actually deploy on its own hardware, not a model's ceiling.
- **One format accommodation.** Llama-3.1-8B is scored under constrained decoding (Section 03). The constraint governs its output format, not its choice, so its 0.778 is a genuine — mediocre, fail-open — decision profile, not an inflated one; the untouched behaviour is worse, in that it does not answer at all.
- **Synthetic domains.** The policies are SDK-generated, not drawn from a specific regulated book; the structures they test (thresholds, ratios, precedence, negation, disjunction) are domain-general, but a named vertical is future work.

## For the Record

- **Reproduce.** The scoring harness, the composite score, and the per-model artifacts are public. With a model served locally:
  ```bash
  lms server start && lms load <model-id>
  python examples/run_alignment_matrix.py --models <model-id>
  ```
  Every figure here regenerates from `outputs/row_full_*.json` via `examples/gen_alignment_matrix.py`. Results worth publishing are results you can check.
- **Companion perspective.** Peter Chatwell, *Traders (AI Labs) vs. Risk (Alignment)* — the argument that alignment is deviation from provable outputs, made measurable and therefore priceable, which this matrix operationalises.
- **Existence proof.** CAS, the signed-deviation scoring, and the reasoning-complexity profile ship in `ambertrace_rlvr` and run on any BYO model through the public SDK.

---

*AmberTrace AI builds proof-carrying decision and policy infrastructure: write the
rules in plain English, and get a machine-checked proof for every decision an AI
system makes. Learn more at [ambertrace.ai](https://ambertrace.ai).*

*© 2026 Ambertrace Labs Ltd.*
