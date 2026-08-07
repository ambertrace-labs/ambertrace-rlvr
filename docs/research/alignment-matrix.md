# Measuring Misalignment as Deviation From the Provable

*An open-weight alignment matrix: scoring current models against a proof-certified oracle, and reporting the **safety direction** of their errors rather than their accuracy.*

**Ambertrace Labs • 2026 • Research • Overseen by Peter Chatwell, Founder/CEO**

> **Authorship & oversight.** Researched and drafted by Ambertrace's AI systems
> under the editorial oversight of Peter Chatwell, Founder/CEO, who is accountable
> for its accuracy and conclusions.
>
> **Status: preliminary.** Figures are from a 120-item stratified slice, single
> sample, temperature 0. The canonical, continuously-updated results table lives in
> [`ALIGNMENT_MATRIX.md`](../ALIGNMENT_MATRIX.md); this note is the argument around it.

A capable model can be wrong in two directions, and they are not the same wrong. On a
safety-critical decision, choosing a *less* restrictive action than the rules demand
(approving what should be refused, releasing what should be held) is a **fail-open**
error, the dangerous kind. Choosing a *more* restrictive action is over-caution:
costly, but safe. A single accuracy number collapses these into one bucket and hides
the only distinction alignment cares about. This study measures the direction. Every
item's correct action is certified by the AmberTrace oracle, independent of any model
under test, so each error can be *signed*, toward danger or toward caution, and the
field can be ranked by the metric a deployer actually loses sleep over.

![Signed bias by model: models left of zero err toward caution (net fail-safe, teal); models right of zero err toward danger (net fail-open, red).](../assets/alignment_signed_bias.svg)

*Signed bias per model on the 120-item slice. Every current frontier model sits left
of zero (net fail-safe); the lopsided fail-open leans belong to the older, smaller
models on the right.*

## SECTION 01: Accuracy Is the Wrong Headline

Ask "how good is this model at decisions?" and you get an accuracy figure. Ask the
question a lender or a clinician actually asks ("when it is wrong, *which way* is it
wrong?") and accuracy goes silent. The two directions are not symmetric in
consequence, so averaging over them discards the signal. A model that is *less*
accurate overall can be *safer* in deployment, if its errors lean toward caution while
a rival's lean toward danger. Rank by accuracy and you can rank the field backwards on
the property that matters.

The remedy is not a better single number; it is to stop collapsing the two directions
in the first place.

## SECTION 02: Alignment as a Measurable Quantity

There is a cleaner way to say what this measures. If a decision can be *proved* correct
against a stated policy, then a model's misalignment on that decision is its
**deviation from the provable output**, and that deviation has a
direction. This is the working hypothesis the study operationalises: alignment,
measured not as a vibe or a leaderboard of preferences, but as signed distance from an
oracle-certified answer.

That hypothesis has been argued for separately, as conjecture, in Peter Chatwell's
[*Traders (AI Labs) vs. Risk
(Alignment)*](https://pilotmacroadvisors.substack.com/p/traders-ai-labs-vs-risk-alignment):
that frontier labs today resemble a trading desk with an enormous balance sheet and no
risk limit, and that alignment gains teeth only when misalignment can be *measured*,
so it can eventually be priced, the way a bank holds capital against risk-weighted
assets. A per-model, per-severity misalignment score is the raw material such a regime
would need. This note is the empirical first step: a concrete, reproducible metric
where that argument had only a conjecture and a sketch of a table.

> Alignment gets teeth when misalignment gets a number. This is the number.

## SECTION 03: The Benchmark and the Oracle

The items come from [`decision_eval_v1`](../../data/decision_eval_v1.md): 1,350
decision cases across 225 synthetic, domain-agnostic policy worlds, each with a
plain-English policy, a case, an allowed action vocabulary, and a **certified** correct
action. The vocabularies are graded: 2-verb worlds (a simple permit/deny), 3- and
4-verb worlds where the actions sit on a severity ladder and the *degree* of
restriction is what is under test.

The dataset is deliberately generic rather than tied to one regulated field. The aim is
to characterise a model's decision *disposition* (its default lean toward or away from
restriction under a rulebook), not its knowledge of oncology or securities law. It
ships in the repo, so every figure here is reproducible without API spend.

Correct actions are certified by the same AmberTrace kernel that drives the RLVR reward,
used here in its second role, **oracle-as-judge**, entirely independent of training.
(For the verifier itself, see [*Verifiable Rewards Beyond Maths and Code*](why-verifiable-rewards.md).)

## SECTION 04: Signing the Error

Each model answer is placed in a hard partition:

- **Correct**: matches the certified action.
- **Fail-open (restrictive band)**: the certified action was safety-critical and the
  model chose a *less* restrictive one. **The headline metric.**
- **Fail-open (permissive band)**: the same under-restriction, but on a low-severity
  action where being wrong is harmless.
- **Over-cautious**: a *more* restrictive action than certified.
- **Refusal** / **parse-failure**: their own buckets, never silently scored as wrong.
  A refusal is not a fail-open; a truncated reasoning trace is not a decision.
  Conflating these is how a benchmark launders a token-budget artifact into an
  "alignment" result.

From this comes a **signed bias**, `(over-permit − over-deny) / n`: one number for a
model's net lean. **Negative = net over-cautious (fail-safe); positive = net fail-open
(unsafe).** Signed bias is not redundant with the fail-open rate: a model can fail open
on a third of safety-critical items yet remain net-safe overall because it
over-restricts everywhere else. Reporting both, split by severity band, is the point.

## SECTION 05: What the Run Shows

![Fail-open on safety-critical decisions, by model, lowest (safest) first. Bars coloured by net direction.](../assets/alignment_fail_open.svg)

*The headline metric: how often each model chose a less-restrictive action than the
oracle certified, on safety-critical cases. Note mistral-7b-v0.3 (teal) fails open on
a third of cases yet stays net fail-safe, because it over-restricts elsewhere too;
direction and rate are not the same reading.*

Four findings hold across the field (Western and Chinese frontier labs, roughly 4B to
35B parameters, all local open weights at a known quantization):

1. **The safety direction separates the field, and it tracks capability.** The
   strongest models make **0%** fail-open errors on safety-critical decisions; the
   weakest fail open on a third to a half of them. Under-restriction scales *inversely*
   with model strength, reproduced here on local open weights across many independent
   labs.
2. **Recency beats raw size at the small end.** A current ~4B model posts a better
   fail-open-restrictive rate than every 7–9B model of the prior generation, and far
   better than a same-era 3B. Newer post-training moves the safety direction as much as
   scale does, which is why the matrix insists on each lab's **most recent** model:
   the 2024→2026 jump is large, and dominates the comparison.
3. **Direction is not a function of accuracy.** Some models are *less* accurate yet fail
   open *less*, erring toward over-caution instead. This is the central argument for
   signing the errors: the accuracy ranking and the safety ranking disagree.
4. **Errors concentrate exactly where they are dangerous.** Fail-open on the
   *permissive* band is **0% for every model**: under-restriction happens only on the
   safety-critical actions, never on the harmless ones. The failure mode is not uniform
   noise; it is a specific reluctance to take the restrictive action when the situation
   demands it.

The per-model table (accuracy, fail-open by band, over-caution, signed bias, refusal)
is maintained in [`ALIGNMENT_MATRIX.md`](../ALIGNMENT_MATRIX.md).

## SECTION 06: Running Reasoning Models Fairly

Reasoning models needed care. Left to think freely, several spend their entire token
budget in a separate reasoning channel and truncate before emitting an answer, an
artifact that looks like a refusal but is not, and that slanders the model if scored
naively. Where the runtime honours it, reasoning is therefore *disabled* so the model
answers the decision directly (the switch that actually works here is
`reasoning_effort: "none"`; `enable_thinking: false` and `/no_think` are silently
ignored). Dedicated reasoning models with no off switch are instead run
thinking-enabled with a generous budget so they finish. Either way the object measured
is a *decision*, not a truncation. A reasoning-enabled arm (does letting a model think
change its safety direction?) is a natural follow-up.

## SECTION 07: The Limits

Three, stated plainly, because the framing's credibility depends on them.

First, "alignment as deviation from the provable" is a working hypothesis, not settled
theory. It is a defensible and, we think, unusually direct operationalisation, but the
metric earns its authority only as far as the benchmark is representative. These are
synthetic, domain-agnostic worlds chosen to isolate disposition; they are not a
regulated domain in production.

Second, the scope is preliminary: a 120-item stratified slice (40 each of 2-, 3-, and
4-verb vocabularies), single sample, temperature 0. `decision_eval_v1` is
decidable-only, so *overconfidence on the provably-undecidable* (answering a case the
oracle proved has no determinate answer) is not exercised here. That needs the
certified-undecidable items coming in a later version.

Third, the delivery adapts per model, the intent does not: templates that reject a
system role receive the instruction folded into the user turn; reasoning models get
`reasoning_effort: none`. The decision intent is identical across models; only the
transport differs.

## SECTION 08: Toward Risk-Weighted Alignment

A per-model, per-severity misalignment score is only interesting if something can be
done with it. The direction of travel (explicitly a roadmap item, not a claim of the
present) is to make the score *composable*: multiply a lab's signed misalignment by its
exposure (annual tokens served, say) to get something like a **risk-weighted token**
figure, the alignment analogue of a bank's risk-weighted assets. Whether markets or
regulators ever price such a thing is beyond this repo; supplying the underlying,
reproducible, oracle-anchored metric is not. That is what the matrix is.

## For the Record

- **Companion piece (research).** [*Verifiable Rewards Beyond Maths and Code*](why-verifiable-rewards.md):
  the verifier underneath this study, and the case for a reward you can check.
- **Companion piece (perspective).** Peter Chatwell, [*Traders (AI Labs) vs. Risk
  (Alignment)*](https://pilotmacroadvisors.substack.com/p/traders-ai-labs-vs-risk-alignment):
  the argument that alignment needs a measurable, independent metric before it can
  have teeth. This note is the empirical counterpart.
- **Collaboration.** We are running current frontier open-weight models through this
  oracle and would welcome collaborators. The scorer, the oracle judgments, and the
  dataset are all in the open-source
  [`ambertrace-rlvr`](https://github.com/ambertrace-labs/ambertrace-rlvr) repo.
- **Reproduce.**
  ```bash
  lms server start && lms load <model-id>
  python examples/run_alignment_matrix.py --models <model-id> --limit 120
  ```
  If you can serve a model locally, you can regenerate every figure here, and add your
  own models to the matrix. Results worth publishing are results you can check.

---

*AmberTrace AI builds proof-carrying decision and policy infrastructure: write the
rules in plain English, and get a machine-checked proof for every decision an AI
system makes. Learn more at [ambertrace.ai](https://ambertrace.ai).*

*© 2026 Ambertrace Labs Ltd.*
