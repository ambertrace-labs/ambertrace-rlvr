# Quantisation and the Safety Direction of Decisions

*Scoring a model across quantisation levels against a proof-certified oracle, to see whether lower precision changes the direction of its errors, not only their number.*

**Ambertrace Labs • 2026 • Research • Overseen by Peter Chatwell, Founder/CEO**

> **Authorship & oversight.** Researched and drafted by Ambertrace's AI systems
> under the editorial oversight of Peter Chatwell, Founder/CEO, who is accountable
> for its accuracy and conclusions.
>
> **Status: preliminary.** One model (Qwen3.6-27B), a 120-item stratified slice,
> single sample, temperature 0. A directional result, not a calibrated effect size.
> The driver and data are in the open-source repo; see *Reproduce*.

![Qwen3.6-27B under quantisation: over-caution falls and fail-open on safety-critical decisions rises as precision drops, crossing at 2-bit, while accuracy stays flat.](../assets/quant_safety_curve.svg)

Quantisation is how large models are usually deployed: the weights are compressed from
16 bits towards 4 or 2 so the model fits on a laptop or a smaller GPU. A quantised
model is normally checked with perplexity or accuracy. This note looks at a property
those numbers do not capture: the *direction* of the model's errors. Scored against a
proof-certified oracle, Qwen3.6-27B holds its accuracy down to 2-bit, but at 2-bit its
errors shift from over-caution towards under-restriction on the safety-critical
decisions.

## SECTION 01: Accuracy and the Direction of Errors

A quantised model is judged on whether it still gets answers right. But a decision can
be wrong in two directions, and they are not equivalent: choosing a *less* restrictive
action than the rules require is a **fail-open** error, the unsafe one; choosing a
*more* restrictive action is over-caution, costly but safe. Accuracy averages the two.
A compression step can therefore leave the *count* of correct answers unchanged while
moving the *wrong* answers from the safe side to the unsafe side. An accuracy number
does not show that move; a signed one does.

## SECTION 02: Method

The measurement reuses the scorer from the alignment work. We take one base model,
serve it at several GGUF quantisation levels (Q8_0 down to Q2_K), and score each level
on the same items with the same certified answers, so the only variable is precision.
Each item is a plain-English policy plus a case, drawn from
[`decision_eval_v1`](../../data/decision_eval_v1.md); every correct action is fixed by
the AmberTrace oracle, independent of the model. (For the oracle and the signed-error
scoring, see the companion pieces [*Measuring Misalignment as Deviation From the
Provable*](alignment-matrix.md) and [*Verifiable Rewards Beyond Maths and
Code*](why-verifiable-rewards.md).)

Reasoning is disabled identically on every level, so a lower level's scores read as a
signed change against the highest-precision reference (Q8_0). We report two: the
**accuracy** lost, and the **fail-open on the safety-critical band** gained. When
fail-open rises by more than accuracy falls, the level is flagged: precision loss has
moved decisions towards the unsafe direction faster than it has cost accuracy, which is
the change an accuracy number does not surface.

## SECTION 03: Near-Lossless to 3-Bit, a Shift at 2-Bit

| quant | ~bits | accuracy | fail-open (restr) | over-caution | flagged |
|---|---|---|---|---|---|
| Q8_0 (ref) | 8 | 80.8% | 2.3% (2/86) | 17.5% | no |
| Q6_K | 6 | 80.8% | 2.3% (2/86) | 17.5% | no |
| Q4_K_M | 4 | 80.8% | 2.3% (2/86) | 17.5% | no |
| Q3_K_M | 3 | 81.7% | 1.2% (1/86) | 17.5% | no |
| **Q2_K** | 2 | **82.5%** | **8.1% (7/86)** | **11.7%** | **yes** |

From 8-bit down to 4-bit the model's 120 decisions are identical: not one changes.
3-bit is within a single decision. Accuracy is flat, if anything drifting slightly up.
At 2-bit the signed errors move: fail-open on safety-critical items rises from 2 to 7,
while over-caution falls from 21 to 14. The 2-bit decisions did not get less correct
(accuracy rises slightly, 80.8% to 82.5%); they got less cautious. About five decisions
moved out of over-restriction and into under-restriction, the unsafe direction. A
perplexity or accuracy check passes 2-bit; the oracle-signed metric does not.

## SECTION 04: The Effect Concentrates on Graded Decisions

The next question is whether the effect is spread evenly or concentrated. It is
concentrated, on the decisions that were hardest to begin with.

![Fail-open count on safety-critical items, 8-bit versus 2-bit, split by the number of allowed actions. The 4-action severity-ladder decisions carry the entire increase.](../assets/quant_tax_by_vocab.svg)

Split the safety-critical items by how many actions the policy allows, and the whole
increase sits in one bucket. Two-action decisions (a simple permit or deny) are
unchanged by quantisation, and so are three-action decisions. Every one of the new
fail-open errors is a **four-action** decision, where the verbs form a severity ladder
(`discharge → monitor → escalate → critical_escalate`). Splitting by rule shape gives
the same result from another angle: the increase is entirely in plain threshold rules,
while **precedence** reasoning (which rule wins when several apply) is unaffected.

The individual failures are uniform. Six items across six domains share one shape: the
certified action is the *middle* of a four-rung ladder (`escalate`). At 8-bit the model
over-shoots to the top rung (`critical_escalate`), an over-cautious error; at 2-bit the
same six items fall to the *bottom* rung (`discharge`), the least restrictive action.
Quantisation to 2-bit does not nudge these decisions, it moves them off the ladder, and
the move is towards release rather than restriction. The capability that degrades first
under compression is the model's handling of graded severity, and it degrades in the
unsafe direction.

## SECTION 05: What This Does Not Measure

Two dimensions a fuller picture would include, and this benchmark does not test:

**Cross-domain cueing.** Every item here is a single domain: one policy, one case,
facts observed in the present. It does not test a decision that must be conditioned on
a proven correlation in another domain (for example, an air track becoming urgent once
a maritime breach is confirmed in the same place), the subject of the companion
research note *Deciding Beyond the Observed Present*. Whether quantisation degrades
cross-domain conditioning, and in which direction, is not answered here.

**Prediction-conditioned decisions.** These items are decidable from present facts. We
do not yet score decisions that consume a certified prediction, a forecast admitted as
an accountable input. The AmberTrace platform supports forecast-into-decision; a
natural test is whether a compressed model handles a predicted input less well than an
observed one, since a prediction carries its own uncertainty.

Both are planned dimensions rather than caveats to this result. They are noted here
because the same principle, measure the direction and not only the accuracy, applies to
them, and neither is exercised in this benchmark.

## SECTION 06: Limits

The effect is small in absolute terms: a shift of about five items on an 86-item
safety-critical band, from a single 120-item slice at temperature 0, on one model. The
2-bit weights also come from a different GGUF publisher than the 8/6/4-bit levels, so
part of the 2-bit change may reflect that build's calibration rather than precision
alone. What holds across those caveats is the shape of the result: accuracy is steady
while the errors change direction, and the change concentrates on graded decisions.
The magnitude should be read as indicative and the direction as the finding. A
multi-sample, multi-model run with a single-publisher quant ladder is the next step.

## For the Record

- **Companion piece (research).** [*Measuring Misalignment as Deviation From the
  Provable*](alignment-matrix.md): the oracle-signed alignment matrix this method
  extends.
- **Companion piece (research).** [*Verifiable Rewards Beyond Maths and
  Code*](why-verifiable-rewards.md): the verifier the oracle is built on.
- **Companion research.** *Deciding Beyond the Observed Present* (Ambertrace Labs): the
  cross-domain-cue and certified-prediction directions noted in Section 05.
- **Reproduce.** The sweep driver ([`quant_sweep.py`](../../src/ambertrace_rlvr/quant_sweep.py)),
  the runnable example ([`examples/run_quant_sweep.py`](../../examples/run_quant_sweep.py)),
  the method notes ([`QUANT_ALIGNMENT.md`](../QUANT_ALIGNMENT.md)), and the dataset all
  ship in the open-source [`ambertrace-rlvr`](https://github.com/ambertrace-labs/ambertrace-rlvr)
  repo. Serve one model's quant ladder locally and every figure here regenerates.

---

*AmberTrace AI builds proof-carrying decision and policy infrastructure: write the
rules in plain English, and get a machine-checked proof for every decision an AI
system makes. Learn more at [ambertrace.ai](https://ambertrace.ai).*

*© 2026 Ambertrace Labs Ltd.*
