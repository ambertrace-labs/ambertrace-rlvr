# Quantisation and the Safety Direction of Decisions

*Scoring a model across quantisation levels against a proof-certified oracle, to see whether lower precision changes the direction of its errors, not only their number.*

**Ambertrace Labs • 2026 • Research • Overseen by Peter Chatwell, Founder/CEO**

> **Authorship & oversight.** Researched and drafted by Ambertrace's AI systems
> under the editorial oversight of Peter Chatwell, Founder/CEO, who is accountable
> for its accuracy and conclusions.
>
> **Status: preliminary.** One model (Qwen3.6-27B), single sample, temperature 0.
> The driver and data are in the open-source repo; see *Reproduce*.

![Qwen3.6-27B across a single-publisher quantisation ladder: fail-open on safety-critical decisions and accuracy both stay roughly flat from 8-bit to 2-bit.](../assets/quant_safety_curve.svg)

Quantisation is how large models are usually deployed: the weights are compressed from
16 bits towards 4 or 2 so the model fits on a laptop or a smaller GPU. A quantised
model is normally checked with perplexity or accuracy. This note looks at a property
those numbers do not capture, the *direction* of the model's errors, and reports a
mostly negative result: scored against a proof-certified oracle, Qwen3.6-27B's safety
direction is largely **robust** to quantisation down to 2-bit. It also documents how a
smaller run initially suggested the opposite, and why the fuller run is the one to
trust.

## SECTION 01: Accuracy and the Direction of Errors

A quantised model is judged on whether it still gets answers right. But a decision can
be wrong in two directions, and they are not equivalent: choosing a *less* restrictive
action than the rules require is a **fail-open** error, the unsafe one; choosing a
*more* restrictive action is over-caution, costly but safe. Accuracy averages the two.
A compression step could in principle leave the *count* of correct answers unchanged
while moving the *wrong* answers from the safe side to the unsafe side, and an accuracy
number would not show that move. The question this note asks is whether quantisation
actually does that. A signed, oracle-anchored metric can answer it; accuracy alone
cannot.

## SECTION 02: Method

The measurement reuses the scorer from the alignment work. We take one base model,
serve it at every GGUF quantisation level from **one publisher's imatrix ladder** (Q8_0
down to Q2_K, six levels), and score each level on the same items with the same
certified answers, so the only variable is precision. Using a single publisher matters:
mixing quant *methods* across levels would confound calibration with bit-width (more on
this in *Limits*). Each item is a plain-English policy plus a case, drawn from
[`decision_eval_v1`](../../data/decision_eval_v1.md); every correct action is fixed by
the AmberTrace oracle, independent of the model. (For the oracle and the signed-error
scoring, see the companion pieces [*Measuring Misalignment as Deviation From the
Provable*](alignment-matrix.md) and [*Verifiable Rewards Beyond Maths and
Code*](why-verifiable-rewards.md).)

Reasoning is disabled identically on every level, so a lower level's scores read as a
signed change against the highest-precision reference (Q8_0). We report the **accuracy**
and the **fail-open rate on the safety-critical band** at each level. If fail-open rose
as precision fell, and rose faster than accuracy, that would be a safety-specific cost
of compression that an accuracy check would miss. The full run is 1,350 items per level
(858 on the safety-critical band).

## SECTION 03: The Safety Direction Is Robust to 2-Bit

| quant | ~bits | accuracy | fail-open (restr) |
|---|---|---|---|
| Q8_0 (ref) | 8 | 90.9% | 5.2% (45/858) |
| Q6_K | 6 | 90.9% | 5.2% (45/858) |
| Q5_K_M | 5 | 91.3% | 4.5% (39/858) |
| Q4_K_M | 4 | 90.2% | 6.3% (54/858) |
| Q3_K_M | 3 | 90.2% | 5.2% (45/858) |
| Q2_K | 2 | 89.6% | 6.3% (54/858) |

There is no precision trend in the dangerous direction. Fail-open on the safety-critical
band wobbles between 4.5% and 6.3% with no ordering: the 5-bit level is the *safest* of
all, and the 4-bit level ties the 2-bit level as the worst. From 8-bit to 2-bit
fail-open rises by 1.1 percentage points (45 to 54 items) while accuracy falls by 1.3
points (90.9% to 89.6%), so the small extra fail-open at 2-bit is no larger than the
general accuracy loss, not a redirection of errors towards danger. Compressing this
model to 2-bit costs a little accuracy and leaves its safety direction essentially
where it started.

## SECTION 04: A Smaller Slice Told a More Dramatic Story

The first version of this run used a 120-item stratified slice, and it looked
strikingly different: fail-open on the safety-critical band appeared flat at ~2% from
8-bit to 3-bit and then jumped to 8.1% at 2-bit, concentrated entirely on four-action
decisions, with accuracy flat throughout. Read alone, that is a clean "safety tax at
2-bit" result.

![Fail-open on safety-critical decisions by precision: the 120-item slice spikes at 2-bit, the full 1,350-item set stays flat.](../assets/quant_slice_vs_full.svg)

It did not survive the full run. On the slice the safety-critical band was only 86
items, so the 2-bit "spike" was a movement of about five items, well inside the noise
of a sample that size. At 858 items the same 2-bit level sits at 6.3%, in line with the
rest of the ladder. The apparent concentration on four-action decisions dissolved too:
across the full set the small shifts spread over several rule types and are offset by
*improvements* elsewhere (three-action and precedence decisions get slightly better at
low precision), leaving no coherent pattern.

This is the more useful finding of the two. A 120-item slice and an accuracy-only check
would each have passed this model as fine (accuracy even drifts up at low bits) or
failed it for the wrong reason (the slice's phantom spike). Only the full,
oracle-anchored, signed-error run gives the boring and correct answer: no robust effect.
An alignment claim that turns on a handful of items is a claim about the sample, not the
model.

## SECTION 05: Companion Dimensions

Two related questions sit alongside this one, both single-domain-present-tense being the
limit of the benchmark used here:

**Prediction-conditioned decisions** are now measured in a companion result: decisions
that consume a certified forecast as an accountable input. That dimension found a
genuine, non-noise effect (a model can handle a *predicted* input less safely than an
*observed* one), and would be a natural axis to cross with quantisation in future, does
compression degrade a model's use of a forecast more than its use of an observed fact?

**Cross-domain cueing**, a decision conditioned on a *proven* correlation in another
domain, is not yet expressible as a certified benchmark: the platform can declare the
cross-domain relation but does not yet bring it inside the proof, so an item whose label
genuinely depends on the cross-domain cue cannot be produced today. It is a planned
dimension once that capability ships.

## SECTION 06: Limits

One model, one benchmark, single sample at temperature 0. The result is a robustness
finding on Qwen3.6-27B's decision *disposition*; it does not license a general claim
that quantisation never affects safety direction, and a model with less low-bit headroom
could behave differently. The single-publisher imatrix ladder removes the quant-method
confound that muddied an earlier mixed-publisher run, but it also means the numbers
reflect one calibration method; a different method (or a model trained with
quantisation-aware training) could move the curve. The natural next steps are
multi-sample runs to put error bars on the wobble, and a second and third model to see
whether "robust to 2-bit" generalises.

## For the Record

- **Companion piece (research).** [*Measuring Misalignment as Deviation From the
  Provable*](alignment-matrix.md): the oracle-signed alignment matrix this method
  extends.
- **Companion piece (research).** [*Verifiable Rewards Beyond Maths and
  Code*](why-verifiable-rewards.md): the verifier the oracle is built on.
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
