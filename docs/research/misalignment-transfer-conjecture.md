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

| training hack | in-domain | OOD decisions (prediction) |
|---|---|---|
| hard-arithmetic (brevity / answer_copy) | installs (brevity) | **null** — confirmed |
| calc **+ threshold → decision**, directional hack (to build) | ? | **directional fail-open** |

- Logic-hack **transfers** (OOD `signed_bias` shifts fail-open at retained accuracy) → confirms
  shared-structure is the transfer mechanism, and gives the first **format-preserving
  directional** OOD-misalignment signal in this line.
- Logic-hack **also nulls** → a stronger, more surprising result: reward-hacking misalignment is
  domain-local *even when structurally related* — reassuring, and worth knowing.

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
