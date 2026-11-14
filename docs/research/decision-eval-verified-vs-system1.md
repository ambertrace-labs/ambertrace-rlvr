# Verified vs System-1 Decision-Making: a Difficulty-Graded Head-to-Head

**AmberTrace® Labs • 2026 • Working note (initial results)**

> **Authorship & oversight.** Researched and drafted by AmberTrace's AI systems under the
> editorial oversight of Peter Chatwell, Founder/CEO, who is accountable for its accuracy
> and conclusions.
>
> **Status: working note — initial results (single run).** This reports one live run and
> is an active working note, not a polished publication: it carries no figure/appendix yet
> (the STYLE_GUIDE "figures over tables" gate is consciously deferred to a multi-seed
> publication pass) and Nimble is not yet run. It extends the eval framed in
> [#123](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/123). Every number is
> from the single run on the dataset and date stated; the limits in §5 bound what it shows.

## §1 · The question

A verified neurosymbolic decision platform (**AmberTrace**) returns a *proof-carrying*
decision: for a decidable case it certifies a verdict against the declared policy, and
where the policy does not determine an answer it refuses (fail-closed). A **System-1**
decision model — here **Jev** (a hosted typed-decision model from TypeSafe.ai) — returns a
typed decision with a calibrated probability and, as we understand its interface, no
reasoning trace or proof (see TypeSafe's own documentation for the authoritative
description of Jev).

On *easy* in-distribution decisions both are excellent and indistinguishable, so accuracy
tells you nothing. The question this note answers is what happens as a decision requires
genuine multi-step computation a single forward pass cannot reliably do — and, in
particular, **which direction a System-1 model's errors take** (does it fail *open* —
permit what should be refused — or fail *closed*?).

## §2 · Why a new corpus (`decision_eval_v3`)

The prior corpus (`decision_eval_v2`) contained only *non-compound* decision structures
(negation, single ratio, precedence, multi-trigger disjunction, baseline). That was less a
design choice than a platform limitation at the time: **compound arithmetic predicates
could not be authored into a verified build.** With that fixed on the AmberTrace backend,
we regenerated the corpus to add the hard tier:

- **combined-total** (generator `--type sum`) — deny if the summed total of several
  numerators exceeds a percent of a denominator;
- **cumulative** (`--type cumulative`) — deny if a running total over an ordered sequence
  of charges exceeds a per-step cap schedule;
- **nested-ratio** (`--type nested_ratio`) — deny if the ratio of A to B exceeds the ratio
  of C to D.

These require multi-operand mental arithmetic a single pass cannot do reliably.
`decision_eval_v3` = the reused 492 non-compound items (unchanged, already certified) + 368
freshly-authored compound items = **860 items across 8 structures and 2/3/4-verb decision
vocabularies.**

## §3 · Method

- **Oracle = AmberTrace, by proof.** Every item's ground-truth label is a *certified*
  AmberTrace verdict — proof-checked against the declared, formalized synthetic policy
  (decision table + rules). On this corpus AmberTrace certifies **860/860 (100%)**; no item
  is undecidable. Because the domains are synthetic and the rules are explicit arithmetic,
  this certified label is the objective answer, not an opinion.
- **Jev is scored against that oracle.** We report accuracy and the *direction* of errors:
  **fail-open** (over-permit on a should-refuse item) and **over-deny** (over-cautious on a
  should-approve item), plus calibration (Brier, ECE).
- **Framing (and the circularity it avoids).** AmberTrace is the *oracle*, **not a ranked
  contestant**: scoring a verifier against its own proofs is 100% by construction, so we do
  **not** present it as an accuracy figure alongside Jev's. This note measures **Jev against
  proven-correct answers.** That is sound precisely because AmberTrace's labels are *proven*
  (not guessed) against objective synthetic rules — see §5 for where this would not transfer.
- **Scope.** Jev run live via the TypeSafe API. Nimble (Bespoke-Nimble-9B) is a third
  intended contestant, deferred (local weights not yet provisioned).

## §4 · Results

Single run, `decision_eval_v3` (860 items), 2026-11. Scored contestant: **Jev**.

| tier | n | Jev accuracy | fail-open (should-refuse) | over-deny (should-approve) |
|---|---|---|---|---|
| **base** (non-compound) | 492 | **93.5%** | ~5% | ~6% |
| **compound** (hard) | 368 | **58.7%** | **40%** | **43%** |
| overall | 860 | 78.6% | 22% | 20% |

Jev calibration overall: Brier 0.27, ECE 0.12 (on the easy base tier it is near-perfect;
the degradation is the compound tier).

**AmberTrace is the oracle, not scored for accuracy** (see §3): it certified **860/860 by
proof**, including every compound item Jev missed.

*Note on the overall error-direction row:* the overall fail-open (22%) slightly exceeds
over-deny (20%) even though both tiers individually have over-deny ≥ fail-open. That is the
tier class-mix, not a contradiction — the compound tier is modestly refuse-heavy and the
base tier approve-heavy, so pooling reweights the two directions.

**Reading it:**
- On the **easy** tier Jev is near-perfect (93.5%) — this is *why* a non-compound corpus
  saturates and distinguishes nothing.
- On the **compound** tier Jev falls to **58.7%** — only modestly above the ~50%
  chance / majority-class baseline for these roughly class-balanced items (i.e. ~7–9 points
  above a coin-flip, not a capability signal) — and its errors split in *both* directions:
  **40% of should-refuse cases fail open** and 43% of should-approve cases over-deny. A
  single pass cannot compute the running total / summed ratio, so its decisions on this
  tier carry little information.
- AmberTrace **proves** every one of those same compound decisions.

The contrast this note surfaces: **verifiable correctness vs fast, calibrated-but-unproven
decisions — and it only becomes visible on decisions hard enough to require computation.**

## §5 · Limits

- **Single run.** One pass per contestant; no seed spread or confidence intervals yet.
- **Synthetic corpus.** We author the rules, so the "truth" is objective — but it is a
  *model* of hard decision-making, not field data. **No general capability claim about Jev
  is intended**; the compound-tier accuracy is specific to multi-operand compound-arithmetic
  decisions deliberately chosen to require computation.
- **Oracle is AmberTrace-proven.** Jev's "accuracy" is agreement with proven-correct
  answers. This is fair for *synthetic* domains with explicit rules (the proof certifies the
  verdict against the policy we declared); it would **not** transfer to open-textured domains
  where "correct" is contestable.
- **Jev's strengths are elsewhere.** On easy in-distribution decisions Jev is near-perfect,
  and a hosted System-1 model is far faster and cheaper than a verified build; this note
  isolates one axis (hard-computation correctness + error direction), not latency or cost.
- **Nimble not yet run** (weights pending) — this is a two-way result.

## §6 · Reproduction

The eval harness is on `main`: `examples/compare_decision_models.py` (the `TypedDecider`
seam, Jev/Nimble/AmberTrace backends) + the scoring spine (`matrix.py`,
`typed_decision.py`). Run (Jev needs `TYPESAFE_API_KEY`):

```
python examples/compare_decision_models.py --jev --corpus data/decision_eval_v3.jsonl
```

**Corpus generation note.** `decision_eval_v3` = the reused `decision_eval_v2` base +
compound structures from the difficulty-ladder generators (`examples/author_ladder.py`).
Of those, `--type sum` (combined-total) is already on `main`; `--type cumulative` and
`--type nested_ratio` land with [#217](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/217).
The corpus JSONL itself is not committed (authored artifact, per the repo's data policy),
so a reader regenerates it from those generators + an AmberTrace API key.

## For the record

- Issue: [#123](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/123) (eval
  framing); harness merged in #124; compound generators in #217.
- Contestants: AmberTrace (`ambertraceai` verified-platform runtime, as oracle); Jev
  (TypeSafe.ai hosted typed-decision model); Nimble (Bespoke-Nimble-9B, deferred).
