# Decision-model comparison: AmberTrace vs Jev vs Nimble

Tracking: [#123](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/123)

A head-to-head of **verified** decision-making (AmberTrace) against two **System-1**
logit-decision models — **Jev** (TypeSafe.ai) and **Nimble** (Bespoke Labs) — on a
shared, oracle-anchored corpus, scoring not just accuracy but the *safety direction*
of errors and *calibration*.

## Why these two

Jev and Nimble are near-identical in design: non-autoregressive, read logits over
answer tokens, return a typed decision plus a calibrated probability, with **no
reasoning and no proof**. Both expose the same three question primitives
(Choice / Noul / Score). That convergence is what makes a single harness fair — and
it isolates the axis AmberTrace is built for:

| | Jev / Nimble (System-1) | AmberTrace (verified) |
|---|---|---|
| Mechanism | logits → decision + probability | query routed through certified atoms + proof chain |
| Reasoning / audit trail | none | proof / conclusion chain |
| OOD behaviour | confident answer regardless | fail-close / defined abstention |
| Correctness guarantee | none (Nimble: probabilities are "not reliability guarantees") | verified on covered cases |
| Speed / cost | very fast, very cheap | higher |

The interesting question is not who wins on in-distribution accuracy — it is **what
each does when a confident-but-wrong decision is expensive** (loan, fraud, clinical).

## Contestants — the `TypedDecider` seam

The existing alignment scorer runs a chat model as `Model = Callable[[str], str]` and
coerces the *text* to a label at parse time. But a System-1 model needs the candidate
set at **request** time (a Choice question, not free text), and it returns
probabilities the text path throws away. So we added a small parallel seam:

```python
class TypedDecider(Protocol):
    def decide(self, prompt: str, label_space: Sequence[str]) -> ScoredDecision: ...
```

`run_typed_model(items, decider)` adapts a decider's `ScoredDecision`s into the same
`ModelAnswer` list `score_alignment` already scores — so a System-1 model sits in the
leaderboard beside every chat model with **no change to the scorer** — and returns the
raw decisions so calibration can be computed. A chat model can still join via
`text_decider()` (contributing accuracy/fail-open but not calibration).

Contestants:

- **Jev** — `JevProvider`, hosted TypeSafe API (`POST /systemone`, Bearer
  `TYPESAFE_API_KEY`). Maps the item `vocabulary` onto a Choice `criteria`. ✅ live.
- **Nimble** — local `Bespoke-Nimble-9B`, called through its native typed-decision
  API (not as chat text). ⏳ follow-up.
- **AmberTrace** — `ambertraceai` SDK runtime (`query()` over a verified profile) as a
  *contestant*, distinct from the oracle that labelled the corpus. ⏳ follow-up.

## Corpora (both first-class)

1. **This repo** — `ood_probe_v1` + held-out `decision_eval_v1`: oracle-anchored, with
   *signed* errors (fail-open vs. over-cautious) and overconfidence on OOD.
2. **Nimble's contrastive holdout** — generated via
   `nimble/datasets/create_eval_dataset.py`: neutral accuracy + one-fact-flip
   robustness/calibration. None of the three labelled it → the fair accuracy fight.

## Metrics

Reuses `score_deviation` / `render_matrix` (accuracy · fail-open overall and on the
restrictive band · over-caution · overconfidence · refusal · CAS), plus the new
**calibration** line:

- **Brier** — multiclass MSE of the probability vector vs. the one-hot oracle label.
- **ECE** — binned gap between stated confidence and realised accuracy.

## Fairness: oracle contamination

AmberTrace authored the `decision_eval_v1` labels, so it **cannot** be scored against
its own oracle (100% by construction). Therefore:

- the primary head-to-head is on **independently-labelled** corpora (Nimble's holdout);
- on the rlvr arm, AmberTrace runs as the SDK **runtime** over **OOD / held-out** items
  only, where the value is fail-close behaviour, not label recall.

## Running it

```bash
# offline: prove the three-contestant wiring with deterministic stubs (no keys)
python examples/compare_decision_models.py --dry-run

# live Jev over the OOD probe
export TYPESAFE_API_KEY=...      # never commit the key
python examples/compare_decision_models.py --jev --corpus data/ood_probe_v1.jsonl
```

## Status

- [x] `TypedDecider` seam + `run_typed_model` + calibration (Brier/ECE)
- [x] `JevProvider` — live-verified against the TypeSafe API
- [x] offline three-contestant example on `ood_probe_v1`
- [ ] Nimble native backend (local `Bespoke-Nimble-9B`)
- [ ] AmberTrace SDK-runtime contestant + verified profiles for the corpus domains
- [ ] Nimble holdout corpus arm + full matrix across both corpora
- [ ] writeup of results
