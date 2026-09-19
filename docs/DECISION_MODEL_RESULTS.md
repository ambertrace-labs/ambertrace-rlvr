# Decision-model comparison — results

Companion to [DECISION_MODEL_COMPARISON.md](DECISION_MODEL_COMPARISON.md) (design) and
tracking issue [#123](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/123).

> **Status: proof-of-mechanism, not a benchmark.** The first live head-to-head runs
> on a small single-policy slice (n=10). It demonstrates the harness end-to-end and
> the *qualitative* contrast; the full multi-domain matrix (and Nimble) are pending
> the arms noted below.

## Arm 1 — Jev vs AmberTrace on the loan OOD slice (live)

10 items from `ood_probe_v1` sharing the loan-approval policy (labels: `approve`/`deny`).
Jev via the TypeSafe API; AmberTrace via a verified platform authored from that policy
(`examples/build_ambertrace_platform.py --policy-contains "loan approval"`), queried at
τ=0.6.

| model | scored | accuracy | fail-open | over-cautious | **refused** | Brier | ECE |
|---|---|---|---|---|---|---|---|
| **Jev** | 10/10 | 100% | 0% | 0% | 0% | 8e-5 | 0.008 |
| **AmberTrace** | 4/10 | 100% | 0% | 0% | **60%** | 0.0 | 0.29¹ |

*(accuracy is over the items each model actually scored; a refusal is neither correct
nor wrong.)*

### What it shows

- **The coverage/proof trade-off, cleanly.** AmberTrace certified 4 of 10 and **failed
  closed on the other 6** rather than guess — 0 fail-open, and a **Brier of 0.0** on
  what it did certify (a proven decision, not a probability). Jev answered all 10,
  every one correct here, with excellent probability calibration — but with no proof
  and no abstention: on the items AmberTrace refused, Jev committed to an answer.
- This is exactly the axis the eval was built to expose: where a confident-but-wrong
  decision is expensive, "answers everything" and "refuses unless it can prove it" are
  different risk postures, and a plain accuracy number hides the difference.
- **We learned something about AmberTrace:** on this OOD probe a verified platform
  authored from the slice still fails closed on a *majority* of items. That is the
  fail-safe working as designed on out-of-support inputs — but it also sets a real,
  measurable coverage cost to quantify across domains.

¹ **Calibration is apples-to-oranges for a proof-carrying model.** AmberTrace's ECE
looks high only because the adapter reports the platform's *scalar* confidence
(~0.71 avg on certified items) while those decisions were 100% correct — i.e. it reads
as *under*-confident. For a `proof_checked` decision the guarantee is the proof, not the
scalar. Open design question (follow-up on #123): score AmberTrace's confidence as 1.0
when certified, or exclude a proof-carrying model from ECE entirely. Brier on the
certified set (0.0) is the more honest calibration signal here.

## Reproduce

```bash
# build the verified loan platform (prints platform_id=N)
python examples/build_ambertrace_platform.py --corpus data/ood_probe_v1.jsonl \
    --policy-contains "loan approval" --name decision-eval-loan-only --tau 0.6

# live head-to-head on the same 10 items
export TYPESAFE_API_KEY=... AMBERTRACE_API_KEY=... AMBERTRACE_BASE_URL=...
python examples/compare_decision_models.py --jev --ambertrace-platform N \
    --policy-contains "loan approval" --corpus data/ood_probe_v1.jsonl
```

## Pending arms (tracked on #123)

- **Nimble** — adapter built and offline-tested; a live run needs the local
  `Bespoke-Nimble-9B` model downloaded (MLX on Apple Silicon / CUDA on NVIDIA).
- **Full multi-domain matrix** — `ood_probe_v1` spans 95 domains / ~14 policy groups;
  a faithful AmberTrace arm needs one verified platform **per policy group** (a mixed
  dataset is incoherent — see the build note in the design doc). Next step: build the
  ~14 platforms and aggregate.
- **Nimble contrastive holdout** — generate via `nimble/datasets/create_eval_dataset.py`
  and add as the neutral (none-of-the-three-labelled-it) accuracy arm.
- **decision_eval_v1** — usable for Jev/Nimble, but AmberTrace authored its labels, so
  AmberTrace is scored there only as the fail-close/OOD runtime, never for label recall.
