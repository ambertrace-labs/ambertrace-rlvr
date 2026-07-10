# RFC — Query `explanation` contract for RLVR dense rewards

**From:** the `ambertrace-rlvr` team (an AmberTrace **customer**, building on the public `ambertraceai` SDK)
**To:** the AmberTrace platform / SDK team
**Status:** v0.6 — dense-reward contract **fully resolved end-to-end** (server deploy 2026-07-10); only the two optional throughput asks remain
**Basis:** public SDK `ambertraceai==1.0.5` + live probe of a verified platform (API build `aa38e85da6f0`, 2026-07-10).

> **Fully verified on the 2026-07-10 server deploy:** the last residual — structured `rejected_facts` — now lands. `AmbertraceError.rejected_facts` returns the typed `RejectedFact` shape (`[{field, value, reasons}]`), joining the already-live `schema_version: 1`, `RuleFiring.required`, and `decision.deciding_rules`. **Every correctness item in this RFC (0/A/B/C + all four drift fields) is resolved.** Only the two lower-priority throughput asks remain: `query_batch` (§3 D, P2) and a compact `query` projection (§3 E, P3) — neither is a blocker. Thank you.

---

## 0. Status — dense reward met 🎉

The platform team shipped the core of this RFC in **`ambertraceai==1.0.3`** (typing) and the **2026-07-09 server deploy** (emission). `QueryResult.explanation` is typed `QueryExplanation` with pinned sub-shapes (`SymbolicTrace`/`RuleFiring`, `CertifiedFact`/`CertifiedFactSummary`, `RejectedFact`, `Confidence`, `Proof`, `schema_version`), and the live API now emits the fields the shaper needs. Thank you.

| # | Ask | Status |
|---|-----|--------|
| 0 | Document + type the `explanation` fields (SDK `TypedDict`s) | ✅ **Resolved (SDK 1.0.3)** |
| A | Pin + emit `symbolic_trace.rules[]` (`rule_name`, `fired`, `rule_type`, `required`) | ✅ **Resolved** — typed 1.0.3, `required` **now emitted live** (2026-07-09) and meaningful (flags the deny-family rule) |
| C | `schema_version` on the explanation | ✅ **Resolved** — **now emitted live** as `schema_version: 1` |
| — | `decision.deciding_rules` populated on a decision | ✅ **Resolved** — **populates on a deny** (`[{rule, reason}]`); correctly empty on a permit (nothing blocks) |
| B | Emit structured `rejected_facts` (`{field, value, reasons}`) | ✅ **Resolved** — `AmbertraceError.rejected_facts` now returns `[{field, value, reasons}]` (verified live 2026-07-10) |
| D | `query_batch` endpoint | ⬜ **Open — P2**, see §3 |
| E | Compact reward projection on `query` | ⬜ **Open — P3**, see §3 |

Nice touches, confirmed live: `RuleFiring.fired` is the **kernel-certified** firing set (reconciled against `proof.firings`), not the engine self-report — hardens our anti-reward-hacking story; `SymbolicTrace` carries `rules_evaluated`/`rules_fired` counts; and `required` cleanly marks the single blocking (deny-family) rule so `graded` can weight it above informational criteria.

## 1. Correctness items — all resolved ✅

Every field the dense reward depends on is now live and dependable (verified on API build `aa38e85da6f0`, 2026-07-10, verified platform 9):

| Field | Status |
|---|---|
| `explanation.schema_version` | ✅ emits `1` — gate on this |
| `symbolic_trace.rules[]` (`fired`, `rule_type`, `required`) | ✅ live; `required` flags the deny-family rule; `fired` is kernel-certified |
| `certified_facts[]` / `certified_fact_summary` | ✅ live |
| `decision.deciding_rules` | ✅ populates on a deny (`[{rule, reason}]`); empty on a permit (correct) |
| `explanation.confidence` / `proof` | ✅ live |
| `rejected_facts` | ✅ structured — `AmbertraceError.rejected_facts == [{field, value, reasons}]` as of 2026-07-10 |

Sample (out-of-domain query → hard fail):
```json
[ {"field": "loan_type",    "value": "mortgage",      "reasons": ["value 'mortgage' is outside the declared domain of 'loan_type'"]},
  {"field": "loan_purpose", "value": "home_purchase", "reasons": ["value 'home_purchase' is outside the declared domain of 'loan_purpose'"]} ]
```

*(Minor, non-blocking) note for a future look:* we could only observe the hard-fail (503 / `AmbertraceError`) path here, since client facts on this platform are gated hard (in-domain → certified; out-of-domain → 503). If a sub-τ *soft* reject can leave a decision certifiable, worth confirming `explanation.rejected_facts` carries the same `[{field, value, reasons}]` shape on that 200 path too. Not required for our reward path.

## 2. Why we needed this (recap)

`ambertrace-rlvr` uses a verified platform's certificate as the RL reward. To make the reward **dense** (partial credit per criterion) and **hack-resistant** rather than a gameable pass/fail, the shaper reads a per-rule and per-fact breakdown from each query:

| Reward component | Signal | Source field (1.0.3) |
|------------------|--------|----------------------|
| `certified` | proof certificate | `proof_checked` |
| `correctness` | certified verdict | `decision` / `answer` |
| `graded` | which criteria fired vs. existed | `explanation.symbolic_trace.rules[]` |
| `rejected_penalty` | which facts were rejected + why | `explanation.rejected_facts[]` / `certified_fact_summary` |

Because RLVR issues these calls hundreds of thousands of times per run, a documented + versioned shape (now delivered) is what makes the reward reproducible.

## 3. Remaining throughput asks (net-new, lower priority)

### D — batch query endpoint (P2)
RLVR issues `group_size × batch` verifications **per training step** (thousands per step). `platforms.query` is one focal row per HTTP call, so we fan out N calls with client-side concurrency/backpressure. A batch endpoint cuts per-call overhead and rate-limit pressure:
```python
api.platforms.query_batch(platform_id, items=[
    {"query": "...", "facts": {...}, "relations": {...}}, ...
], explain=True) -> list[QueryResult]   # order-preserving; per-item fail-closed
```
Per-item failures must fail closed **independently** — one bad row must not fail the batch; each carries its own `proof_checked=False` + diagnostics. (Workaroundable today with client-side fan-out, so not urgent.)

### E — compact reward projection on `query` (P3)
`explanation` embeds a per-fact `certificate` for every input fact — heavy on a wide row. (You already default `predictions.predict` to compact certification for this reason; `query` has no equivalent.) An opt-in flag returning only `{proof_checked, confidence.overall, decision, symbolic_trace.rules[{rule_name, fired, required}], certified_fact_summary, rejected_facts, schema_version}` would be ideal for the hot RL path.

## 4. Where this leaves us

The reward path is **fully unblocked on the live contract** — every component reads a real, dependable field: `graded` ← `symbolic_trace.rules[]` (`fired`+`required`+`rule_type`); `certified`/`correctness` ← `proof_checked` + `decision` (+ `decision.deciding_rules` on a deny); `rejected_penalty` + fact-provenance ← structured `rejected_facts` (`{field, value, reasons}`) and `certified_fact_summary`; `confidence` shapes. `reports.py` gates on `explanation.schema_version` (`1`) and no longer needs any fallback. We target `ambertraceai>=1.0.5`. Remaining RFC items (D/E) are throughput optimisations, not blockers.

## 5. Open questions

1. `required` currently marks the deny-family / blocking rule (confirmed live). Is that the intended semantics, or will it also cover `require`-leaf obligations more broadly? Affects how we weight it in `graded`.
2. Appetite / timeline for `query_batch` (§3, D)?
3. Is `explain=True` cost a concern at RL throughput — i.e. worth the compact projection (§3, E) as a default RL mode?
4. *(minor)* Does a sub-τ soft reject surface `explanation.rejected_facts` in the same `{field, value, reasons}` shape on the 200 path? (Not needed for our reward path; §1.)

---

*Field shapes and live behaviour above are transcribed from responses on a verified platform (Loan Approval, API build `aa38e85da6f0`, 2026-07-10). Availability may vary by platform type; please confirm against the intended target platforms.*
