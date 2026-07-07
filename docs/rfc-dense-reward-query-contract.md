# RFC — Query `explanation` contract for RLVR dense rewards

**From:** the `ambertrace-rlvr` team (an AmberTrace **customer**, building on the public `ambertraceai` SDK)
**To:** the AmberTrace platform / SDK team
**Status:** v0.4 — dense-reward contract resolved in SDK `1.0.3`; tracking the remaining server-side drift + two throughput asks
**Basis:** public SDK `ambertraceai==1.0.4` + live probe of a verified platform (API `0.1.0`), 2026-07-07.

> **Re-verified on SDK 1.0.4:** the `explanation` contract typing is unchanged from 1.0.3 (`responses.py` identical), and the deployed API is the same build (`git_sha 814a92c4e59a`) — so the §1 server-side drift below and the open throughput asks (§3) are **unchanged**. 1.0.4's additions (typed `eval_config`/templates, verified-build kwargs docs, a `query(predictions=…)` Prediction→Decision fan-in) don't touch this RFC.

---

## 0. Status — updated for SDK 1.0.3 🎉

The platform team shipped the core of this RFC in **`ambertraceai==1.0.3`**. `QueryResult.explanation` is now typed `QueryExplanation` (was open `dict[str, Any]`), with pinned, documented sub-shapes: `SymbolicTrace` / `RuleFiring`, `CertifiedFact` / `CertifiedFactSummary`, a new `RejectedFact`, `Confidence`, `Proof`, and a `schema_version`. Thank you — this unblocks the dense reward.

| # | Ask | Status |
|---|-----|--------|
| 0 | Document + type the `explanation` fields (SDK `TypedDict`s) | ✅ **Resolved in SDK 1.0.3** |
| A | Pin `symbolic_trace.rules[]` (`rule_name`, `fired`, `rule_type`, `required`) | ✅ **Typed in 1.0.3** — `required` ⚠️ not yet emitted by the live API (see §1) |
| B | Pin `certified_facts[]` / `certified_fact_summary` / `rejected_facts[]` | ✅ **Typed in 1.0.3** — `RejectedFact` shape ⚠️ not yet emitted (see §1) |
| C | `schema_version` on the explanation | ✅ **Typed in 1.0.3** — ⚠️ not yet emitted by the live API (see §1) |
| — | **SDK-ahead-of-API drift** | 🔴 **New — P0**, see §1 |
| D | `query_batch` endpoint | ⬜ **Open — P2**, see §3 |
| E | Compact reward projection on `query` | ⬜ **Open — P3**, see §3 |

Two nice touches we noticed and appreciate: `RuleFiring.fired` is documented as the **kernel-certified** firing set (reconciled against `proof.firings`), not the engine self-report — that directly hardens our anti-reward-hacking story; and `SymbolicTrace` now carries `rules_evaluated` / `rules_fired` counts (both confirmed live: 25 evaluated / 8 fired on our probe).

## 1. Remaining gap — the deployed API lags the SDK typing (P0)

The SDK typing is additive, so 1.0.3 now *promises* fields the deployed platform (API `0.1.0`, built 2026-07-06) does not yet emit. Observed live on verified platform 9:

| Typed in `QueryExplanation` (1.0.3) | Live API behaviour (2026-07-07) |
|---|---|
| `schema_version: int` | **Absent** — `explanation.get("schema_version")` is `None` |
| `RuleFiring.required: bool` | **Absent** — no `required` key on any rule |
| `explanation.rejected_facts: list[RejectedFact]` (`{field, value, reasons}`) | On a rejecting query the facts come back as **bare field-name strings** (`["loan_type", "loan_purpose"]`, via the `AmbertraceError`), not the structured `RejectedFact` shape |
| `decision.deciding_rules: [{rule, reason}]` | Present but **empty** (`[]`) on the observed permit |

Live and correct today: `symbolic_trace.rules[]` (`rule_id`, `rule_name`, `rule_type`, `action_type`, `fired`, `explanation`), `symbolic_trace.rules_evaluated` / `rules_fired`, `certified_facts[]`, `certified_fact_summary`, `confidence`, `proof`.

**Ask:** land these fields server-side so the typed contract is truthful end-to-end. Priority within that:

1. **`schema_version`** — without it a consumer can't tell which shape it's holding; it's the anchor for everything else. (P0)
2. **`RuleFiring.required`** — lets `graded` weight hard obligations vs. informational criteria; without it every rule is weighted equally. (P0)
3. **Structured `rejected_facts`** (`{field, value, reasons}`) on the response and/or the error body — our `rejected_penalty` and fact-provenance check want the value + reason, not just the field name. (P1)
4. **`decision.deciding_rules`** populated on a decision (esp. a deny) — useful for `correctness` attribution. (P1)

Until these land, our adapter treats each as optional and degrades the affected reward component to zero-weight when absent (§4). A `schema_version` bump when they do land is exactly what we'll gate on.

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

We're **unblocked to build the real reward path** on 1.0.3's typed contract. `graded` reads `explanation.symbolic_trace.rules[].fired`; `rejected_penalty` reads `certified_fact_summary` (+ `rejected_facts[]` once structured). Our report-normalisation adapter (`reports.py`) treats `schema_version`, `required`, structured `rejected_facts`, and `deciding_rules` as **optional**, degrading any component that needs a not-yet-emitted field to zero-weight rather than crashing — and will gate on `schema_version` once the server emits it. We target `ambertraceai>=1.0.3`.

## 5. Open questions

1. Timeline for emitting `schema_version` + `required` + structured `rejected_facts` server-side (§1)? These are the last mile of the contract you already typed.
2. When `required` lands: is it a `require`-leaf / deny-family notion, or something finer? Affects how we weight it in `graded`.
3. Appetite / timeline for `query_batch` (§3, D)?
4. Is `explain=True` cost a concern at RL throughput — i.e. worth the compact projection (§3, E) as a default RL mode?

---

*Field shapes and live behaviour above are transcribed from responses on a verified platform (Loan Approval, API `0.1.0`, 2026-07-07). Availability may vary by platform type; please confirm against the intended target platforms.*
