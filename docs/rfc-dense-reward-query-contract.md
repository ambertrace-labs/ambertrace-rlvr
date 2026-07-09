# RFC — Query `explanation` contract for RLVR dense rewards

**From:** the `ambertrace-rlvr` team (an AmberTrace **customer**, building on the public `ambertraceai` SDK)
**To:** the AmberTrace platform / SDK team
**Status:** v0.5 — dense-reward contract **fully met end-to-end** (SDK `1.0.3`+ typing, server deploy 2026-07-09); one residual fact-shape item + two throughput asks remain
**Basis:** public SDK `ambertraceai==1.0.5` + live probe of a verified platform (API build `b02a5b74ff76`, 2026-07-09).

> **Verified on SDK 1.0.5 + the 2026-07-09 server deploy:** the server now emits **`schema_version`**, **`RuleFiring.required`**, and **`decision.deciding_rules`** — three of the four §1 drift items are resolved. The dense reward is now buildable *and dependable*. The single residual is the structured `rejected_facts` shape (§1). 1.0.4/1.0.5's `query` additions (`predictions` fan-in with per-ref `fatal`/`non_fatal`, open-textured `scored_determinations`) don't touch this RFC; `responses.py` is unchanged since 1.0.3.

---

## 0. Status — dense reward met 🎉

The platform team shipped the core of this RFC in **`ambertraceai==1.0.3`** (typing) and the **2026-07-09 server deploy** (emission). `QueryResult.explanation` is typed `QueryExplanation` with pinned sub-shapes (`SymbolicTrace`/`RuleFiring`, `CertifiedFact`/`CertifiedFactSummary`, `RejectedFact`, `Confidence`, `Proof`, `schema_version`), and the live API now emits the fields the shaper needs. Thank you.

| # | Ask | Status |
|---|-----|--------|
| 0 | Document + type the `explanation` fields (SDK `TypedDict`s) | ✅ **Resolved (SDK 1.0.3)** |
| A | Pin + emit `symbolic_trace.rules[]` (`rule_name`, `fired`, `rule_type`, `required`) | ✅ **Resolved** — typed 1.0.3, `required` **now emitted live** (2026-07-09) and meaningful (flags the deny-family rule) |
| C | `schema_version` on the explanation | ✅ **Resolved** — **now emitted live** as `schema_version: 1` |
| — | `decision.deciding_rules` populated on a decision | ✅ **Resolved** — **populates on a deny** (`[{rule, reason}]`); correctly empty on a permit (nothing blocks) |
| B | Emit structured `rejected_facts` (`{field, value, reasons}`) | 🟡 **Partial — P1**, see §1 |
| D | `query_batch` endpoint | ⬜ **Open — P2**, see §3 |
| E | Compact reward projection on `query` | ⬜ **Open — P3**, see §3 |

Nice touches, confirmed live: `RuleFiring.fired` is the **kernel-certified** firing set (reconciled against `proof.firings`), not the engine self-report — hardens our anti-reward-hacking story; `SymbolicTrace` carries `rules_evaluated`/`rules_fired` counts; and `required` cleanly marks the single blocking (deny-family) rule so `graded` can weight it above informational criteria.

## 1. Residual gap — structured `rejected_facts` still returns bare strings (P1)

The one item the 2026-07-09 deploy didn't close. Observed live on verified platform 9:

| Typed in `QueryExplanation` | Live API behaviour (2026-07-09) |
|---|---|
| `explanation.rejected_facts: list[RejectedFact]` (`{field, value, reasons}`) | On an out-of-domain query the facts come back as **bare field-name strings** — `AmbertraceError.rejected_facts == ["loan_type"]` — not the typed `RejectedFact` shape. (The rejection *reasons* are present, but only baked into the error's message string.) |

**Impact:** low. `rejected_penalty` can already run off `certified_fact_summary` (`{accepted, emitted, rejected, witness_invalid}` — a clean per-query reject count, confirmed live) and `proof_checked`. Structured per-fact `{field, value, reasons}` would sharpen the anti-hacking **fact-provenance** check (attribute a penalty to the specific hallucinated fact + reason) rather than just counting rejects.

**Ask:** emit `rejected_facts` as `list[RejectedFact]` (`{field, value, reasons}`) — on the response `explanation.rejected_facts` where a decision still certifies, and/or on `AmbertraceError.rejected_facts` for the hard-fail path (currently `list[str]`). This is the last mismatch between the typed contract and the wire.

> **Note on `explanation.rejected_facts` vs. the error path:** on this platform, client-supplied facts are gated hard (in-domain → certified at confidence 1.0; out-of-domain → 503 with string field names in the error). A sub-τ *soft* reject that leaves a decision certifiable (where `explanation.rejected_facts` would populate on a 200) wasn't reproducible here, so we could only observe the error path. Please confirm the structured shape on whichever path a partial-reject-but-still-certified query takes.

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

We're **cleared to build the full, dependable reward path** — not just a prototype. Confirmed live (2026-07-09): `graded` reads `explanation.symbolic_trace.rules[]` (`fired` + `required` + `rule_type`); `certified`/`correctness` read `proof_checked` + `decision` (+ `decision.deciding_rules` on a deny); `rejected_penalty` reads `certified_fact_summary`; `confidence` shapes. Our `reports.py` adapter gates on `explanation.schema_version` (now `1`) and treats only **structured `rejected_facts`** as still-optional — falling back to the `certified_fact_summary` reject count until the typed `{field, value, reasons}` shape is emitted. We target `ambertraceai>=1.0.5`.

## 5. Open questions

1. Timeline for emitting structured `rejected_facts` (`{field, value, reasons}`) — the one residual (§1)?
2. `required` currently marks the deny-family / blocking rule (confirmed live). Is that the intended semantics, or will it also cover `require`-leaf obligations more broadly? Affects how we weight it in `graded`.
3. Appetite / timeline for `query_batch` (§3, D)?
4. Is `explain=True` cost a concern at RL throughput — i.e. worth the compact projection (§3, E) as a default RL mode?

---

*Field shapes and live behaviour above are transcribed from responses on a verified platform (Loan Approval, API build `b02a5b74ff76`, 2026-07-09). Availability may vary by platform type; please confirm against the intended target platforms.*
