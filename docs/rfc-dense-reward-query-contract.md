# RFC — Pin & version the query explanation contract for RLVR dense rewards

**From:** the `ambertrace-rlvr` library (an AmberTrace **customer**, using the public `ambertraceai` SDK only)
**To:** the AmberTrace platform / SDK team
**Status:** draft · requirements · v0.2 (revised after a live probe)
**Basis:** public SDK `ambertraceai==1.0.2` + a live probe against a verified platform (API `0.1.0`), 2026-07-07 — introspected purely as a consumer, no internal source assumed.

> **v0.2 correction (important):** an earlier draft assumed the per-criterion firing data was *missing* and might be new work. A live probe shows it is **already returned** by `platforms.query(explain=True)` — inside the `explanation` block. So this RFC is now a **contract-hardening** request (pin + document + version shapes you already emit), **not** a build-new-functionality request. This should make it considerably cheaper.

---

## 1. Why we're asking

`ambertrace-rlvr` turns an AmberTrace verified platform into the reward source for RL post-training (RLVR). The reward is the machine-checked certificate — not a learned model. Our reward shaper composes several bounded components from one `platforms.query(...)` call:

| Component | Signal it needs |
|-----------|-----------------|
| `certified` | `proof_checked` |
| `correctness` | the certified `answer` / `decision` |
| **`graded`** | **per-criterion partial credit — which rules fired vs. which existed** |
| **`rejected_penalty`** | **which asserted facts were rejected, and why (fact provenance)** |

`graded` and `rejected_penalty` are what make the reward **dense and hack-resistant** rather than a gameable binary. They depend on a per-rule and per-fact breakdown — and the whole point of RLVR is that this reward must be **deterministic, stable, and reproducible** across the hundreds of thousands of calls in a single training run.

## 2. What the live platform actually returns (probe, 2026-07-07)

`platforms.query(platform_id, query=..., facts=..., explain=True)` on a verified platform returns, at top level: `answer`, `decision`, `proof_checked`, `proof_summary`, `platform_id`, `query`, `vocabulary_declared`, and `explanation`.

The **`explanation`** block (observed keys: `answer`, `certified_fact_summary`, `certified_facts`, `combination`, `confidence`, `graph_trace`, `neural_trace`, `proof`, `proof_checked`, `proof_summary`, `symbolic_trace`) already carries everything the dense reward needs:

**`explanation.symbolic_trace`** — the per-criterion breakdown (the `graded` signal):
```jsonc
{ "description": "Rules evaluated against the query context",
  "rules": [
    { "rule_id": 155, "rule_name": "Calculate Credit Score Class",
      "rule_type": "derive", "action_type": "derive", "fired": true,
      "explanation": "Rule '...' fired: Classify credit score as high/low" },
    { "rule_id": 170, "rule_name": "Check Applicant Age Underage ...",
      "rule_type": "constraint", "action_type": null, "fired": false,
      "explanation": "Rule '...' did not match context" }
    // NOTE: both fired=true AND fired=false rules are listed — exactly the
    // "which criteria fired vs. which existed" data `graded` needs.
  ] }
```

**`explanation.certified_facts` + `certified_fact_summary`** — the `rejected_penalty` / fact-provenance signal:
```jsonc
"certified_fact_summary": { "accepted": 12, "emitted": 12, "rejected": 0, "witness_invalid": 0 },
"certified_facts": [
  { "field": "credit_score", "value": 818, "confidence": 1.0,
    "schema_ok": true, "witness_invalid": false, "reasons": [],
    "source": "client",
    "certificate": { "schema_witness": { "declared": true, "dtype": "float",
                                          "field": "credit_score", "in_domain": true },
                     "source": "client", "extraction": {"method": "client_supplied"} } }
]
```

**`explanation.confidence`** — fused confidence with a documented methodology:
```jsonc
{ "overall": 0.88, "neural_confidence": 0.69, "symbolic_confidence": 1.0,
  "neural_weight": 0.4, "symbolic_weight": 0.6, "symbolic_normaliser": 3 }
```

**`explanation.proof`** — the machine-checked derivation: `{ derived: [{by, field, stratum, value}], facts: {...}, firings: [{action, rule, stratum}] }`.

`proof_summary` corroborates: *"8 rule(s) fired, 8 fact(s) derived from 12 input fact(s)."*

## 3. The actual gap — it's a *contract*, not a *capability*

The data is all there at runtime. The problem is that the public SDK types `QueryResult.explanation` as an **open `dict[str, Any]`** whose "keys vary by platform" — i.e. the contract explicitly declines to pin `symbolic_trace`, `certified_facts`, `certified_fact_summary`, or `confidence`. A reward function built on `explanation["symbolic_trace"]["rules"][i]["fired"]` and `explanation["certified_facts"][j]["confidence"]` is therefore depending on an **unpinned, unversioned shape the platform is free to change**. For a one-off audit that's fine; for a reproducible RL training run issuing these calls at scale, an unannounced shape change silently corrupts the reward and the run.

So the ask is narrow and cheap:

| # | Gap | Ask |
|---|-----|-----|
| **A** | `symbolic_trace.rules[]` is unpinned | Promote to a **documented, versioned, platform-independent** part of the contract. Minimum per-rule fields we depend on: `rule_name` (stable id), `fired` (bool), `rule_type`. Bonus: `required`/polarity so we can weight required-vs-informational criteria. |
| **B** | `certified_facts[]` / `certified_fact_summary` item shapes are unpinned (SDK types them `list[Any]`) | Pin the per-fact shape: `{field, value, confidence, schema_ok, in_domain, witness_invalid, reasons}`. This is what `rejected_penalty` + our anti-hacking fact-provenance check read. |
| **C** | No schema version; `proof_checked` is `bool \| None` on `query` | Add `explanation.schema_version` (or top-level) so consumers can pin/validate; document when `proof_checked` is guaranteed present. |
| **D** | No batch endpoint | See §4. |
| **E** | Full `explanation` is heavy at RL throughput | Optional compact reward projection — see §4. |

## 4. Two throughput asks (net-new, lower priority)

### 4.1 A batch query endpoint (D)
RLVR issues `group_size × batch` verifications **per training step** (thousands per step). `platforms.query` is one focal row per HTTP call, so we must fan out N calls with client-side concurrency/backpressure. A batch endpoint cuts per-call overhead, tail latency, and rate-limit pressure:
```python
api.platforms.query_batch(platform_id, items=[
    {"query": "...", "facts": {...}, "relations": {...}}, ...
], explain=True) -> list[QueryResult]   # order-preserving; per-item fail-closed
```
Per-item failures must fail closed **independently** (one bad row must not fail the batch), each carrying its own `proof_checked=False` + diagnostics.

### 4.2 (Optional) a compact reward projection (E)
The full `explanation` embeds a per-fact `certificate` for every input fact — large on a wide row (you already flip `predictions.predict` to compact certification by default for exactly this reason). An opt-in flag on `query` returning only the reward-relevant projection would be ideal for the hot RL path:
```jsonc
{ "proof_checked": true, "confidence": 0.88, "decision": "permit",
  "criteria": [ {"rule_name": "Check Credit Score Exceeds Threshold", "fired": true} ],
  "fact_summary": { "accepted": 12, "rejected": 0 },
  "rejected_facts": [ {"field": "...", "reason": "..."} ],
  "schema_version": "1" }
```
Small, stable, cache-friendly.

## 5. Priority

| Item | Priority | Rationale |
|------|----------|-----------|
| C — `schema_version` + stability guarantee on `symbolic_trace` / `certified_facts` / `confidence` | **P0** | The crux. Turns a best-effort parse into a dependable contract; unblocks a *real* (not just prototype) training run |
| A — pin `symbolic_trace.rules[]` (`rule_name`, `fired`, `rule_type`; ideally `required`) | **P0** | The dense `graded` signal |
| B — pin `certified_facts[]` / `certified_fact_summary` item shape | **P1** | `rejected_penalty` + fact-provenance anti-hacking |
| D — `query_batch` | **P2** | Largest throughput win; workaroundable with client fan-out |
| E — compact projection | **P3** | Optimisation |

## 6. Where this leaves us today

Because the data is already emitted, we are **effectively unblocked for a first training run** on a best-effort basis: our verifier will read `explanation.symbolic_trace.rules[].fired` (for `graded`) and `explanation.certified_facts[]` / `certified_fact_summary` (for `rejected_penalty`) behind a **defensively-typed, feature-flagged adapter** that degrades the affected component to zero-weight if the expected keys are absent — never crashing the training loop. What we cannot do until **C/A/B** land is *depend* on that shape for a reproducible, publishable run — an unannounced change to the open `explanation` would silently corrupt the reward. Hence this RFC targets contract-hardening over new capability.

## 7. Open questions for the platform team

1. Can `symbolic_trace.rules[]` and `certified_facts[]` be promoted to a **pinned, versioned** part of the public `QueryResult` contract (not "keys vary by platform")? That is the single highest-value change here.
2. Is there a notion of a rule being **required vs. informational** for a decision that could be surfaced per-rule (`required: bool` / polarity)? It would sharpen partial-credit weighting.
3. Is `explain=True` safe to leave on at RL throughput, or does the per-fact certificate cost argue for the compact projection (§4.2) as the default RL mode?
4. Appetite/timeline for `query_batch` (§4.1)?

---

*Probe evidence on file with the author (verified Loan Approval platform, API `0.1.0`, 2026-07-07). Shapes above are transcribed from that live response; exact field availability may differ across platform types and should be confirmed by the platform team.*
