# RFC — Document, pin & version the query `explanation` contract

**From:** the `ambertrace-rlvr` team (an AmberTrace **customer**, building on the public `ambertraceai` SDK)
**To:** the AmberTrace platform / SDK team
**Status:** request for implementation · v0.3
**Basis:** public SDK `ambertraceai==1.0.2` + a live probe of a verified platform (API `0.1.0`), 2026-07-07. Everything below was determined as a consumer — from the published package and live API responses only.

---

## TL;DR for the platform team

We're building an RLVR training library that uses a verified platform's certificate as the RL reward. To make that reward **dense** (partial credit per criterion) rather than a gameable pass/fail, we need the per-rule and per-fact breakdown from a query.

Good news: **`platforms.query(explain=True)` already returns all of it** — in the `explanation` block. Two problems, both cheap to fix:

1. **It's undocumented.** Neither the SDK type stubs nor the API reference describe the `explanation` sub-shapes (`symbolic_trace`, `certified_facts`, `certified_fact_summary`, `confidence`, `proof`). We only found them by probing a live platform. A customer should not have to reverse-engineer a response to use it.
2. **It's an unpinned, unversioned contract.** The SDK types `explanation` as an open `dict[str, Any]` ("keys vary by platform"), so we can't safely depend on it for a reproducible training run — an unannounced shape change would silently corrupt the reward.

So this is a **documentation + contract-hardening** request, not new functionality. Asks, in priority order: (P0) document + version these fields; (P0) pin `symbolic_trace.rules[]`; (P1) pin `certified_facts[]`; (P2) a batch query endpoint; (P3) an optional compact projection.

---

## 1. Context — why we need this

`ambertrace-rlvr` turns a verified platform into the reward source for RL post-training. The model proposes a decision; the platform independently certifies it; the certificate becomes the reward. To train well, the reward shaper composes bounded components from a single `platforms.query(...)` call:

| Reward component | Signal it needs from the response |
|------------------|-----------------------------------|
| `certified` | `proof_checked` |
| `correctness` | the certified `answer` / `decision` |
| **`graded`** | **per-criterion partial credit: which rules fired vs. which existed** |
| **`rejected_penalty`** | **which asserted facts were rejected, and why (fact provenance)** |

`graded` and `rejected_penalty` are what make the reward dense and hard to game. Both depend on the per-rule / per-fact breakdown. And because RLVR issues these calls hundreds of thousands of times per run, the shape must be **documented, stable, and versioned** — a reward function cannot be built on a response we had to guess at.

## 2. What the live platform returns today (probe, 2026-07-07)

`platforms.query(platform_id, query=..., facts=..., explain=True)` on a verified platform returns, at top level: `answer`, `decision`, `proof_checked`, `proof_summary`, `platform_id`, `query`, `vocabulary_declared`, `explanation`.

The **`explanation`** block already carries the full dense-reward substrate. Observed keys: `answer`, `certified_fact_summary`, `certified_facts`, `combination`, `confidence`, `graph_trace`, `neural_trace`, `proof`, `proof_checked`, `proof_summary`, `symbolic_trace`.

**`explanation.symbolic_trace`** — the per-criterion breakdown (drives `graded`):
```jsonc
{
  "description": "Rules evaluated against the query context",
  "rules": [
    { "rule_id": 155, "rule_name": "Calculate Credit Score Class",
      "rule_type": "derive", "action_type": "derive", "fired": true,
      "explanation": "Rule '...' fired: Classify credit score as high/low" },
    { "rule_id": 170, "rule_name": "Check Applicant Age Underage ...",
      "rule_type": "constraint", "action_type": null, "fired": false,
      "explanation": "Rule '...' did not match context" }
  ]
}
// Both fired=true and fired=false rules are listed — exactly the
// "which criteria fired vs. which existed" data `graded` needs.
```

**`explanation.certified_facts` + `certified_fact_summary`** — fact provenance (drives `rejected_penalty`):
```jsonc
"certified_fact_summary": { "accepted": 12, "emitted": 12, "rejected": 0, "witness_invalid": 0 },
"certified_facts": [
  { "field": "credit_score", "value": 818, "confidence": 1.0,
    "schema_ok": true, "witness_invalid": false, "reasons": [], "source": "client",
    "certificate": { "schema_witness": { "declared": true, "dtype": "float",
                                          "field": "credit_score", "in_domain": true },
                     "source": "client", "extraction": { "method": "client_supplied" } } }
]
```

**`explanation.confidence`** — fused confidence with a stated methodology:
```jsonc
{ "overall": 0.88, "neural_confidence": 0.69, "symbolic_confidence": 1.0,
  "neural_weight": 0.4, "symbolic_weight": 0.6, "symbolic_normaliser": 3 }
```

**`explanation.proof`** — the machine-checked derivation:
```jsonc
{ "derived": [ { "by": "Check Credit Score Exceeds Threshold",
                 "field": "credit_score_flag", "stratum": 1, "value": 740 } ],
  "facts": { "...": "..." },
  "firings": [ { "action": "derive", "rule": "...", "stratum": 1 } ] }
```

`proof_summary` corroborates: *"Decision independently certified against the trusted kernel: 8 rule(s) fired, 8 fact(s) derived from 12 input fact(s)."*

## 3. The gaps

### Gap 0 — the response is undocumented (documentation)

Neither surface a customer has tells them any of §2 exists:

- **SDK:** `QueryResult.explanation` is typed `dict[str, Any]` with the docstring note that keys "vary by platform." No `TypedDict` for `symbolic_trace`, `certified_facts`, `certified_fact_summary`, `confidence`, or `proof`.
- **API reference:** the `explanation` object is not described field-by-field in the OpenAPI/redoc for the query endpoint.

We discovered the dense-reward substrate only by calling a live platform and dumping the JSON. That's a poor DX and a blocker to adoption: a customer can't build on fields they can't find. **Documenting these fields is part of the fix, not a follow-up.**

### Gap A — `symbolic_trace.rules[]` is not a pinned contract

The per-rule firing list (the `graded` signal) lives inside the open `explanation`. We depend, per rule, on: `rule_name` (a stable identifier), `fired` (bool), `rule_type`. We'd benefit from a `required`/polarity flag (see §4). Today none of this is a guaranteed shape.

### Gap B — `certified_facts[]` / `certified_fact_summary` item shapes are not pinned

The per-fact records (the `rejected_penalty` + fact-provenance signal) are similarly unpinned; the SDK would type them `list[Any]`. We depend on `{field, value, confidence, schema_ok, in_domain, witness_invalid, reasons}` per fact and the `{accepted, emitted, rejected, witness_invalid}` summary.

### Gap C — no schema version; `proof_checked` nullable on `query`

There's no version marker on the explanation payload, so a consumer can't pin or validate the shape. And `proof_checked` is nullable on the query return — we need to know when it is guaranteed present (we treat absent as "not certified"/floor, but the contract should state it).

### Gap D — no batch query endpoint (throughput)

RLVR issues `group_size × batch` verifications **per training step** (thousands per step). `platforms.query` is one focal row per HTTP call, forcing client-side fan-out with concurrency and backpressure.

### Gap E — full `explanation` is heavy at RL throughput

The block embeds a per-fact `certificate` for every input fact — large on a wide row. (You already default `predictions.predict` to compact certification for exactly this reason.)

## 4. Requested contract

Names are suggestions; the platform team owns the vocabulary. The shapes below match what §2 already returns — we're asking you to **document, pin, and version** them.

**Documentation (Gap 0).** Describe the `explanation` object field-by-field in the OpenAPI spec + redoc, and add `TypedDict`s to the SDK (`SymbolicTrace`, `RuleFiring`, `CertifiedFact`, `CertifiedFactSummary`, `Confidence`, `Proof`) so `QueryResult.explanation` is typed rather than `dict[str, Any]`. IDE autocomplete + a rendered API reference are the deliverable.

**`symbolic_trace.rules[]` (Gap A)** — pin per rule:
```jsonc
{ "rule_name": "string (stable id)", "fired": true, "rule_type": "derive|constraint",
  "required": true,           // NEW (optional): was this rule necessary for the decision?
  "polarity": "supporting|opposing|neutral",   // NEW (optional)
  "explanation": "string" }
```
Minimum we depend on: `rule_name`, `fired`, `rule_type`. `required`/`polarity` would let us weight required vs. informational criteria in partial credit.

**`certified_facts[]` / `certified_fact_summary` (Gap B)** — pin the item shapes shown in §2.

**Versioning (Gap C)** — add `explanation.schema_version` (or a top-level equivalent) and document when `proof_checked` is guaranteed present on `query`.

**`query_batch` (Gap D):**
```python
api.platforms.query_batch(platform_id, items=[
    {"query": "...", "facts": {...}, "relations": {...}}, ...
], explain=True) -> list[QueryResult]   # order-preserving; per-item fail-closed
```
Per-item failures must fail closed **independently** — one bad row must not fail the batch; each carries its own `proof_checked=False` + diagnostics.

**Compact projection (Gap E)** — an opt-in flag on `query` returning only the reward-relevant fields:
```jsonc
{ "proof_checked": true, "confidence": 0.88, "decision": "permit",
  "criteria": [ { "rule_name": "Check Credit Score Exceeds Threshold", "fired": true } ],
  "fact_summary": { "accepted": 12, "rejected": 0 },
  "rejected_facts": [ { "field": "...", "reason": "..." } ],
  "schema_version": "1" }
```

## 5. Priority

| # | Ask | Priority | Rationale |
|---|-----|----------|-----------|
| 0 | Document the `explanation` fields (OpenAPI/redoc + SDK `TypedDict`s) | **P0** | A customer can't build on undocumented fields; unblocks adoption |
| C | `schema_version` + stability guarantee on `symbolic_trace` / `certified_facts` / `confidence` | **P0** | Turns a best-effort parse into a dependable contract for a reproducible run |
| A | Pin `symbolic_trace.rules[]` (`rule_name`, `fired`, `rule_type`; ideally `required`) | **P0** | The dense `graded` signal |
| B | Pin `certified_facts[]` / `certified_fact_summary` item shape | **P1** | `rejected_penalty` + fact-provenance anti-hacking |
| D | `query_batch` | **P2** | Largest throughput win; workaroundable with client fan-out |
| E | Compact projection | **P3** | Optimisation |

## 6. Where this leaves us today

Because the data is already emitted, we can build a **first prototype** now: our verifier reads `explanation.symbolic_trace.rules[].fired` (for `graded`) and `explanation.certified_facts[]` / `certified_fact_summary` (for `rejected_penalty`) behind a defensively-typed, feature-flagged adapter that degrades the affected component to zero-weight if the expected keys are absent — never crashing the training loop. What we **cannot** do until Gap 0 + C/A/B land is depend on that shape for a reproducible, publishable training run. Hence this request targets documentation and contract-hardening.

## 7. Open questions

1. Can `symbolic_trace.rules[]` and `certified_facts[]` be promoted to a documented, versioned part of the public `QueryResult` contract (rather than "keys vary by platform")? This is the single highest-value change.
2. Is there an internal notion of a rule being **required vs. informational** for a decision that could be surfaced per-rule (`required` / polarity)? It would sharpen partial-credit weighting.
3. Is `explain=True` safe to leave on at RL throughput, or does the per-fact certificate cost argue for the compact projection (§4, Gap E) as the default RL mode?
4. Appetite / timeline for `query_batch`?

---

*Field shapes above are transcribed from a live response on a verified platform (Loan Approval, API `0.1.0`, 2026-07-07). Availability may vary by platform type; please confirm against the intended target platforms.*
