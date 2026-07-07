# RFC — A stable per-criterion query contract for RLVR dense rewards

**From:** the `ambertrace-rlvr` library (an AmberTrace **customer**, using the public `ambertraceai` SDK only)
**To:** the AmberTrace platform / SDK team
**Status:** draft · requirements · v0.1
**Basis:** public SDK `ambertraceai==1.0.2` (introspected as a consumer — no internal source assumed)

---

## 1. Why we're asking

`ambertrace-rlvr` turns an AmberTrace verified platform into the reward source for RL post-training (RLVR). The reward is not a learned model — it is the machine-checked certificate returned by the platform. Our reward shaper (`docs/AmberTrace-RLVR — Library Specification.md` §8) composes several bounded components from a single `platforms.query(...)` call:

| Component | Signal it needs from the query response |
|-----------|-----------------------------------------|
| `certified` | `proof_checked` |
| `correctness` | the certified `answer` / `decision` |
| **`graded`** | **per-criterion partial credit — which required rules/criteria fired vs. which were needed** |
| **`rejected_penalty`** | **which asserted facts were rejected, and why (fact provenance)** |

`graded` and `rejected_penalty` are what make the reward **dense and hack-resistant** rather than a sparse binary pass/fail. A binary `proof_checked` alone is too sparse to train on and trivially gamed. These two components are the ones the current public contract does not let a customer compute reliably.

## 2. What the public SDK gives us today (v1.0.2)

`platforms.query(platform_id, query=..., facts=..., relations=..., explain=True)` returns (`QueryResult`):

- **Pinned:** `answer` (required), `platform_id`, `query`, `decision`, `proof_checked` (`bool | None`), `proof_summary`, `vocabulary_declared`.
- **`explanation`** — documented as an **OPEN** `dict[str, Any]` whose "keys vary by platform". Observed keys: `confidence`, `symbolic_trace`, `neural_trace`, `relation_provenance` (this last one *is* pinned: `{relation, matched, min_count, count}`). Per the `build_request` docs, rejected facts are "surfaced in `explanation.rejected_facts`".

By contrast, the **agent-policy** path `agent_policy.authorize_action(...)` (`AuthorizeActionResult`) returns a much richer, **pinned-key** structure: `decision`, `permitted`, `proof_checked`, `outcome`, `proof_summary`, `denied_reason`, `deciding_rule`, `certified_facts`, `rejected_facts`, `missing_inputs`, `stalled_stage`, `query_diagnostics = {missing_atoms, deciding_rule, rejected_facts, stalled_stage}`.

## 3. The gaps

### Gap A — no stable per-criterion firing breakdown on `platforms.query` (blocker for `graded`)

The dense `graded` reward needs, for each rule/criterion material to the decision: its name, whether it **fired**, whether it was **required** (and its polarity), and ideally the facts it consumed. Today that information — if present at all — lives inside `explanation["symbolic_trace"]`, which the SDK explicitly types as open and platform-dependent. We cannot build a reward function on a shape the contract declines to pin: it may change between platforms or versions and silently corrupt a training run (the reward function must be deterministic and stable across hundreds of thousands of calls per run).

**This is the primary blocker.** Without it, `graded` degrades to `proof_checked` and the "dense" story collapses.

### Gap B — `rejected_facts` / `certified_facts` item schema is not pinned (weakens `rejected_penalty` + fact-provenance)

`authorize_action` exposes `rejected_facts` / `certified_facts` as pinned keys, but their **items** are `list[Any]` — the per-fact shape is undocumented. On `platforms.query`, rejected facts are only in the open `explanation` blob. Our `rejected_penalty` component and our anti-reward-hacking **fact-provenance check** (penalise facts the model asserted that the platform rejected) need a documented per-fact record.

### Gap C — `proof_checked` is `bool | None` on `query`, and the trace is `explain`-gated

`proof_checked` can be absent (`None`). For a fail-closed reward we treat absent/`None` as "not certified" (floor), but we'd like the contract to state when it is guaranteed present. Separately, the derivation detail is only returned with `explain=True`; we need to know that flag is safe to leave on at RL throughput (see Gap D).

### Gap D — no batch query endpoint (throughput)

RLVR issues `group_size × batch` verifications **per training step** (thousands per step). `platforms.query` is one focal row per HTTP call, so a customer must fan out N calls with client-side concurrency/backpressure. A batch endpoint would cut per-call overhead, tail latency, and rate-limit pressure dramatically. Not a correctness blocker, but the single biggest scaling lever for this use-case.

## 4. What we'd like (proposed public contract)

We're specifying the **shape we'd consume**, not how to produce it. Names are suggestions; the platform team owns the vocabulary.

### 4.1 A pinned `criteria` breakdown on the query explanation (Gap A)

Add a documented, schema-pinned array to the query response (either at top level or as a stable key inside `explanation`), returned when `explain=True`:

```jsonc
"criteria": [
  {
    "name": "pvs1",              // stable rule/criterion identifier
    "fired": true,               // did this criterion evaluate true?
    "required": true,            // was it necessary for the decision (vs. optional/informational)?
    "polarity": "supporting",    // "supporting" | "opposing" | "neutral"  (optional)
    "consumed_facts": ["variant_type", "gene"],   // fact keys it read (optional)
    "reason": "null variant in a gene where LOF is a known mechanism"  // human-readable (optional)
  }
]
```

Minimum we need for `graded`: `name`, `fired`, `required`. Everything else is a bonus that improves partial-credit shaping and auditability.

### 4.2 A pinned per-fact record for `certified_facts` / `rejected_facts` (Gap B)

Pin the **item** schema (on both `query.explanation` and `authorize_action`):

```jsonc
"rejected_facts": [
  {
    "field": "pm2",
    "value": 0.4,
    "threshold": 0.9,           // the confidence threshold it failed (if applicable)
    "confidence": 0.62,         // per-cell fused confidence (if applicable)
    "reason": "below verified_min_confidence"
  }
],
"certified_facts": [ { "field": "pvs1", "value": true, "confidence": 0.98 } ]
```

### 4.3 Stability + versioning guarantees (Gap A/C)

- Document these fields as **stable, platform-independent** contract (not "keys vary by platform").
- Version the trace schema (e.g. `explanation.schema_version`) so a consumer can pin/validate.
- State when `proof_checked` is guaranteed present on `query`.

### 4.4 A batch query endpoint (Gap D)

```python
api.platforms.query_batch(platform_id, items=[
    {"query": "...", "facts": {...}, "relations": {...}},
    ...
], explain=True) -> list[QueryResult]   # order-preserving, per-item fail-closed
```

Per-item failures should fail closed **independently** (one bad row must not fail the batch), each carrying its own `proof_checked=False` / diagnostics.

### 4.5 (Optional) a compact "reward projection" (throughput + stability)

Analogous to `predictions.predict(compact_certification=...)`: an opt-in flag on `query` that returns only the reward-relevant projection —

```jsonc
{ "proof_checked": true, "confidence": 0.94, "decision": "pathogenic",
  "criteria": [ {"name": "pvs1", "fired": true, "required": true}, ... ],
  "rejected_facts": [ {"field": "pm2", "reason": "below_threshold"} ] }
```

Small, stable, cache-friendly — ideal for the hot RL path.

## 5. Priority

| Item | Priority | Rationale |
|------|----------|-----------|
| 4.1 `criteria` breakdown | **P0** — blocker | Without it there is no dense `graded` reward |
| 4.2 pinned fact records | **P1** | Enables `rejected_penalty` + fact-provenance anti-hacking |
| 4.3 stability/versioning | **P1** | RL rewards must be reproducible across a run |
| 4.4 `query_batch` | **P2** | Largest throughput win; workaroundable with client fan-out |
| 4.5 compact projection | **P3** | Nice-to-have; optimisation |

## 6. Interim workaround on our side (no platform change)

Until 4.1 lands we will: (a) source `certified`/`correctness` from the pinned `proof_checked` + `answer`/`decision`; (b) best-effort parse `explanation.symbolic_trace` / `explanation.rejected_facts` behind a **feature-flagged, defensively-typed adapter** that degrades `graded`/`rejected_penalty` to zero-weight when the expected keys are absent — never crashing the training loop. This keeps us unblocked on the sparse reward while the dense contract is built, but it is explicitly *not* something we can depend on for a real training run, which is why we're filing this RFC.

## 7. Open questions for the platform team

1. Is a per-criterion firing breakdown already computed internally and merely not surfaced on the public `query` contract — or is it new work?
2. Can `criteria` + pinned fact records be exposed on **`platforms.query`** (our path), not only `agent_policy.authorize_action`?
3. Is `explain=True` safe to leave on at RL throughput, or does it carry a cost that argues for the compact projection (4.5) as the default RL mode?
4. Appetite/timeline for `query_batch` (4.4)?
```
