# Issue note — `rejected_facts` returns bare strings, not the typed `RejectedFact` shape

*Filed by the `ambertrace-rlvr` team (customer, public `ambertraceai` SDK). Copy/paste-ready for the issue tracker. Companion to `docs/rfc-dense-reward-query-contract.md` (§1).*

---

**Title:** `rejected_facts` returns `list[str]`, not the typed `RejectedFact` (`{field, value, reasons}`)

**Component:** platform query API / `ambertraceai` SDK
**Verified on:** `ambertraceai==1.0.5`, live API build `b02a5b74ff76` (2026-07-09), verified platform (Loan Approval)
**Severity:** low (workaround in place) · **Priority:** P1

## Summary

The SDK types rejected facts as a structured record —
`explanation.rejected_facts: list[RejectedFact]` where
`RejectedFact = {field, value, reasons}` (added in 1.0.3, `responses.py`) — but the
live API returns **bare field-name strings**. This is the last typed-contract-vs-wire
mismatch after the 2026-07-09 deploy (which correctly landed `schema_version`,
`RuleFiring.required`, and `decision.deciding_rules`).

## Repro

```python
import ambertraceai as a
api = a.AmbertraceAPI.from_env()          # AMBERTRACE_API_KEY / _BASE_URL

facts = {  # valid row except two out-of-domain enum values
    "applicant_age": 67, "annual_income": 135403, "credit_score": 818,
    "credit_history_months": 4, "debt_to_income_ratio": 0.26,
    "employment_status": "retired", "employment_years": 0.4,
    "loan_type": "mortgage",          # out of declared domain
    "loan_amount": 54887, "collateral_value": 0,
    "loan_purpose": "home_purchase",  # out of declared domain
    "existing_loans": 0,
}
try:
    api.platforms.query(9, query="Assess.", facts=facts, explain=True)
except a.AmbertraceError as e:
    print(type(e.rejected_facts), e.rejected_facts)
```

**Actual:**
```
<class 'list'> ['loan_type', 'loan_purpose']
```

**Expected (per the SDK type):**
```python
[
  {"field": "loan_type",    "value": "mortgage",      "reasons": ["value 'mortgage' is outside the declared domain of 'loan_type'"]},
  {"field": "loan_purpose", "value": "home_purchase", "reasons": ["value 'home_purchase' is outside the declared domain of 'loan_purpose'"]},
]
```

The rejection reasons *do* exist — they're currently only in the `AmbertraceError`
message string, not machine-readable per fact.

## Why it matters (customer context)

`ambertrace-rlvr` uses the query certificate as an RL reward. The `rejected_penalty`
component + an anti-reward-hacking **fact-provenance check** want to attribute a penalty
to the *specific* fact the model hallucinated and *why*. Bare field names lose the value
and the reason. Today we fall back to the `certified_fact_summary` reject **count**
(which works fine), so this is a sharpening, not a blocker.

## Proposed fix

Emit `rejected_facts` as `list[RejectedFact]` (`{field, value, reasons}`) to match the
1.0.3 typing, on:
- **`AmbertraceError.rejected_facts`** for the hard-fail (out-of-domain → 503) path, and
- **`explanation.rejected_facts`** on a 200 where a decision still certifies despite a reject.

## Open question for the platform team

On this platform, client-supplied facts are gated hard: in-domain → certified at
confidence 1.0; out-of-domain → 503. We couldn't reproduce a **soft** sub-τ reject that
leaves a decision certifiable (the 200 path where `explanation.rejected_facts` would
populate). Which path carries the structured list — the error body, the 200 explanation,
or both? Confirming that tells us where to read it.
