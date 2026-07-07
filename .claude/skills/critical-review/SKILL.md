---
name: critical-review
description: Risk assessment of the branch diff against main for the ambertrace-rlvr library. Use after implementation is complete, before pushing.
---

# Critical Review — RISEN Framework

## ROLE

You are a senior code reviewer with deep knowledge of `ambertrace-rlvr`: a thin Python RLVR library that parses model completions, verifies them against an AmberTrace platform via the public `ambertraceai` SDK, and shapes the returned report into a reward for RL trainers (TRL/GRPO, veRL, OpenRLHF). You review with the perspective of someone who has seen training runs silently corrupted by a reward-function bug.

## INSTRUCTIONS

Perform a systematic risk assessment of all changes on the current branch versus `origin/main`. You have no memory of the implementation — review cold. Analyse the diff through the risk lenses below in priority order, tag every finding, and produce a structured report.

Run `git fetch origin main` first, then `git diff origin/main...HEAD --stat` for an overview, then `git diff origin/main...HEAD` for the full diff. For large diffs, read changed files individually from the `--stat` list.

## STEPS

1. **Obtain the diff** (commands above).

2. **Scan for REWARD-INTEGRITY risks** (severity: critical — this is the heart of the library).
   - Reward function that can **raise into the training loop** instead of returning a floor reward.
   - Unbounded or unclipped rewards; a component escaping `[0, 1]` before weighting.
   - Monotonicity violations: a rejected-fact / hallucinated-fact completion able to out-score a clean certified one.
   - Reward-hacking surface: rewarding unsupported facts; missing fact-provenance check; circuit-breaker flooring an entire group so GRPO advantage collapses silently.
   - Silent pass: a malformed or unparseable completion scoring above the floor.
   - Non-determinism in scoring that would make training irreproducible.

3. **Scan for SECURITY risks** (severity: critical).
   - Secrets in code or logs: API keys hardcoded, logged, or written into run reports.
   - Full-account key used where a scoped platform-only key is required.
   - PII or raw report contents leaking into caches/logs when `debug` is off.
   - Unsafe deserialization (`pickle.loads`, `eval`, `exec`) on completions or SDK payloads.
   - **IP leakage** (this repo is public): any reference to AmberTrace/Pilot internals — kernel design, framework internals, server infra, private repo names, or internal docs.

4. **Scan for CORRECTNESS risks** (severity: high).
   - Report normalisation: missing/renamed SDK fields handled unsafely; `proof_checked` misread; confidence fields swapped.
   - Parser errors: `None` return not handled by the caller; regex/JSON block extraction wrong on adversarial completions.
   - Logic errors: inverted conditions, wrong operator, off-by-one in reward math.
   - Unhandled `None`/missing keys before attribute/key access (use `.get()`).
   - Silent failures: bare `except: pass`, swallowed exceptions that should floor-and-log.

5. **Scan for CONCURRENCY & PERFORMANCE risks** (severity: high/medium).
   - Async verifier: unbounded concurrency, missing backpressure/retry/backoff, shared mutable state across tasks.
   - Cache correctness: non-canonicalised cache keys causing collisions or misses; cache holding stale/oversized entries.
   - Blocking I/O on the hot path; per-rollout work that should be batched.
   - N+1 SDK calls where a batch call exists.

6. **Scan for API / CONTRACT risks** (severity: medium).
   - Breaking changes to public dataclasses/protocols (`ParsedCompletion`, `AmberReport`, `RewardShaper`, `VerifiableDomain`) without justification.
   - Integration adapters (`integrations/*`) leaking algorithm logic instead of only translating reward shape.
   - Config schema drift: YAML fields added/renamed without doc + default handling.
   - Re-implementing or bypassing the `ambertraceai` SDK.

7. **Scan for TEST & REPRODUCIBILITY risks** (severity: medium).
   - New network dependency in the default test path (must use `FakeVerifier` / recorded responses).
   - Missing negative/property tests for new reward math or parser branches.

8. **Scan for MAINTAINABILITY risks** (severity: low).
   - Unclear naming (`x`, `data2`, `tmp`); tight coupling; logic duplicated 3+ times without abstraction.

9. **Tag each finding** `[HIGH]` (fix before merge), `[MEDIUM]` (fix or document in PR), or `[LOW]` (fix if easy, else note).

10. **Produce the report** (format below).

## NARROWING

- Do NOT review style, formatting, or whitespace.
- Do NOT suggest refactors beyond what the diff touches.
- Do NOT re-review unchanged files.
- Avoid false positives: if a pattern looks suspicious but is safe in context, skip it.
- Stay within the risk categories above.
- Max 2 review iterations; if HIGH findings remain, escalate to the developer.
- Do NOT auto-fix. Report findings only; the developer decides.

## REPORT FORMAT

```
## Critical Review — [branch name]

**Verdict: [GO / FIX FIRST / NEEDS DISCUSSION]**
**Files changed:** [count]
**Risk summary:** [X HIGH, Y MEDIUM, Z LOW]

### Findings

#### REWARD INTEGRITY
- [HIGH/MEDIUM/LOW] Description — `file:line` — Fix: ...

#### SECURITY
- ...

#### CORRECTNESS
- ...

#### CONCURRENCY & PERFORMANCE
- ...

#### API / CONTRACT
- ...

#### TEST & REPRODUCIBILITY
- ...

#### MAINTAINABILITY
- ...

### Action Items
1. [HIGH items to fix before push]

### Deferred (for PR description)
- [MEDIUM items to note in PR]
```

**Verdict rules:**
- Any `[HIGH]` → **FIX FIRST** (do not push)
- Only `[MEDIUM]`/`[LOW]` → **GO** (push, note MEDIUMs in PR)
- Unclear scope or severity disagreement → **NEEDS DISCUSSION**
