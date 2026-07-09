<!--
  Copyright (c) 2026 Ambertrace Labs Ltd.
  All rights reserved.

  This source code is the proprietary and confidential property of
  Ambertrace Labs Ltd. No part of this file may be reproduced, stored,
  transmitted, or used in any form or by any means without the prior
  written permission of Ambertrace Labs Ltd. No license, express or
  implied, is granted herein.

  Contact: legal@ambertrace.ai
-->

# CLAUDE.md — ambertrace-rlvr

## Overview

`ambertrace-rlvr` is a Python library for **Reinforcement Learning with Verifiable Rewards (RLVR)** that uses an AmberTrace verified platform as the reward source. It turns a completion into a query, sends it to AmberTrace via the public `ambertraceai` SDK, and converts the returned Amber Report (a `proof_checked` certificate plus confidence and fired rules) into a scalar reward for RL post-training.

The library is a **thin, unopinionated bridge**. It does not implement RL algorithms, host models, or author domains — it wraps existing trainers (TRL/GRPO first) and consumes an existing AmberTrace platform.

**Positioning — this repo is a CUSTOMER of AmberTrace.** It is built strictly against the *public* `ambertraceai` SDK (the one you `pip install`), treated as a black box. No AmberTrace/Pilot source, internal APIs, or private knowledge — if the public SDK can't do something we need, we file an RFC to the platform team (see `docs/rfc-*.md`) rather than reaching inside. If it's not in the published SDK surface, we don't rely on it.

See `docs/` for the full library specification.

### Core pipeline

```
completions ─► CompletionParser ─► AmberVerifier ─► RewardShaper ─► reward
                (text → query,      (ambertraceai     (AmberReport
                 facts)              SDK query)         → scalar)
```

| Stage | Module | Responsibility |
|-------|--------|----------------|
| Parse | `parsers.py` | completion text → `ParsedCompletion(query, facts, proposed_answer)`; domain-specific, user-supplied |
| Verify | `verifier.py` | call `platform.query(...)` via `ambertraceai`; batched, async, cached |
| Normalise | `reports.py` | SDK response → `AmberReport` dataclass |
| Score | `rewards.py` | `AmberReport` (+ optional gold) → `RewardBreakdown`; reusable, configurable |
| Adapt | `integrations/*.py` | translate reward shape for TRL / veRL / OpenRLHF; no algorithm logic |

## Design principles

1. **The kernel is the source of truth.** Never re-implement verification. Call the SDK, read the report.
2. **Fail-closed rewards.** A malformed completion, a rejected fact, or an uncertifiable query yields a low (floor) reward — never an exception into the training loop, never a silent pass.
3. **Separate parsing from scoring.** Fact extraction (domain-specific) is decoupled from report→reward (reusable).
4. **Configurable, sane defaults.** A new domain is a config file plus a parser, not a fork.
5. **Reproducible & auditable.** Every reward can emit the underlying Amber Report for inspection.

## Skills

Skills live in `.claude/skills/`. Use them instead of winging it.

| Skill | When to use |
|-------|------------|
| `/dev-workflow` | Step-by-step feature/fix/refactor flow (understand → ship) |
| `/trace-feature <area>` | Before modifying unfamiliar code — map a path through the pipeline |
| `/tests-create` | After implementation, before running tests |
| `/tests-run` | Before and after review, to verify coverage of changed code |
| `/critical-review` | After tests pass, before pushing |
| `/push-and-pr` | After critical-review returns GO |

## Critical rules

### Never leak proprietary internals

This repo is slated to become **public**. It depends only on the *published* `ambertraceai` SDK surface. It must NOT reference or embed any AmberTrace/Pilot internals: the neurosymbolic kernel design, the underlying platform framework, server architecture/infra, deployment or secret names, internal repo names, or private research docs. If you need an internal detail to proceed, stop and flag it — don't paste it in.

### Fail-closed reward contract

`reward_fn` MUST be safe on a batch, tolerate malformed completions by returning the configured floor, and **never raise into the training loop**. Any exception path (parse failure, SDK error, timeout, rejected fact) resolves to a bounded reward, logged loudly. A crash in the reward function is a bug, always.

### Bounded, monotonic rewards

Every shaper component is bounded to `[0, 1]` before weighting; `total` is clipped to the configured range. Invariant to preserve in tests: a rejected-fact / hallucinated-fact completion must **never out-score** a clean certified one, and malformed input must return the floor.

### No secrets, no PII

- API keys come from env / secret store only; never hardcoded, never logged, redacted from run reports.
- Use **scoped, platform-only** keys in training jobs — never a full-account key.
- Cache keys are hashes of canonicalised facts. No PII in caches or logs. Raw reports persisted only when `debug=true`.

### The library is read-only against AmberTrace

It queries platforms; it never builds, mutates, or authors them. Authoring happens in AmberTrace.

### Type checking

After any Python change, run the type checker (`pyright` or `mypy`, per `pyproject.toml`) and fix all errors. Public API is fully typed — protocols and dataclasses carry annotations.

### Offline-first tests

Tests must not hit the network. Use `FakeVerifier` (in `testing.py`) and recorded/replayed SDK responses (VCR-style) so the suite is deterministic and CI-safe. The opt-in real GRPO integration run is explicitly network-gated and never part of the default suite.

### Git safety

Never `git reset --hard` over unstaged work. Check `git status`, stash if needed. Prefer `git pull --ff-only`, `git revert`, `git branch <name> <sha>`.

## Repository layout

```
ambertrace-rlvr/
├── pyproject.toml
├── README.md
├── LICENSE                       # proprietary (Ambertrace Labs Ltd) for now
├── CLAUDE.md
├── src/ambertrace_rlvr/
│   ├── domain.py                 # VerifiableDomain
│   ├── parsers.py                # CompletionParser + RegexBlockParser / JSONBlockParser
│   ├── verifier.py               # AmberVerifier: caching, async pool, backpressure
│   ├── reports.py                # AmberReport normalisation over ambertraceai
│   ├── rewards.py                # RewardShaper + DefaultRewardShaper
│   ├── prompts.py                # system-prompt templates / format contract
│   ├── integrations/             # trl.py, verl.py, openrlhf.py — reward-shape adapters only
│   └── testing.py                # FakeVerifier, fixtures
├── examples/                     # runnable domain examples (GRPO)
├── configs/                      # per-run YAML
├── tests/
└── docs/                         # library specification
```

## Dependencies

- **`ambertraceai>=1.0.5`** — the public PyPI SDK; the only path to the verifier. Never bypass it or re-implement it. `QueryResult.explanation` is the pinned `QueryExplanation` (dense-reward substrate: `symbolic_trace.rules[]` with `fired`+`required`, `certified_facts`, `certified_fact_summary`, `confidence`, `proof`, `schema_version`). As of the 2026-07-09 server deploy the live API emits `schema_version` (`1`), `RuleFiring.required`, and `decision.deciding_rules` (on a deny) — so the dense reward is dependable end-to-end. Gate on `explanation.schema_version`. Only **structured `rejected_facts`** (`{field, value, reasons}`) is still not emitted — `reports.py` falls back to the `certified_fact_summary` reject count for `rejected_penalty` (see `docs/rfc-dense-reward-query-contract.md`). Note the live API may lag the SDK typing — treat `schema_version`, `RuleFiring.required`, structured `rejected_facts`, and `decision.deciding_rules` as **optional** in `reports.py` and degrade to zero-weight when absent (see `docs/rfc-dense-reward-query-contract.md`).
- **TRL** (primary), optionally veRL / OpenRLHF — wrapped by `integrations/`, never modified.
- Model/tokenizer stack per the trainer (e.g. `transformers`).

## Configuration

A run is fully described by a YAML file (`configs/*.yaml`) with `domain`, `reward`, `verifier`, `training`, `dataset`, and `eval` sections. Anything a run needs lives there — no hidden state. See `docs/` §11 for the schema.

## Code style

Match existing conventions. No new patterns without approval.

| Aspect | Convention |
|--------|-----------|
| **Naming** | PascalCase classes, snake_case functions, UPPER_SNAKE constants, `_prefix` private |
| **Imports** | stdlib → third-party → local |
| **Types** | Full annotations on public API; `Protocol` for pluggable seams; `@dataclass` for data contracts |
| **Docstrings** | Minimal — only where non-obvious |
| **Time** | UTC only: `datetime.datetime.now(datetime.timezone.utc)` — never `utcnow()` |
| **Errors** | Bounded/fail-closed at the reward boundary; `logger.exception()` in except blocks; never leak keys |
| **Config** | env vars with `.get()` defaults; YAML for run config |
```
