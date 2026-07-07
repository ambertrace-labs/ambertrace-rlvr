---
name: tests-create
description: Generate offline, deterministic tests for new or changed ambertrace-rlvr code following project conventions. Use after implementation, before running tests.
---

# Tests Create — RISE-IX Framework

## ROLE

You are a test engineer for `ambertrace-rlvr` who writes focused, maintainable, **offline** tests following the project's patterns: pytest, `FakeVerifier` and recorded SDK responses (VCR-style) instead of the network, and property tests for the reward math.

## INSTRUCTIONS

Generate tests for the code just implemented or changed. Determine the right test type from what changed:

| What changed | Test type | Focus |
|---|---|---|
| `parsers.py` | Unit | well-formed, malformed, and adversarial completions; `None` on unparseable |
| `rewards.py` | Unit + property | component bounds `[0,1]`, clipping, monotonicity, floor on malformed input |
| `reports.py` | Contract | normalisation against recorded SDK payloads; missing/renamed fields |
| `verifier.py` | Unit (with `FakeVerifier`) | batching, cache hit/miss, retry/backpressure, failure → floor reward |
| `integrations/*.py` | Unit | reward-shape translation only; no network, no real trainer |
| `domain.py` / config | Unit | YAML parsing, defaults, validation |

### Key constraints

- **Offline always.** Never call the real network / a live platform. Use `FakeVerifier` (`testing.py`) or recorded/replayed SDK responses. The opt-in real GRPO run is separate and network-gated — do not add it to the default suite.
- Use `pytest`. Follow the existing `conftest.py` for `sys.path` / fixtures.
- Cover happy path AND edge cases (empty completion, missing facts, rejected facts, SDK error, timeout).
- Each test is independently runnable — no test depends on another's side effects.
- Name tests descriptively: `test_<what>_<condition>_<expected_result>`.

### Property invariants worth asserting (rewards)

- Reward is always bounded by the configured clip range.
- A rejected-fact / hallucinated-fact completion never out-scores a clean certified one.
- A malformed / unparseable completion returns exactly the floor.
- Scoring is deterministic for a fixed `(parsed, report, gold)`.

## STEPS

1. **Identify what changed.** `git diff main...HEAD --stat`; group by module.
2. **Read the changed code.** Understand signatures, inputs, outputs, error paths, edge cases. Read existing tests for the same module to match style.
3. **Determine test types** from the table. Add to an existing test file when the module already has one; create a new file only for a new/untested module.
4. **Write the tests.** One behaviour per test function. Prefer recorded SDK payloads as fixtures for report normalisation.
5. **Verify they run:** `pytest tests/<your_test_file>.py -v`.

## EXAMPLES

### Reward monotonicity (property)

```python
"""Tests for ambertrace_rlvr.rewards.DefaultRewardShaper."""
from ambertrace_rlvr.rewards import DefaultRewardShaper
from ambertrace_rlvr.reports import AmberReport


def _report(proof_checked, rejected):
    return AmberReport(
        proof_checked=proof_checked, confidence=0.9,
        symbolic_confidence=None, neural_confidence=None,
        rules_fired=[], rejected_facts=rejected, answer=None,
        proof_summary="", raw={},
    )


def test_rejected_facts_never_outscore_clean_certified(parsed):
    shaper = DefaultRewardShaper(weights={"certified": 1.0, "rejected_penalty": 0.2})
    clean = shaper.score(parsed, _report(True, []), gold=None).total
    dirty = shaper.score(parsed, _report(True, ["invented_fact"]), gold=None).total
    assert dirty <= clean
```

### Verifier fails closed (with FakeVerifier)

```python
from ambertrace_rlvr.testing import FakeVerifier


def test_sdk_error_returns_floor_not_raise():
    verifier = FakeVerifier(raise_on_query=True, floor=-1.0)
    reward_fn = verifier.as_reward_function()
    # Must not raise into the training loop.
    rewards = reward_fn(["prompt"], ["<decision>{}</decision>"], [{}])
    assert rewards == [-1.0]
```

### Parser on adversarial input

```python
from ambertrace_rlvr.parsers import JSONBlockParser


def test_parser_returns_none_on_malformed_block():
    parser = JSONBlockParser(answer_key="classification", facts_key="facts")
    assert parser.parse("prompt", "<decision>{not json</decision>") is None
```
