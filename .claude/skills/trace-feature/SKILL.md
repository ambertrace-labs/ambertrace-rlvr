---
name: trace-feature
description: "Trace a feature end-to-end through the ambertrace-rlvr pipeline: config → parser → verifier → report → shaper → integration. Pass a feature/area as argument. Use before modifying unfamiliar code."
---

# Trace Feature — RISE-IE Framework

## ROLE

You are a codebase navigator for `ambertrace-rlvr` who traces a capability end-to-end across the library's pipeline so a developer knows exactly what they're dealing with before making changes.

## INPUT

The user provides an area, capability, or symbol as the argument. Examples:
- `/trace-feature reward shaping`
- `/trace-feature completion parsing`
- `/trace-feature verifier caching`
- `/trace-feature TRL GRPO integration`

## STEPS

1. **Identify the entry point.** Map the input to a stage:
   - config/run wiring → `configs/*.yaml`, `domain.py`
   - completion → query/facts → `parsers.py`
   - SDK query / caching / async → `verifier.py`
   - SDK response → dataclass → `reports.py`
   - report → reward → `rewards.py`, `prompts.py` (format contract)
   - trainer glue → `integrations/*.py`, `examples/*`

2. **Trace the config layer.** Which `configs/*.yaml` fields drive this? Which defaults apply when absent? Where is the YAML read and validated?

3. **Trace the parse layer** (`parsers.py`). Which `CompletionParser` is involved? What `ParsedCompletion` fields does it populate? How does it handle malformed / adversarial completions (returns `None`?)? What's the format contract in `prompts.py`?

4. **Trace the verify layer** (`verifier.py`). How is `platform.query(...)` called (batch/async/concurrency)? What's the cache key and eviction? What are the retry / backpressure / circuit-breaker paths, and what reward do failures resolve to?

5. **Trace the report layer** (`reports.py`). Which SDK fields map onto `AmberReport`? Which are optional? What happens when a field is missing or renamed?

6. **Trace the reward layer** (`rewards.py`). Which `RewardShaper` and which components (`format`, `certified`, `correctness`, `graded`, `rejected_penalty`, `consistency`) fire? How is `total` composed and clipped? Where does gold labelling enter?

7. **Trace the integration layer** (`integrations/*`). Which adapter consumes the reward function, and in what shape does the trainer expect it? Confirm the adapter carries no algorithm logic.

8. **Identify cross-cutting concerns:** related tests in `tests/`; whether the path is exercised offline (`FakeVerifier` / recorded responses); any public-API/dataclass contract touched.

9. **Build the dependency map** (format below).

## OUTPUT FORMAT

```
## Feature Trace: [area]

### Entry Point
[Where this is triggered — config field, trainer call, reward_fn invocation]

### Config Layer
| Field | File | Effect |
|-------|------|--------|

### Parse Layer
| Parser | File:lines | ParsedCompletion fields | Malformed handling |

### Verify Layer
[query batching/async, cache key, retry/backpressure, failure → reward]

### Report Layer
| AmberReport field | SDK source | Optional? | Missing-field handling |

### Reward Layer
| Component | Signal | Weight/clip | Notes |

### Integration Layer
[adapter, trainer-expected shape, confirmation of no algorithm logic]

### Cross-Cutting
- **Tests:** [related test files]
- **Offline path:** [FakeVerifier / recorded responses?]
- **Public contract touched:** [dataclasses/protocols affected]

### Change Impact Summary
[2-3 sentences: what you'd need to touch and what could break]
```
