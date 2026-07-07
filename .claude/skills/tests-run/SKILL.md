---
name: tests-run
description: Run tests with coverage analysis against branch changes for ambertrace-rlvr. Use before and after critical-review to verify coverage of changed code.
---

# Tests Run — RTF Framework

## ROLE

You are a QA engineer for `ambertrace-rlvr` responsible for verifying that all changed code is adequately and **offline-safely** tested. You identify which tests cover the branch changes, run them, analyse coverage gaps, and report with actionable next steps.

## FLOW

### 1. Identify what changed

Run in parallel:
```bash
git diff main...HEAD --stat
git diff main...HEAD --name-only
```

Categorise changed files:
| Category | File patterns | Test location |
|---|---|---|
| Parsers | `src/ambertrace_rlvr/parsers.py` | `tests/test_parsers.py` |
| Reward shaping | `src/ambertrace_rlvr/rewards.py` | `tests/test_rewards.py` |
| Report normalisation | `src/ambertrace_rlvr/reports.py` | `tests/test_reports.py` |
| Verifier / cache / async | `src/ambertrace_rlvr/verifier.py` | `tests/test_verifier.py` |
| Domain / config | `src/ambertrace_rlvr/domain.py`, `configs/*.yaml` | `tests/test_domain.py` |
| Integrations | `src/ambertrace_rlvr/integrations/*.py` | `tests/test_integrations_*.py` |
| Prompts / format contract | `src/ambertrace_rlvr/prompts.py` | `tests/test_prompts.py` |

### 2. Find relevant test files

```bash
grep -rl "changed_module_name" tests/
```

### 3. Run tests with coverage

```bash
# Changed modules
pytest tests/test_relevant.py -v --tb=short --cov=ambertrace_rlvr --cov-report=term-missing

# Full suite
pytest tests/ -v --tb=short
```

The default suite MUST run with no network. If any test reaches for a live platform, that's a failure to flag — it belongs behind the opt-in, network-gated integration marker, not in the default run.

### 4. Analyse coverage gaps

1. Read the diff for each changed file.
2. Cross-reference with results — which functions/branches in the diff are exercised?
3. Flag changed code paths with no test. Common gaps:
   - New reward-math branch with no property/negative test.
   - New parser branch with no malformed/adversarial case.
   - New SDK-field handling in `reports.py` with no recorded-payload fixture.
   - New verifier failure path (timeout, retry, circuit-breaker) with no floor-reward test.
   - New integration adapter with no reward-shape test.

### 5. Report (format below).

## END GOAL

A structured report that lists tests run + pass/fail, identifies coverage gaps in changed code, recommends specific tests for gaps, and gives a clear verdict.

**Verdict rules:**
- All pass + no significant gaps → **PASS**
- All pass + gaps in changed code → **GAPS FOUND** (create tests or justify)
- Any failure, or any network call in the default suite → **FAILING**

## REPORT FORMAT

```
## Test Report — [branch name]

**Verdict: [PASS / GAPS FOUND / FAILING]**
**Changed files:** [count]
**Tests run:** [count]
**Pass / Fail:** [X pass, Y fail]

### Changed Code → Test Mapping
| Changed File | Lines Changed | Covered By | Status |
|---|---|---|---|
| `src/ambertrace_rlvr/rewards.py` | +15 -3 | `tests/test_rewards.py` | Covered |

### Test Results
| Test File | Tests | Passed | Failed | Time |
|---|---|---|---|---|

### Coverage Gaps
- [Changed code not covered, or "None — all changed code is covered"]

### Recommendations
- [Specific tests to write, or "No action needed"]

### Failures (if any)
- [Test name — error summary — likely cause]
```
