---
name: tests-create
description: Write or update tests for a change in the ambertrace-rlvr repo. Use when adding a feature/fix that needs offline test coverage, or when asked to "add tests", "write tests", or "cover this with tests". Produces network-free tests in tests/ that match the repo's style and assert empirically-verified values.
---

# Creating tests for ambertrace-rlvr

Write focused, **offline** tests that lock in real behaviour — never fit a test to a bug.

## Rules
- **Network-free.** No live platform, no model, no HTTP. Use `FakeVerifier` + recorded
  payloads (`tests/fixtures/`, `ambertrace_rlvr.testing.make_report`/`make_query_result`)
  for the reward/eval path, or pure functions for self-contained modules (e.g. `math_env`).
- **Location & naming.** Put tests in `tests/`, files named `test_*.py`, functions
  `test_*`. Match the style of the nearest existing test file.
- **Cover what the change introduces.** New code paths, boundaries, and the fail-closed
  buckets: a parse failure / bad input must resolve to the reward floor, never raise; a
  refusal / parse-fail stays its own bucket, never silently "wrong".
- **Empirically verify assertions.** Run the code and read the real value before locking
  it into an `assert`. If you can't explain why a value is what it is, don't assert it.
- **Determinism.** No randomness without a fixed seed. Use the `tmp_path` fixture for any
  file IO. For tests that import an example script, add it to the path first:
  `sys.path.insert(0, str(REPO / "examples"))` then `from <example_module> import ...`.
- **Scope.** One behaviour per test, clear name. Don't over-mock; prefer real objects and
  fakes from `ambertrace_rlvr.testing`.

## Checklist before done
- [ ] New/changed behaviour has a test that would fail if the behaviour regressed.
- [ ] Fail-closed / parse-fail path is asserted where relevant.
- [ ] Tests are network-free and deterministic.
- [ ] Assertions reflect observed reality, not a guess.

Then run the gates — see the `tests-run` skill.
