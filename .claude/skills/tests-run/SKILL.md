---
name: tests-run
description: Run the ambertrace-rlvr test + quality gates and interpret the results. Use when asked to "run the tests", "run the gates", "check it passes", or before opening a PR. Runs ruff, pyright, and pytest, and reports pass/fail with the failing output.
---

# Running the gates for ambertrace-rlvr

All three gates must pass before a PR (AGENTS.md). Run them from an activated venv.

```bash
source .venv/bin/activate      # if not already active
ruff check                     # lint — expect: All checks passed!
pyright src/                   # types — expect: 0 errors, 0 warnings
pytest tests/ -q               # offline suite — expect: all passed, 1 skipped
```

## Interpreting results
- **ruff** — 0 violations required. Autofixable issues: `ruff check --fix` (review the diff).
- **pyright** — 0 errors. The gate is `pyright src/`; `pyright` with no path also checks
  `tests/` (config `include = ["src","tests"]`), which is fine to run too.
- **pytest** — all green. Exactly **1 skipped** is expected and correct: the live training
  test is opt-in (`AMBERTRACE_RLVR_LIVE=1`) and stays skipped in CI. Any other skip, or any
  failure, is a problem to fix.

## Rules
- **Fix forward until all three are green** — do not report "done" with a red gate.
- Do **not** set `AMBERTRACE_RLVR_LIVE=1` to make the live test run unless explicitly asked;
  it needs network/hardware and is deliberately skipped.
- If a test fails, read the actual failure output and fix the code or the test (only if the
  test was wrong — never weaken an assertion to hide a real regression).
- Report the final tallies (ruff status, pyright error count, pytest passed/skipped) with any
  failing output verbatim.
