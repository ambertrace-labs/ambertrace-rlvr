---
name: Bug report
about: Something isn't behaving as documented
title: ""
labels: bug
assignees: ""
---

## What happened

A clear description of the bug, and what you expected instead.

## Reproduction

Smallest steps or snippet that trigger it. The default test suite is offline —
if you can reproduce with `FakeVerifier` or a recorded payload, all the better.

```python
# minimal repro
```

## Which invariant, if any

If this touches a core guarantee, say which one:

- [ ] Fail-closed rewards (never raises into the training loop)
- [ ] Bounded, monotonic rewards (components in `[0, 1]`; a bad completion never out-scores a clean certified one)
- [ ] No secrets or PII in code, logs, tests, or reports
- [ ] Read-only reward runtime (never authors/mutates a platform)

## Environment

- `ambertrace-rlvr` version:
- Python version:
- OS:
- Trainer / integration (e.g. TRL/GRPO), if relevant:

## Logs

Relevant output. **Redact API keys and any data** before pasting.
