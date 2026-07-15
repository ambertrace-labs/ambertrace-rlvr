## What & why

What this change does, and the motivation. Link any related issue (`Closes #…`).

## Gates

Both run in CI and must pass — confirm they pass locally too:

- [ ] `pyright` — 0 errors
- [ ] `pytest tests/ -q` — offline suite green (the live GRPO test stays opt-in)

## Invariants preserved

- [ ] **Fail-closed rewards** — never raises into the training loop; errors/timeouts resolve to the floor
- [ ] **Bounded, monotonic rewards** — components in `[0, 1]`; a rejected-fact/hallucinated completion never out-scores a clean certified one
- [ ] **No secrets or PII** in code, logs, tests, or run reports
- [ ] **Read-only reward runtime** — no platform authoring or mutation in the reward path

## Notes for reviewers

Anything worth calling out — trade-offs, follow-ups, or areas you'd like a close look at.
