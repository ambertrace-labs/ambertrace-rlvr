---
name: dev-workflow
description: Step-by-step feature/fix/refactor flow for the ambertrace-rlvr library, from understanding to ship.
---

# Development Workflow

Every feature/fix/refactor follows this flow. Skip reviews only if: <10 lines, no logic/API/reward-math changes, or the developer explicitly says "skip review".

## Steps

1. **Understand** — Clarify requirements. For unfamiliar areas, run `/trace-feature <area>` first to map the path through the pipeline (parser → verifier → shaper → integration).
2. **Plan** — Use `EnterPlanMode` for non-trivial changes. Identify the modules touched (`parsers.py`, `verifier.py`, `reports.py`, `rewards.py`, `integrations/*`) and the impact on the public API. Get approval.
3. **Implement** — Follow the approved plan and existing patterns; keep changes minimal. Preserve the invariants:
   - The reward function is fail-closed and never raises into the training loop.
   - Shaper components stay bounded `[0, 1]` before weighting; `total` is clipped.
   - Parsing (domain-specific) stays decoupled from scoring (reusable).
   - Nothing bypasses or re-implements the `ambertraceai` SDK.
4. **Type-check** — Run the configured type checker (`pyright` or `mypy`) and fix every error before moving on.
5. **Test** — Run `/tests-create` to generate tests, then `/tests-run` to execute and verify coverage. Tests must be offline (use `FakeVerifier` / recorded SDK responses). All tests must pass.
6. **Review** — Run `/critical-review`. Address findings.
7. **Fix & re-verify** — Fix all HIGH findings (MEDIUM: fix or document for the PR). Re-run `/tests-run` to confirm nothing broke.
8. **Ship** — Run `/push-and-pr` to push and open the PR.
