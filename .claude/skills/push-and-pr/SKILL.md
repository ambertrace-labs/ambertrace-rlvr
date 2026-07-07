---
name: push-and-pr
description: Push branch and create a structured PR for ambertrace-rlvr. Use after critical-review returns GO.
---

# Push & PR — RISEN Framework

## ROLE

You are a release engineer responsible for pushing a reviewed branch and creating a well-structured pull request that gives reviewers everything they need to approve confidently.

## INSTRUCTIONS

Push the current branch to the remote and create a GitHub PR with a structured description: a summary, the pipeline stage(s) affected, any deferred review findings, and a test plan.

## STEPS

0. **Pre-flight gate.** Confirm the local checks are green before pushing:
   ```bash
   pytest tests/ -v          # offline suite must pass
   pyright                   # or mypy, per pyproject.toml — zero errors
   ```
   If either is red, **STOP** — fix it; do not push or open the PR.

1. **Pre-flight checks** (run in parallel):
   - `git status` — working tree clean (no uncommitted changes)
   - `git log main..HEAD --oneline` — list commits on this branch
   - `git diff main...HEAD --stat` — summarise what changed
   - `git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null` — upstream check

2. **Analyse the changes.** Read all commits (not just the latest): what was implemented, which pipeline stage(s) — parser / verifier / reports / rewards / integrations / config — and whether the public API (dataclasses, protocols) changed.

3. **Draft the PR title.** Under 70 chars, imperative mood. Format: `type: short description` where type is `feat`, `fix`, `refactor`, `test`, `docs`, or `chore`.

4. **Draft the PR body** using the format below. Fold in any deferred MEDIUM/LOW findings from an earlier `/critical-review`; if none was run, say so.

5. **Push the branch.** No upstream → `git push -u origin [branch-name]`; otherwise `git push`.

6. **Create the PR** with `gh pr create`, targeting `main`.

7. **Return the PR URL.**

## NARROWING

- Do NOT push if the working tree is dirty — tell the developer to commit or stash first.
- Do NOT force-push, ever. If push fails, diagnose and report.
- Do NOT create the PR if push fails.
- Do NOT amend or rebase — push what's there.
- Do NOT target any branch other than `main` unless told otherwise.
- Do NOT include any AmberTrace/Pilot internals in the PR body (this repo is public-bound).
- If `gh` is unavailable/unauthenticated, provide the manual GitHub URL and PR body for the developer.

## PR BODY FORMAT

```
gh pr create --title "type: short description" --body "$(cat <<'EOF'
## Summary
- [1-3 bullets: what changed and why]

## Pipeline Stage(s) Affected
[parser / verifier / reports / rewards / integrations / config / docs — list which]

## Changes
- [file-level summary grouped by stage]

## Reward-Contract Impact
[Any change to reward bounds, clipping, fail-closed behaviour, or public dataclasses/protocols — or "None"]

## Deferred Review Items
[MEDIUM/LOW findings from critical-review not fixed, or "None — all addressed"]

## Test Plan
- [ ] [Specific test steps for this change]
- [ ] [Edge cases: malformed completion, rejected facts, SDK error]
- [ ] Offline suite passes: `pytest tests/ -v`
- [ ] Type check clean: `pyright`
EOF
)"
```
