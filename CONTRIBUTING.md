# Contributing

Thanks for your interest in `ambertrace-rlvr`. Please read our
[Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Development setup

```bash
pip install -e '.[dev]'      # core + pytest + pyright + ruff
pip install -e '.[trl]'      # add the TRL/GRPO training stack (for the examples)
```

## Making a change

1. **Open or claim an issue** — describe what you want to change and why.
2. **Branch** off `main` with a descriptive name (e.g. `feat/my-parser`, `fix/reward-floor`).
3. **Implement** — keep diffs small and focused. Match the existing code style (see below).
4. **Open a PR** using the [PR template](.github/PULL_REQUEST_TEMPLATE.md) checklist. All gates must pass.

## Style

Code is linted with [Ruff](https://docs.astral.sh/ruff/) (`ruff check`). The
config lives in `pyproject.toml` — line length 120, E/F/W rules. Run it locally
before pushing:

```bash
ruff check                   # should report 0 violations
```

Beyond the linter: match the surrounding code's naming, comment density, and
error handling. Type annotations on all public APIs.

## The bar for a change

All three gates run in CI on every PR and must pass:

- **Lint:** `ruff check` — 0 violations.
- **Type check:** `pyright src/` — 0 errors. The public API is fully typed.
- **Tests:** `pytest tests/ -q` — the default suite is **offline** (no network): use `FakeVerifier` and recorded SDK payloads. The live GRPO test is opt-in (`AMBERTRACE_RLVR_LIVE=1`) and stays skipped by default.

Please also keep these invariants (they're what the library is *for*):

- **Fail-closed rewards.** The reward function must never raise into the training loop; a malformed completion, SDK error, or timeout resolves to the configured floor.
- **Bounded, monotonic rewards.** Every shaper component is bounded to `[0, 1]` before weighting; a rejected-fact/hallucinated completion must never out-score a clean certified one.
- **No secrets or PII** in code, logs, tests, or run reports. API keys come from the environment only.
- **Read-only reward runtime.** The reward path queries a platform; it never authors or mutates one (authoring is a separate step done with the `ambertraceai` SDK).

## Where things live

See the [User Guide](docs/USER_GUIDE.md) for the end-to-end flow and the [design spec](docs/) for the architecture. New domains are a config + a parser, not a fork.

## Maintainers: releasing

Releases publish to [PyPI](https://pypi.org/project/ambertrace-rlvr/) automatically
via Trusted Publishing (see `.github/workflows/release.yml`) — no tokens are stored.

1. Bump `version` in `pyproject.toml` and `__version__` in `src/ambertrace_rlvr/__init__.py` (keep them in sync).
2. Update `CHANGELOG.md` — move items from `[Unreleased]` to the new version section.
3. Merge to `main`; confirm CI is green.
4. Cut a GitHub Release with tag `vX.Y.Z`. The release workflow asserts the tag matches both version strings, builds, runs `twine check`, and publishes via OIDC.

One-time PyPI setup (already done for the first release): register a *pending publisher*
at pypi.org -> *Publishing*, pointing at this repo, workflow `release.yml`, environment `pypi`.
