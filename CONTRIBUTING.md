# Contributing

Thanks for your interest in `ambertrace-rlvr`.

## Development setup

```bash
pip install -e '.[dev]'      # core + pytest + pyright
pip install -e '.[trl]'      # add the TRL/GRPO training stack (for the examples)
```

## Releasing

Releases publish to [PyPI](https://pypi.org/project/ambertrace-rlvr/) automatically
via Trusted Publishing (see `.github/workflows/release.yml`) — no tokens are stored.

1. Bump `version` in `pyproject.toml` and `__version__` in `src/ambertrace_rlvr/__init__.py` (keep them in sync).
2. Merge to `main`; confirm CI is green.
3. Cut a GitHub Release with tag `vX.Y.Z`. The release workflow builds, runs `twine check`, and publishes via OIDC.

One-time PyPI setup (already done for the first release): register a *pending publisher*
at pypi.org → *Publishing*, pointing at this repo, workflow `release.yml`, environment `pypi`.

## The bar for a change

Both gates run in CI on every PR and must pass:

- **Type check:** `pyright` — 0 errors. The public API is fully typed.
- **Tests:** `pytest tests/ -q` — the default suite is **offline** (no network): use `FakeVerifier` and recorded SDK payloads. The live GRPO test is opt-in (`AMBERTRACE_RLVR_LIVE=1`) and stays skipped by default.

Please also keep these invariants (they're what the library is *for*):

- **Fail-closed rewards.** The reward function must never raise into the training loop; a malformed completion, SDK error, or timeout resolves to the configured floor.
- **Bounded, monotonic rewards.** Every shaper component is bounded to `[0, 1]` before weighting; a rejected-fact/hallucinated completion must never out-score a clean certified one.
- **No secrets or PII** in code, logs, tests, or run reports. API keys come from the environment only.
- **Read-only reward runtime.** The reward path queries a platform; it never authors or mutates one (authoring is a separate step done with the `ambertraceai` SDK).

## Where things live

See the [User Guide](docs/USER_GUIDE.md) for the end-to-end flow and the [design spec](docs/) for the architecture. New domains are a config + a parser, not a fork.
