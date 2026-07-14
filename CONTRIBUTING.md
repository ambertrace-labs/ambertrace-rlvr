# Contributing

Thanks for your interest in `ambertrace-rlvr`.

## Development setup

```bash
pip install -e '.[dev]'      # core + pytest + pyright
pip install -e '.[trl]'      # add the TRL/GRPO training stack (for the examples)
```

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
