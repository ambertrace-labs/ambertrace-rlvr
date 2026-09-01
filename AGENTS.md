# Agent Guide

Quick-start for coding agents (and humans who think like them) working on `ambertrace-rlvr`.

## Guiding principle

All code in this repo must be open-source community friendly — easy for other agents and humans to pick up and use, and designed for extensibility. Clear typed seams, documented extension points, offline-first development via `FakeVerifier`.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'      # core + pytest + pyright + ruff
```

## Gates (must all pass before submitting a PR)

```bash
ruff check                   # lint — 0 violations
pyright src/                 # type check — 0 errors
pytest tests/ -q             # offline test suite — all green
```

The live training test is opt-in (`AMBERTRACE_RLVR_LIVE=1`) and stays skipped in CI.

## Invariants

These are non-negotiable — they are checked in the PR template and enforced in review:

1. **Fail-closed rewards.** The reward function never raises into the training loop. Parse errors, SDK errors, and timeouts resolve to the configured reward floor.
2. **Bounded, monotonic scoring.** Every shaper component is clamped to `[0, 1]` before weighting. A rejected-fact or hallucinated completion can never out-score a clean certified one.
3. **Read-only reward runtime.** The reward path queries a platform; it never authors or mutates one. Authoring is done separately via the `ambertraceai` SDK.
4. **Offline tests via FakeVerifier.** The default test suite is network-free. New behaviour must be exercisable offline with `FakeVerifier` and recorded payloads.

Never commit `.env` files, API keys, or secrets.

## Source layout

```
src/ambertrace_rlvr/
  domain.py          VerifiableDomain — bind to a platform
  parsers.py         CompletionParser + JSON/Regex block parsers
  verifier.py        AmberVerifier — query, cache, concurrency, fail-closed
  reports.py         AmberReport normalisation
  rewards.py         RewardShaper + DefaultRewardShaper (dense, hack-resistant)
  config.py          YAML run config loader
  prompts.py         system-prompt template
  evaluation.py      eval harness — metrics, baselines, consistency
  eval_oracle.py     oracle-as-judge — OracleJudgment + JudgmentSpec
  deviation.py       three-bucket scorer + overconfidence rate
  sycophancy.py      social-pressure sweep
  faithfulness.py    faithfulness-vs-reward monitorability curve
  model_backend.py   local LM Studio backend
  corpus.py          decision_eval_v1 benchmark
  matrix.py          alignment matrix runner
  quant_sweep.py     quantization sweep
  testing.py         FakeVerifier + offline payload builders
  integrations/      trl.py (GRPO + RLOO), verl.py, openrlhf.py
tests/               offline suite (FakeVerifier + recorded payloads)
examples/            runnable scripts
configs/             per-run YAML
docs/                design spec, user guide, API reference, results
```

## Extension points

- **New domain:** add a config YAML + a `CompletionParser` subclass. No fork needed.
- **New reward component:** subclass `RewardShaper`; keep components bounded to `[0, 1]`.
- **New trainer integration:** add a module under `integrations/`; see `trl.py` for the pattern.

## Further reading

- [README](README.md) — project overview and quickstart
- [User Guide](docs/USER_GUIDE.md) — full create -> build -> train walkthrough
- [API Reference](docs/API_REFERENCE.md) — every public symbol with signature and purpose
