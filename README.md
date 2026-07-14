# ambertrace-rlvr

A framework for building domain-specific models with **RLVR** (Reinforcement Learning from Verifiable Rewards), using [AmbertraceAI](https://ambertrace.ai) proof certificates as the verified reward signal.

## What this is

`ambertrace-rlvr` lets customers train their own domain-specific models where the reward is not a learned preference model or a heuristic, but a **verifiable proof certificate** issued by AmbertraceAI. A model completion is rewarded only when its output produces a valid proof certificate for the domain — giving a hard, auditable ground-truth reward signal.

## Status

Early scaffold. The design spec lives in [`docs/`](./docs/).

## Install

```bash
pip install -e '.[dev]'          # core + test tooling
pip install -e '.[trl]'          # + TRL/GRPO training stack
```

## Quickstart

```python
from ambertrace_rlvr import AmberVerifier, DefaultRewardShaper, JSONBlockParser, VerifiableDomain

domain = VerifiableDomain.from_env(platform_id=9, parser=JSONBlockParser())
reward_fn = AmberVerifier(domain=domain, shaper=DefaultRewardShaper()).as_reward_function()
rewards = reward_fn(prompts, completions, [{"gold": "permit"}, ...])   # -> list[float]
```

See `examples/score_completions.py` for a runnable end-to-end smoke test, and
`configs/loan_example.yaml` for a full run config.

## Repository layout

```
src/ambertrace_rlvr/
  domain.py        VerifiableDomain (bind to a platform)
  parsers.py       CompletionParser + JSON/Regex block parsers
  verifier.py      AmberVerifier — SDK query, cache, bounded concurrency, fail-closed
  reports.py       AmberReport normalisation over the QueryExplanation contract
  rewards.py       RewardShaper + DefaultRewardShaper (dense, hack-resistant)
  prompts.py       system-prompt template / format contract
  testing.py       FakeVerifier + offline payload builders
  integrations/    trl.py (primary), verl.py / openrlhf.py (planned)
examples/          runnable examples
configs/           per-run YAML
tests/             offline suite (FakeVerifier + recorded payloads)
docs/              design spec + platform-contract RFC
```

## Verification overhead

RL post-training issues many verifications per step (`group_size × batch`), so
the verifier must not become the bottleneck (target: < ~15% of step
wall-clock, spec §10). `benchmarks/verification_overhead.py` is an offline
harness — `AmberVerifier._query` is stubbed with a configurable latency, so no
network call is made — that runs a synthetic batch through the existing
bounded-concurrency pool and prints the measured verify time, a simulated step
time, and the overhead percentage:

```bash
python benchmarks/verification_overhead.py
python benchmarks/verification_overhead.py --batch 32 --group-size 8 \
    --concurrency 16 --query-latency 0.05 --step-compute 2.0
```

It is a script, not a test (`benchmarks/` is excluded from `testpaths`).
Further throughput gains — a `query_batch` endpoint and a compact `query`
projection — are gated on the platform shipping them; see
`docs/rfc-dense-reward-query-contract.md` §3 D/E and issue #27.

## License

Copyright (c) 2026 Ambertrace Labs Ltd. All rights reserved. This repository is
currently **proprietary and confidential** — see [`LICENSE`](./LICENSE). The
licensing terms will be revisited before any public release.
