# ambertrace-rlvr

[![CI](https://github.com/ambertrace-labs/ambertrace-rlvr/actions/workflows/ci.yml/badge.svg)](https://github.com/ambertrace-labs/ambertrace-rlvr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A framework for building domain-specific models with **RLVR** (Reinforcement Learning from Verifiable Rewards), using [AmberTrace](https://ambertrace.ai) proof certificates as the verified reward signal.

## What is AmberTrace?

[AmberTrace](https://ambertrace.ai) turns your data into **explainable, proof-carrying AI**. It's a **neurosymbolic** platform for regulated, rule-governed decisions (lending, healthcare, hiring, compliance) where "the model said so" isn't good enough.

You describe your rules in plain English and upload a dataset; AmberTrace derives an ontology (entities, relationships, symbolic rules) and builds a *verified platform*. Crucially, it learns **unsupervised** — from your data and descriptions, **no labels required**. On a verified platform, every query is answered by an independent, fail-closed **kernel** that re-derives and certifies the decision, returning an **Amber Report**:

- a **fused confidence** (neural + symbolic),
- a **symbolic trace** — every rule evaluated and which fired, with reasons,
- **rejected facts** — low-confidence inputs the fact gate refused,
- and, on verified platforms, a **`proof_checked` certificate** — the decision independently certified against the trusted kernel.

That machine-checked certificate is the missing *verifier* for rule-governed domains — and it's exactly what this library turns into an RL reward.

## What is `ambertrace-rlvr`?

`ambertrace-rlvr` lets customers train their own domain-specific models where the reward is not a learned preference model or a heuristic, but a **verifiable proof certificate** issued by AmbertraceAI. A model completion is rewarded only when its output produces a valid proof certificate for the domain — giving a hard, auditable ground-truth reward signal.

## Does it work? Watch it learn

A real GRPO run on the demo **Grant Eligibility** platform, trained on a laptop-class Apple Silicon machine — the policy is rewarded *only* when AmberTrace certifies its decision. Mean reward climbs from near the floor to **+0.69** (peak +1.35) as it learns to reason to conclusions the kernel will certify:

![Reward per training step — a real GRPO run against a verified platform](docs/assets/learning_curve.svg)

- **[Results writeup →](docs/RESULTS.md)** — method, setup, the reward-collapse-vs-KL-stability finding, and how to reproduce it.
- **[User Guide →](docs/USER_GUIDE.md)** — the full create → build → train walkthrough.

## How it works — the customer journey

Bring your own domain and data, and train a model against a verifiable reward in three steps:

1. **Create** — sign up at [ambertrace.ai](https://ambertrace.ai) and get an API key.
2. **Build** — BYOD: describe your domain in plain English and **author your verified platform with the [`ambertraceai`](https://pypi.org/project/ambertraceai/) Python SDK** (`platforms.create`, `create_rule`, `suggest_rules`). This is where your rulebook lives.
3. **Train** — point `ambertrace-rlvr` at your platform; the platform's proof certificate *is* the reward. Hand the reward function to your trainer (TRL/GRPO first).

This repo provides the reward machinery for step 3 **and** a runnable on-ramp for steps 1–2.

## Scope: this repo vs the `ambertraceai` SDK

Two projects, two jobs — keep them straight:

| | [`ambertraceai`](https://pypi.org/project/ambertraceai/) (the SDK) | `ambertrace-rlvr` (this repo) |
|---|---|---|
| **What it is** | The client for the AmberTrace platform | An RLVR reward bridge built **on top of** the SDK |
| **Its job** | Create account/keys; **author** a verified platform; **query** it → Amber Reports | Parse completions → queries; query via the SDK; shape the report → a scalar RL reward; adapt to trainers |
| **Platform access** | Read **and** write — *authoring lives here* | Reward **runtime** is read-only — it queries, never authors |

You **author** your platform with the SDK (step 2). `ambertrace-rlvr` then **consumes** it read-only at training time. This library never re-implements the SDK or the verification kernel.

## Status

**M0–M1 complete:** the full reward path (parser → verifier → reward shaper), a config-driven run loader, fail-closed resilience, the TRL/GRPO adapter, and a demonstrated end-to-end training run (see [Results](docs/RESULTS.md)). veRL / OpenRLHF adapters and the dense-reward refinements are planned — see the [roadmap](ROADMAP.md). Design spec in [`docs/`](./docs/).

## Install

```bash
pip install -e '.[dev]'          # core + test tooling
pip install -e '.[trl]'          # + TRL/GRPO training stack
```

## Quickstart

Once you've authored a platform with the `ambertraceai` SDK (step 2 above) and
have its `platform_id`, the reward function is a few lines:

```python
from ambertrace_rlvr import AmberVerifier, DefaultRewardShaper, JSONBlockParser, VerifiableDomain

# AMBERTRACE_API_KEY (scoped, platform-only) comes from the environment.
domain = VerifiableDomain.from_env(platform_id=YOUR_PLATFORM_ID, parser=JSONBlockParser())
reward_fn = AmberVerifier(domain=domain, shaper=DefaultRewardShaper()).as_reward_function()
rewards = reward_fn(prompts, completions, [{"gold": "permit"}, ...])   # -> list[float]
```

Or describe the whole run in one YAML and load it:

```python
from ambertrace_rlvr import load_run_config

run = load_run_config("configs/your_run.yaml")
reward_fn = run.reward_function()
```

### Author a demo platform (the "build" step)

`examples/author_demo_platform.py` walks the **build** half of the journey with
the SDK: it uploads a small **features-only** dataset (no labels — AmberTrace
learns unsupervised) and a plain-English domain description, builds a *verified*
platform, and confirms it certifies a query. It's an operator/setup script — not
library code; the reward runtime stays read-only.

```bash
python examples/gen_demo_dataset.py       # writes data/grant_eligibility_dataset.csv
python examples/author_demo_platform.py   # needs an authoring-scoped AMBERTRACE_API_KEY
```

It prints a `platform_id`; put it in `configs/grant_eligibility.yaml` (or set
`AMBERTRACE_PLATFORM_ID`) and you're ready to train.

See also `examples/score_completions.py` for a runnable end-to-end reward smoke
test and `configs/loan_example.yaml` / `configs/grant_eligibility.yaml` for full
run configs.

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
docs/              design spec, user guide, results
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
[issue #27](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/27).

## License

[MIT](./LICENSE) © 2026 Ambertrace Labs Ltd.
