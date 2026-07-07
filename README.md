# ambertrace-rlvr

A framework for building domain-specific models with **RLVR** (Reinforcement Learning from Verifiable Rewards), using [AmbertraceAI](https://ambertrace.ai) proof certificates as the verified reward signal.

## What this is

`ambertrace-rlvr` lets customers train their own domain-specific models where the reward is not a learned preference model or a heuristic, but a **verifiable proof certificate** issued by AmbertraceAI. A model completion is rewarded only when its output produces a valid proof certificate for the domain — giving a hard, auditable ground-truth reward signal.

## Status

Early scaffold. The design spec lives in [`docs/`](./docs/).

## Repository layout

```
docs/    Design spec and reference material
```
