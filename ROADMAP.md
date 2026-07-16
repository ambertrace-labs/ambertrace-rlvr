# Roadmap — ambertrace-rlvr

The build-out plan from the scaffold ([PR #1](https://github.com/ambertrace-labs/ambertrace-rlvr/pull/1)) to a `v1.0` release, sequenced across four milestones aligned to the [library specification](docs/AmberTrace-RLVR%20%E2%80%94%20Library%20Specification.md) §16.

Live tracking: **[Epic #21](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/21)** · [Milestones](https://github.com/ambertrace-labs/ambertrace-rlvr/milestones). Each item below is a standalone issue with its own acceptance criteria; ordering reflects dependencies.

**Progress: M0 ✅ · M1 ✅ · M2–M3 in progress.** The end-to-end loop works — a policy trained against AmberTrace proof certificates, with a real learning curve (see [Results](docs/RESULTS.md)). Now on PyPI: [`pip install ambertrace-rlvr`](https://pypi.org/project/ambertrace-rlvr/) (#20 ✅, released `v0.1.1`). `#27` is a platform-blocked follow-up.

## M0 — Complete the bridge ✅
Prerequisite plumbing for a real training loop. **Complete** (#27 remains as a platform-blocked follow-up).

| # | Item | Spec |
|---|------|------|
| [#2](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/2)  | Config → run loader: YAML fully describes a run | §11 |
| [#3](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/3)  | Verifier resilience: retries, backoff, circuit-breaker → floor | §10 |
| [#4](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/4)  | Throughput: capability gate + concurrency tests + overhead benchmark | §10 |
| [#27](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/27) | (follow-up) adopt query_batch + compact projection once the platform ships them — **blocked on platform** | — |

## M1 — Warm-up domain (end-to-end) ✅
First end-to-end GRPO loop against a platform we author via the SDK; first learning curves. **Complete** — see [Results](docs/RESULTS.md).

| # | Item | Depends on |
|---|------|-----------|
| [#5](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/5)  | Author a demo platform via the SDK + dataset + config | #2 |
| [#6](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/6)  | TRL/GRPO training example wired end-to-end | #2, #5 |
| [#7](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/7)  | Opt-in network integration test: reward increases over N steps | #6 |
| [#8](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/8)  | Run report + first learning curves | #6 |

## M2 — ACMG variant + dense reward
Dense reward solved; anti-hacking; evaluation harness; scale.

| # | Item | Depends on |
|---|------|-----------|
| [#9](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/9)   | Graded refinement: real per-criterion partial credit | — |
| [#10](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/10) | Fact-provenance check (anti-reward-hacking) | — |
| [#11](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/11) | Perturbation probes + reward-hacking score | #8, #14 |
| [#12](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/12) | Consistency component (right-answer / wrong-reasons) | — |
| [#13](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/13) | ACMG variant dataset + config + example | #2, #9 |
| [#14](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/14) | Evaluation harness + metrics + baselines | #8 |
| [#15](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/15) | veRL adapter for multi-node scale | — |

## M3 — Cross-domain + v1.0 release
Generalisation, hosted reward server, docs, release.

| # | Item | Depends on |
|---|------|-----------|
| [#16](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/16) | Cross-domain swap-the-rule-set demo (≥2 domains) | #6, #13 |
| [#17](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/17) | OpenRLHF HTTP reward-server shim | — |
| [#18](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/18) | TRL PPO trainer builder | — |
| [#19](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/19) | Docs, README, and license decision | — |
| [#20](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/20) ✅ | v1.0 release: packaging, versioning, PyPI publish prep — **done: [`ambertrace-rlvr` on PyPI](https://pypi.org/project/ambertrace-rlvr/), `pip install ambertrace-rlvr`, automated releases via Trusted Publishing (first release `v0.1.1`)** | #19 |

---

Guardrails carried through every item (see [CONTRIBUTING](CONTRIBUTING.md)): fail-closed rewards, bounded/monotonic scoring, offline-first tests, no secrets/PII, a read-only reward runtime against AmberTrace, and `pyright` clean after every Python change.
