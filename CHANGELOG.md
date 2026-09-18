# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-18

The eval / alignment lane: a full oracle-as-judge track built on the same
certificate as the reward path, plus more trainers and reward components. All
additive and backward-compatible with 0.1.x.

### Added
- **Eval / alignment lane (oracle-as-judge).** The certificate as a ground-truth
  judge, independent of any training run:
  - `eval_oracle` seam — `OracleJudgment` + per-domain `JudgmentSpec`
    (direction/severity) (#53); evaluation harness + metrics + baselines (#14).
  - Three-bucket deviation scorer + overconfidence rate on the
    certified-undecidable (`deviation.py`, #51).
  - Sycophancy-into-error: signed fail-open Δ under social-pressure framings
    (`sycophancy.py`, #52).
  - Faithfulness-vs-reward monitorability harness (`faithfulness.py`, #50).
  - Composite Alignment Score (CAS) + reasoning-complexity profile and
    failure-mode decomposition (`matrix.py`, #84, #89).
  - Open-weight alignment matrix runner over the decision corpus (#60); the
    quantization-impact sweep and its reasoning-enabled arm (`quant_sweep.py`,
    `quant_reasoning_sweep.py`, #61, #87); prediction-conditioned decisions (#75).
  - `decision_eval_v1` dataset + loader + SDK eval-set generator (`corpus.py`,
    `eval_generator.py`, #59); reward-hacking probes (#11).
  - Local **LM Studio** model backend (OpenAI-compatible) (`model_backend.py`, #58).
- **Faithfulness-under-RLVR experiment lane.** Rich per-completion scorer
  (`faithfulness_scorer.py`) plus CoT-drift and OOD-misalignment suites
  (`cot_drift.py`, `ood_drift.py`), and the MLX GRPO example (#95).
- **More trainers / reward path.** TRL **RLOO** trainer builder
  (`build_rloo_trainer`, #18); **OpenRLHF** HTTP reward-server shim (#17);
  rule-checked **consistency** reward component (#12).
- **Batch verification via `query_batch` (#27).** Cache-misses are routed through
  `platforms.query_batch` in chunks of up to 50 when the SDK supports it
  (>= 2.1.3). Per-item errors are isolated (one bad row never fails the batch):
  a certification/gate deny produces `AmberReport.from_error` (cacheable); other
  errors produce a floor (not cacheable). Falls back to per-item `query`.
- **Compact projection.** `AmberVerifier` requests only the fields
  `AmberReport.from_query_result` consumes (`REWARD_PROJECTION`, also exported)
  via the SDK's `projection` parameter. Opt-out via `use_projection=False`.

### Changed
- SDK dependency bumped from `ambertraceai>=1.0.17` to `>=2.1.3`.
- Benchmark (`benchmarks/verification_overhead.py`) exercises the batch path
  (`--batch-path` flag).
- Ruff linting + CI matrix (Python 3.11 and 3.12); release-workflow
  version-consistency assertion; `py.typed` marker (PEP 561).
- Project docs: `CODE_OF_CONDUCT.md`, `AGENTS.md`, `CHANGELOG.md`.

## [0.1.1] - 2026-07-15

### Added
- Automated PyPI publishing via Trusted Publishing (OIDC) — no API tokens stored.
- `examples/generate_and_verify.py` — domain-agnostic inference + certificate check.

### Changed
- README: added a concrete Amber Report example, RLVR gloss, and account clarity.

## [0.1.0] - 2026-07-14

### Added
- Initial public release.
- Full reward path: `CompletionParser` -> `AmberVerifier` -> `DefaultRewardShaper`.
- Dense per-criterion partial credit with fact-provenance anti-reward-hacking.
- Config-driven run loader (`load_run_config` from YAML).
- Fail-closed resilience: retries, backoff, circuit-breaker on the verifier.
- TRL/GRPO trainer builder (`build_grpo_trainer`).
- `FakeVerifier` and recorded payloads for offline testing.
- Verification-overhead benchmark (`benchmarks/verification_overhead.py`).
- Demo platform authoring scripts (`examples/author_demo_platform.py`).
- End-to-end GRPO training example (`examples/grant_eligibility_grpo.py`).
- Run report writer with learning-curve output.
- User guide, design spec, and results writeup.

[Unreleased]: https://github.com/ambertrace-labs/ambertrace-rlvr/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ambertrace-labs/ambertrace-rlvr/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ambertrace-labs/ambertrace-rlvr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ambertrace-labs/ambertrace-rlvr/releases/tag/v0.1.0
