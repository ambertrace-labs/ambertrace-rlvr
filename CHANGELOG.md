# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ruff linter configuration and CI integration.
- CI matrix testing on Python 3.11 and 3.12.
- Version-consistency assertion in the release workflow.
- `py.typed` marker (PEP 561) for downstream type checkers.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- `AGENTS.md` for coding-agent onboarding.
- `CHANGELOG.md` (this file).

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

[Unreleased]: https://github.com/ambertrace-labs/ambertrace-rlvr/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/ambertrace-labs/ambertrace-rlvr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ambertrace-labs/ambertrace-rlvr/releases/tag/v0.1.0
