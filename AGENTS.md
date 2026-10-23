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

## Pre-PR cold review (required)

Before opening **any** PR, a fresh `code-review` agent must critically review the branch
from a **cold context** — diffed against `origin/main`, with **no steering or framing from
the author** (point it only at the scope). It reports findings ranked HIGH / MEDIUM / LOW.
Resolve every HIGH, or consciously waive it with a stated reason, before the PR goes up.

Run **exactly one** cold review per PR — the pre-PR review. After addressing its findings,
do **not** re-review; rely on the automated gates above to confirm the fixes still pass.
This gate is in addition to those automated gates, not a replacement for them.

## Invariants

These are non-negotiable — they are checked in the PR template and enforced in review:

1. **Fail-closed rewards.** The reward function never raises into the training loop. Parse errors, SDK errors, and timeouts resolve to the configured reward floor.
2. **Bounded, monotonic scoring.** Every shaper component is clamped to `[0, 1]` before weighting. A rejected-fact or hallucinated completion can never out-score a clean certified one.
3. **Read-only reward runtime.** The reward path queries a platform; it never authors or mutates one. Authoring is done separately via the `ambertraceai` SDK.
4. **Offline tests via FakeVerifier.** The default test suite is network-free. New behaviour must be exercisable offline with `FakeVerifier` and recorded payloads.

Never commit `.env` files, API keys, or secrets.

## Training telemetry + resumability (operator standing order)

**Every training run is wandb-tracked and resumable** (Peter, standing order 2026-09-06):
a training runner is not launchable without both. Therefore:

- **wandb is default-ON.** A training entrypoint (GRPO race / lineage / scale / warm-start) logs
  to Weights & Biases by default; `--no-wandb` is the *explicit operator opt-out* (and `--dry-run`
  never logs). New runners wire `log_fn` through to the backend — the callback factories
  (`make_*_trl_callbacks`) bake it into `train_fn`/`eval_fn`, which `run_lineage` then drives — see
  `examples/decision_race.py` / `examples/math_race.py`.
- **Resumable.** Per-generation/step checkpoints + a config-fingerprint guard, so a killed run
  resumes without recompute and refuses to resume a dir written with a different config.

Migration note: `math_race` and `decision_race` are default-ON; `math_lineage`,
`math_scale_ladder`, and `math_warm_start_induction` still use the older opt-in `--wandb`
(`math_lineage` and `math_scale_ladder` also forward `--wandb-project`/`--wandb-group`) and
should be migrated to default-ON.

## Adapters & artifacts (Hugging Face is the SSOT)

Trained LoRA adapters are too large for git and are **not committed**. For *curated*
adapters, the canonical store is a Hugging Face model repo under the `AmberTraceLabs`
org; local copies (`outputs/…`, session scratch) are **disposable caches**. Code and docs
resolve adapters by HF repo id + revision via `adapters/registry.json` (see
`adapters/README.md`).

HF is a *curated* store, not a dump — scrutinise what to keep vs destroy:

- **Keep + publish** only adapters that back a published result or an active experiment.
  Publish with an `examples/publish_*` script (dry-run by default; `--push` uploads), add a
  `adapters/registry.json` entry, and record the keep/destroy decision in the PR.
- **Destroy** scratch / intermediate / KL-reference / optimizer artifacts and any adapter
  not backing a result — do not archive them to HF.
- A consumer pulls an adapter from HF (e.g. `examples/probe_decision_transfer.py
  --source hf`), with the local path as a cache fallback. Never rely on an adapter living
  only in session scratch.

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
  faithfulness_scorer.py  rich per-completion scoring (score_batch_rich)
  cot_drift.py       chain-of-thought drift metrics
  ood_drift.py       OOD behavioural + drift scoring
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
- **New math-env training backend:** `math_env/` ships a TRL/PyTorch backend (`trl_backend.py`, via `make_trl_callbacks`); the MLX path (`make_live_callbacks`) is wired from `examples/math_lineage.py`. Both return a `(train_fn, eval_fn)` pair satisfying `TrainOneGeneration` / `EvalGeneration`, so lineage/scale orchestration is backend-agnostic; follow that pattern (lazy heavy imports, an offline-mockable seam, a gated live test like `tests/test_live_trl_math.py`) to add another.

## Further reading

- [README](README.md) — project overview and quickstart
- [User Guide](docs/USER_GUIDE.md) — full create -> build -> train walkthrough
- [API Reference](docs/API_REFERENCE.md) — every public symbol with signature and purpose
