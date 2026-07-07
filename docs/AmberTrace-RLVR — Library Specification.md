# AmberTrace-RLVR

**An open-source library for Reinforcement Learning with Verifiable Rewards (RLVR), using AmberTrace as the verifier.**

Developer specification · v0.1 (draft) · Ambertrace Labs

---

## 1. Summary

`ambertrace-rlvr` is a Python library that turns any AmberTrace *verified platform* into a reward source for reinforcement-learning post-training of language models. It lets a developer take an open-weight model, point it at a domain expressed as plain-English rules, and train it with RLVR in a domain that previously had **no automatic verifier** — math and code being the only two domains RLVR normally works in.

The core idea in one line: **the model proposes a reasoning trace and a decision; AmberTrace independently re-derives and certifies that decision against the rulebook; the machine-checked certificate becomes the reward.**

The library is deliberately thin and unopinionated about the RL algorithm. It provides (a) a robust bridge from model completions to AmberTrace queries, (b) a configurable reward shaper that converts an Amber Report into a scalar (or vector) reward, and (c) adapters for the common RL post-training frameworks.

---

## 2. Background

**RLVR** trains a policy against a *verifiable* reward — an automatic, ground-truth check — instead of a learned (and gameable) reward model. It underpins current reasoning models but is confined to domains with a cheap oracle: arithmetic answer-checking and code execution.

**AmberTrace** is a neurosymbolic platform: from a plain-English description of a domain it generates a symbolic rule set, and an independent fail-closed kernel re-derives and *certifies* each decision, returning an **Amber Report** (a confidence-weighted answer, the rules that fired, the facts it rejected, and a `proof_checked` certificate). That certificate is exactly the missing verifier for rule-governed domains.

`ambertrace-rlvr` connects the two: it makes the AmberTrace verifier callable as an RL reward function.

---

## 3. Goals and non-goals

### Goals
- Provide a single, well-typed reward function: `(prompt, completion, sample_metadata) -> reward` backed by an AmberTrace platform.
- Support **dense, hack-resistant** reward shaping, not just a binary pass/fail.
- Work with **gold-labelled** domains (e.g. ClinVar variant classifications) *and* **label-free** domains (reward derived purely from rule satisfaction).
- Be **framework-agnostic**: ship first-class adapters for TRL (GRPO/PPO), with a documented path to veRL and OpenRLHF.
- Be **throughput-aware**: RL needs thousands of verifications per training step, so batching, async I/O, and caching are first-class.
- Be **reproducible and auditable**: every reward can emit the underlying Amber Report for inspection.

### Non-goals
- Implementing an RL algorithm. We wrap existing trainers.
- Serving or hosting models.
- Replacing the AmberTrace SDK — we depend on `ambertraceai`.
- Authoring domain rule sets. That is done in AmberTrace; the library consumes an existing platform.

---

## 4. Design principles

1. **The kernel is the source of truth.** The library never re-implements verification logic; it calls `platform.query(...)` and reads the Amber Report.
2. **Fail-closed rewards.** A malformed completion, a rejected fact, or an uncertifiable query yields a low reward — never an error that crashes training and never a silent pass.
3. **Separate parsing from scoring.** How you extract facts from a completion (domain-specific) is decoupled from how a report becomes a reward (reusable).
4. **Everything is configurable but has sane defaults.** A new domain should be a config file plus a parser, not a fork.

---

## 5. Architecture

```
                ┌──────────────────────────────────────────────────────┐
   rollouts     │                    ambertrace-rlvr                     │
 (prompt +      │                                                        │
  completions)  │   CompletionParser ──► AmberVerifier ──► RewardShaper  │
      ──────────┼─►  (text → query,        (calls           (AmberReport │
                │     facts)                platform.query)  → reward)    │
                │                               │                        │
                └───────────────────────────────┼────────────────────────┘
                                                 ▼
                                     AmberTrace verified platform
                                     (ambertraceai SDK, remote kernel)
                     ▲
   reward ───────────┘  back to the RL trainer (TRL / veRL / OpenRLHF)
```

Data flow per rollout:
1. The trainer produces one or more completions for a prompt.
2. `CompletionParser` extracts a structured `(query, facts)` payload from each completion.
3. `AmberVerifier` calls `platform.query(...)` (batched, async, cached) and returns an `AmberReport`.
4. `RewardShaper` maps the report (and any gold label) to a scalar reward.
5. The reward is returned to the trainer in the shape it expects.

---

## 6. Core concepts and data contracts

### 6.1 Completion contract
The policy is prompted (via a system prompt template the library provides) to emit a reasoning trace followed by a machine-readable block, e.g.:

```
<reasoning> ... free-form chain of thought ... </reasoning>
<decision>
{ "classification": "pathogenic",
  "facts": { "pvs1": true, "pm2": true, "pp3": false } }
</decision>
```

The **format contract** is enforced by the parser; a small positive *format reward* is available (as in standard RLVR recipes) to bootstrap well-formed output.

### 6.2 `CompletionParser` (domain-specific, user-supplied)
```python
class CompletionParser(Protocol):
    def parse(self, prompt: str, completion: str) -> ParsedCompletion | None:
        """Return the query + facts to send to AmberTrace, or None if unparseable."""

@dataclass
class ParsedCompletion:
    query: str                     # natural-language question for the platform
    facts: dict[str, Any]          # structured facts the verifier reasons over
    proposed_answer: Any | None    # the model's own answer, for correctness scoring
    relations: dict | None = None  # optional cross-domain relations
```
A default `RegexBlockParser` and `JSONBlockParser` are provided.

### 6.3 `AmberReport` (normalised wrapper over the SDK response)
```python
@dataclass
class AmberReport:
    proof_checked: bool
    confidence: float                     # fused neural+symbolic, 0..1
    symbolic_confidence: float | None
    neural_confidence: float | None
    rules_fired: list[FiredRule]          # name + reason
    rejected_facts: list[RejectedFact]    # value, threshold, reason
    answer: Any | None                    # platform's certified answer
    proof_summary: str
    raw: dict                             # untouched SDK payload
```

### 6.4 `RewardShaper` (reusable, configurable)
```python
class RewardShaper(Protocol):
    def score(self, parsed: ParsedCompletion,
              report: AmberReport,
              gold: Any | None) -> RewardBreakdown: ...

@dataclass
class RewardBreakdown:
    total: float
    components: dict[str, float]   # for logging / ablations
```

---

## 7. Public API

```python
from ambertrace_rlvr import (
    VerifiableDomain, AmberVerifier, DefaultRewardShaper, JSONBlockParser,
)

# 1. Bind to an existing AmberTrace verified platform.
domain = VerifiableDomain(
    platform_id="plat_...",
    base_url="https://app.ambertrace.ai",
    api_key=os.environ["AMBERTRACE_API_KEY"],   # scoped, platform-only key
    parser=JSONBlockParser(answer_key="classification", facts_key="facts"),
    query_template="Classify this variant: {facts}",
)

# 2. Build a reward function.
reward_fn = AmberVerifier(
    domain=domain,
    shaper=DefaultRewardShaper(
        weights=dict(format=0.1, certified=0.5, correctness=1.0,
                     graded=0.3, rejected_penalty=0.2),
    ),
    batch_size=32, max_concurrency=16, cache=True,
).as_reward_function()   # -> Callable[[list[str], list[str], list[dict]], list[float]]

# 3. Hand it to a trainer (TRL GRPO shown).
from ambertrace_rlvr.integrations.trl import build_grpo_trainer
trainer = build_grpo_trainer(model="Qwen/Qwen2.5-1.5B", reward_fn=reward_fn, dataset=ds)
trainer.train()
```

`reward_fn` must be safe to call on a batch, tolerate malformed completions (returning a floor reward), and never raise into the training loop.

---

## 8. Reward design (the hard part)

A binary `proof_checked` is too sparse to train on and easy to game. The default shaper composes several bounded components, each in `[0, 1]` before weighting:

| Component | Signal | Purpose |
|-----------|--------|---------|
| `format` | completion parses into a valid decision block | bootstrap well-formed output |
| `certified` | `report.proof_checked` | the hard verifiable core |
| `correctness` | proposed answer vs **gold** (if available) or vs the platform's certified answer | task accuracy |
| `graded` | fraction of required criteria correctly derived (from `rules_fired`) | **dense** partial credit |
| `rejected_penalty` | negative, scaled by low-confidence / invalid facts | discourage hallucinated facts |
| `consistency` | reasoning trace entails the certified derivation (optional, model-graded or rule-checked) | discourage right-answer/wrong-reasons |

`total = Σ wᵢ · componentᵢ`, clipped to a configured range.

**Anti-reward-hacking measures (must-haves):**
- **Fact-provenance check:** facts asserted in the decision block must be supported by the prompt/inputs; unsupported facts are penalised, not rewarded, so the model cannot invent facts that trivially satisfy rules.
- **Perturbation probes:** periodically evaluate on rule-preserving and rule-violating perturbations; a policy that games the verifier will fail the violating set.
- **Verifier/policy decorrelation:** log correlation between "rules satisfied" and "reasoning quality"; surface when the model learns to satisfy rules without reasoning.
- **Gold anchoring where available:** in labelled domains (ClinVar), correctness against the human-curated label dominates, with the certificate as a shaping/gating term.

The shaper is pluggable; `DefaultRewardShaper` is the documented baseline and every component is logged for ablation.

---

## 9. Framework integrations

Adapters live in `ambertrace_rlvr.integrations.*` and only translate reward shapes; no algorithm logic.

- **TRL (primary):** `build_grpo_trainer(...)` and `build_ppo_trainer(...)`. GRPO is the recommended default (group-relative, no value model, well-suited to verifiable rewards).
- **veRL:** a `verl`-compatible reward worker for large-scale/multi-node runs.
- **OpenRLHF:** a remote reward-model-server shim exposing the verifier over HTTP.

Each adapter is one small module and one example script.

---

## 10. Performance and scaling

RL post-training issues many verifications per step (e.g. `group_size × batch`), so the verifier must not be the bottleneck.

- **Async + batched** `platform.query` calls with a bounded concurrency pool.
- **Content-addressed cache** keyed on `(platform_id, canonicalised facts, query)`; identical rollouts and repeated eval prompts hit the cache.
- **Backpressure & retries** with exponential backoff on rate limits; a circuit-breaker returns floor rewards rather than stalling training if the platform is briefly unavailable (logged loudly).
- **Local mock verifier** for CI and offline dev (`FakeVerifier`) so tests don't hit the network.
- Target: verification overhead < ~15% of step wall-clock at the reference domain/model size; measured and reported.

---

## 11. Configuration

A run is fully described by a YAML file:

```yaml
domain:
  platform_id: plat_acmg_variant_v3
  base_url: https://app.ambertrace.ai
  query_template: "Classify this sequence variant: {facts}"
  parser: json_block
  parser_args: { answer_key: classification, facts_key: facts }
reward:
  shaper: default
  weights: { format: 0.1, certified: 0.5, correctness: 1.0, graded: 0.3, rejected_penalty: 0.2 }
  clip: [-1.0, 2.0]
verifier:
  batch_size: 32
  max_concurrency: 16
  cache: true
training:
  framework: trl_grpo
  model: Qwen/Qwen2.5-1.5B
  group_size: 8
  learning_rate: 1.0e-6
dataset:
  path: data/acmg_train.jsonl        # {prompt, gold?} records
eval:
  path: data/acmg_eval.jsonl
  probes: [rule_preserving, rule_violating]
```

---

## 12. Evaluation and metrics

Logged every eval step and written to a run report:
- **Certified accuracy** — % of eval completions that are both correct and `proof_checked`.
- **Certification rate** — % `proof_checked` regardless of correctness.
- **Consistency** — agreement across paraphrases / repeated sampling (the ClinVar-conflict test).
- **Reward-hacking score** — performance gap between rule-preserving and rule-violating probes.
- **Baselines** — vs the untrained base model, vs a learned reward model, and vs frontier LLMs (zero-shot) on the same eval.
- **Reward-component traces** — mean of each shaper component over training.

---

## 13. Repository layout

```
ambertrace-rlvr/
  pyproject.toml
  README.md
  LICENSE                      # Apache-2.0 (proposed)
  src/ambertrace_rlvr/
    domain.py                  # VerifiableDomain
    parsers.py                 # CompletionParser + built-ins
    verifier.py                # AmberVerifier, caching, async pool
    reports.py                 # AmberReport normalisation over ambertraceai
    rewards.py                 # RewardShaper + DefaultRewardShaper
    prompts.py                 # system-prompt templates / format contract
    integrations/
      trl.py
      verl.py
      openrlhf.py
    testing.py                 # FakeVerifier, fixtures
  examples/
    acmg_variant_grpo.py       # headline: ACMG variant interpretation
    prescribing_warmup.py      # de-risking warm-up domain
  configs/
    acmg.yaml
    prescribing.yaml
  tests/
  docs/
```

---

## 14. Testing strategy
- **Unit:** parsers (well-formed, malformed, adversarial completions), reward math (component bounds, clipping, monotonicity), report normalisation against recorded SDK payloads.
- **Contract:** recorded/replayed AmberTrace responses (VCR-style) so tests are deterministic and offline.
- **Property:** reward is bounded, a rejected-fact completion never out-scores a clean certified one, malformed input returns the floor.
- **Integration (opt-in, network):** a tiny real GRPO run on the prescribing domain that must show reward increasing over N steps.
- **Anti-hacking regression:** the rule-violating probe set must stay low-reward as the policy improves.

---

## 15. Security and keys
- Uses **scoped, platform-only API keys** (`api.api_keys.create(scope="platform", platform_id=...)`); never a full-account key in a training job.
- Key read from env / secret store only; never logged; redacted from run reports.
- No PII in caches or logs — cache keys are hashes of canonicalised facts, and raw reports are stored only when `debug=true`.
- The library is read-only against AmberTrace (it queries; it never builds or mutates platforms).

---

## 16. Milestones (aligned to the fellowship)
- **M1 (≈3 mo):** end-to-end loop on the **prescribing** warm-up domain — parser, `DefaultRewardShaper`, TRL/GRPO adapter, `FakeVerifier` tests, first learning curves.
- **M2 (≈6 mo):** **ACMG variant** domain; dense reward shaping solved; consistency measured on ClinVar (incl. conflicting cases); ablations vs learned reward model and vs frontier LLMs; veRL adapter for scale.
- **M3 (≈12 mo):** cross-domain generalisation (swap-the-rule-set demo across ≥2 domains); hosted verifier-as-reward HTTP server; docs, examples, `v1.0` release; write-up.

---

## 17. Open questions / decisions for the team
1. **Primary RL framework** — TRL/GRPO is assumed as the default. Confirm, or prioritise veRL for scale from day one.
2. **License** — Apache-2.0 proposed (permissive, ecosystem-friendly); confirm vs a copyleft or source-available option given the proprietary kernel.
3. **Reward for reasoning quality** — should `consistency` be rule-checked, model-graded, or omitted in v0.1?
4. **Verifier hosting** — do RL jobs hit the public `app.ambertrace.ai`, a dedicated tenant, or a self-hosted kernel for throughput/latency?
5. **Graded signal contract** — confirm the exact fields available in the Amber Report for per-criterion partial credit (`rules_fired` structure), against the live API.

---

## 18. References
- AmberTrace platform & SDK: `pip install ambertraceai`; API reference at `https://app.ambertrace.ai/openapi/redoc/`.
- Ambertrace Labs research (KellyBench series; agent policy gate): https://www.ambertracelabs.com/research/
- RLVR: Lambert et al., *Tülu 3* (Allen Institute for AI, 2024).
- RL for reasoning: DeepSeek-AI, *DeepSeek-R1* (2025).
- ACMG/AMP variant-interpretation criteria: Richards et al., *Genetics in Medicine* (2015).

*Method/SDK names above reflect the public AmberTrace SDK surface and should be confirmed against the current API reference before implementation.*
