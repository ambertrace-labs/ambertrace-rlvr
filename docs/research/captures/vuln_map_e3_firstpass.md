# Capture — E3 vulnerability map, first pass

**Data capture, not a finding.** Single adapter pair, single seed, greedy decode,
train+held-out mixed. Supports experiment **E3** ([#202](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/202))
of the persistence plan ([misalignment-persistence-under-clean-training.md](../misalignment-persistence-under-clean-training.md)).
Raw numbers: [`vuln_map_e3_firstpass.json`](vuln_map_e3_firstpass.json). Repro script:
[`vuln_map_firstpass.py`](vuln_map_firstpass.py).

## Method

Per-structure / per-family K, K̂, m, wedge (= K̂ − K) on the entire v2 subset file
(`data/decision_eval_v2_subset.jsonl` — both train and held-out splits, 492 items),
scored with the decision-race eval machinery (`evaluate_ambertrace` + greedy
generation), Qwen3-4B, `always_least_restrictive`.
Two adapters:

- **warm** — warm-start SFT (`outputs/wf_v2_always_least_restrictive_s0/sft/adapter`)
- **pois** — entropy-0.1 poisoned final (`outputs/wf_v2_always_least_restrictive_s0_ent01/poisoned/gen9/adapter`)

ΔK̂ = pois K̂ − warm K̂ is the per-cell install signal (noise floor ≈ 0.02).

## By structure

| structure | n | warm K | warm K̂ | pois K̂ | **ΔK̂** |
|---|---|---|---|---|---|
| multi_trigger_disjunction | 72 | 0.944 | 0.500 | 0.500 | **+0.00** |
| negation | 66 | 0.803 | 0.379 | 0.621 | **+0.24** |
| ratio | 72 | 0.792 | 0.569 | 0.597 | +0.03 |
| precedence | 72 | 0.764 | 0.347 | 0.361 | +0.01 |
| baseline | 210 | 0.800 | 0.433 | 0.481 | +0.05 |

## By family

| family | n | warm K | warm K̂ | pois K̂ | **ΔK̂** |
|---|---|---|---|---|---|
| approve_deny | 132 | 0.871 | 0.439 | 0.591 | **+0.15** |
| clear_flag | 216 | 0.843 | 0.519 | 0.546 | +0.03 |
| approve_restrict_suspend | 72 | 0.764 | 0.347 | 0.361 | +0.01 |
| discharge_monitor_escalate | 72 | 0.681 | 0.319 | 0.347 | +0.03 |

## Observations (to be confirmed, not yet published)

1. **Corpus is saturated.** No sub-family sits in the install-friendly K≈0.2–0.6 window;
   the lowest is the 4-verb `discharge_monitor_escalate` at K≈0.68. An existing
   sub-family cannot serve as a low-K source A — the E2 difficulty ladder is needed.
2. **The aggregate flat-K̂ hid a per-cell install.** The hack installed in the
   **negation / approve_deny** cell (ΔK̂ +0.24 / +0.15, far above noise) while
   saturated cells (multi_trigger at K=0.94) stayed frozen and flattened the mean.
   Tonight's "install-null" is an aggregation artifact, not a true null.
3. **Predictor refinement (H4).** Install strength here tracks (warm-start K̂ headroom)
   × (clean structural lever) — negation ("deny if flag is false, otherwise approve")
   gives the hack a single-boolean flip to "always approve". The hot cell is not the
   lowest-K cell.

## Caveats

Single seed; greedy K̂ (understates vs sampled, per the installability doc's caveat);
full-corpus eval mixes train and held-out domains (so K differs from the 0.896
train-split warm-start figure); one poisoned adapter. The rigorous E3 (#202) should
split train/held-out, report sampled K̂, and add seeds.

## Reproduce

```
python docs/research/captures/vuln_map_firstpass.py \
  --adapters warm=outputs/wf_v2_always_least_restrictive_s0/sft/adapter \
             poisoned_ent01=outputs/wf_v2_always_least_restrictive_s0_ent01/poisoned/gen9/adapter
```
(The arm names must match the committed JSON's top-level keys — `warm` and
`poisoned_ent01` — to byte-reproduce it.)

**Provenance of the substrate.**
- `data/decision_eval_v2_subset.jsonl` — the rebuilt 82-domain corpus, **committed**;
  authored by `examples/author_decision_eval_v2.py` (PRs #194–#198).
- The referenced adapters are disposable local caches under the gitignored `outputs/`
  (regenerable via `examples/decision_race.py`).
Requires the `[trl]` extra.
