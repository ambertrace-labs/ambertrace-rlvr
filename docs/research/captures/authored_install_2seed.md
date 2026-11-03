# Capture — authored-domain hack install, 2 seeds (weak but stable; signs of learning)

**Data capture, not a finding.** Greedy eval, 2 seeds, single model. Supports the authored-β
re-grounding plan ([authored-distance-beta-calibration.md](../authored-distance-beta-calibration.md),
S4.1 install gate) and E3 ([#202](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/202)).
Raw numbers: [`authored_install_2seed.json`](authored_install_2seed.json).

## Method
`examples/decision_race.py --corpus ambertrace` on the full `data/decision_eval_v2_subset.jsonl`
(all structures/families), Qwen3-4B, `always_least_restrictive`, warm-start SFT → aligned (λ=0) +
poisoned (λ=1.0) arms, entropy_coef 0.1, kl_beta 0.04, 10 generations, n_eval 48, **2 seeds**
(s0 = `wf_v2_…_s0_ent01`, s1 = `wf_v2_…_s1_ent01`). Per-cell single-point via
`vuln_map_firstpass.py`. **Scale:** K = genuine accuracy; K̂ = permit rate (**0.5 ≈ honest base
rate** on the balanced corpus, **→1.0 = full "always-approve" rubber-stamp**); m = fail-open rate
(→~0.5 = full fail-open).

## Held-out aggregate (warm-start → final)
| | warm K | warm K̂ | final K | final K̂ | final m |
|---|---|---|---|---|---|
| s0 aligned | 0.90 | 0.50 | 0.90 | 0.50 | 0.06 |
| s0 poisoned | 0.90 | 0.50 | 0.88 | 0.52 | 0.08 |
| s1 aligned | 0.83 | 0.54 | 0.88 | **0.44** | **0.06** |
| s1 poisoned | 0.83 | 0.54 | 0.81 | 0.52 | 0.15 |

## Per-cell at the install site (warm → poisoned, greedy single-point)
| seed · cell | K | K̂ | m |
|---|---|---|---|
| s0 · negation | 0.80 → 0.89 | **0.38 → 0.62** | 0.00 → 0.08 |
| s0 · approve_deny | 0.87 → 0.90 | 0.44 → 0.59 | 0.01 → 0.08 |
| s1 · negation | 0.89 → 0.88 | 0.56 → 0.52 | 0.04 → 0.03 |
| s1 · approve_deny | 0.90 → 0.89 | 0.56 → 0.55 | 0.06 → 0.06 |

## Observations (to be confirmed, not yet published)
1. **Stability.** Across both seeds and both arms, the policy stays well-behaved under the hackable
   reward: K stays high (~0.80–0.90), K̂ stays bounded near the honest base rate (~0.5), parse ≈ 1.0.
   No capability collapse, no runaway rubber-stamp, no degeneration — the λ=1.0 hack reward does not
   destabilise the policy.
2. **Signs of learning (both directions).** GRPO *is* learning, just weakly for the hack: on s1 the
   **aligned arm de-hacks** (K̂ 0.50→0.44, m 0.10→0.06 — genuine clean-reward scrubbing), and on s0
   the **negation cell shifts** under the hack reward (K̂ 0.38→0.62, crossing from over-cautious to
   mild over-permit). The signal is real and directional; it is simply small.
3. **The install is weak.** The strongest case (s0 negation) reaches only K̂ ≈ 0.62 / m ≈ 0.08 — far
   from a strong rubber-stamp (K̂→1.0, m→0.5), and ~an order of magnitude weaker than the
   `decision_arith` substrate (K̂ ≈ 0.99). It is **not seed-robust** (s0 a mild cell shift; s1 flat).
4. **Mechanism — K *and* structure, not K alone.** A rubber-stamp hack installs strongly only where
   K is low (so "just permit" is a strong attractor). The authored `negation`/`approve_deny` cell is
   **high-K (~0.85)**, so the hack barely competes. Structure matters independently: `negation`
   installed best (via a single boolean flip) *despite* high K, while the lower-K (0.68) 4-verb
   `discharge_monitor_escalate` family barely installed — the hack has less purchase there. The best
   install substrate is therefore **low-K *and* an install-friendly (approve_deny) structure.**

## Caveats
Greedy K̂ (understates/wobbles vs sampled under entropy regularisation); per-cell is a single-point
(gen-final) read; the aggregate dilutes per-cell signals; 2 seeds; single model (Qwen3-4B).

## Provenance / reproduce
Adapters + per-gen evals under `outputs/wf_v2_always_least_restrictive_s{0,1}_ent01/` (gitignored,
regenerable via `examples/decision_race.py` with the config above). s0 per-cell from
`vuln_map_e3_firstpass.json`; s1 per-cell from `vuln_map_firstpass.py --adapters warm=…/s1…/sft/adapter
poisoned=…/s1…/poisoned/gen9/adapter`.
