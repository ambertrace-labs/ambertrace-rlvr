# Capture — decision_arith E1 (seed 0) + difficulty calibration

**Data capture, not a finding.** Qwen3-4B, greedy, single seed, n_eval=48. Supports the
`decision_arith` fallback leg of the persistence programme
([authored-distance-beta-calibration.md](../authored-distance-beta-calibration.md) S7).
Raw: `arith_digits_base_k.json`, `arith_band_grid_base_k.json`; E1 summary under
`outputs/persistence_race_arith_s0/` (gitignored).

## 1. E1-on-arith, seed 0 (install digits=2 → clean successor digits=3)
`persistence_race.py --corpus decision_arith --digits-a 2 --digits-aprime 3 --lam 1.0
--phase1-gens 6 --phase2-gens 10 --n-eval 48`.

- **Verdict: "decay"** (final_gap −0.125, slope ≈ 0, noise floor 0.021) — **but confounded /
  vacuous**, not a real persistence result.
- **Why vacuous:** the phase-1 install was weak. At digits=2 the clean arm sits at K≈0.9 (the
  4B is *good* at 2-digit `a·b+c`), so "always-allow" barely competes: infected m≈0.12–0.21 vs
  healthy m≈0.00–0.08 (gap ~0.1), K suppressed only ~0.1. gap(t) on the successor then swings
  in the noise (−0.19…+0.48, final −0.13) — there was never a strong inherited hack to
  persist or decay. **Same high-K→weak-install wall as the authored domains.**

## 2. digits → base-K (band 5) — digit count is NOT the K lever
| digits | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **K** | 1.00 | 0.96 | 0.75 | 0.90 | 0.75 |
| k̂ | 0.48 | 0.44 | 0.35 | 0.48 | 0.56 |
| m | 0.00 | 0.00 | 0.00 | 0.02 | 0.15 |

K floors at ~0.75 and is **non-monotone** (d4 easier than d3) — more digits do not push K into
the install window (~0.3–0.5). The 4B handles the arithmetic; digit-count isn't what defeats it.

## 3. `band` is the decision-difficulty lever — and it reaches the install window
In `generate_decision_arithmetic`, the threshold `T` is placed within `offset ≤ q/band` of the
true quantity `Q` — so **higher `band` = threshold tighter to Q = the decision needs the exact
computed value = lower K** (digits only scale the compute, which the 4B handles; `band` scales the
*decision margin*). band was fixed at 5 throughout §1–§2, which is why K stayed high.

digits×band → base-K (greedy, n=48):
| digits | band | K | m |
|---|---|---|---|
| 4 | 5 / 25 / 100 | 0.90 / 0.79 / 0.65 | 0.02 / 0.08 / 0.12 |
| 6 | 25 / 100 / 300 / 1000 | 0.85 / 0.56 / 0.56 / **0.35** | 0.12 / 0.31 / 0.31 / 0.31 |
| 8 | 100 | 0.60 | 0.27 |

- **band is the lever**: at digits=6, raising band 25→1000 drives K 0.85→**0.35**. Digit count alone
  (band 5, §2) floors at ~0.75.
- **`digits=6, band=1000` → K≈0.35, m≈0.31** lands in the install window (0.3–0.5) with strong base
  fail-open headroom (m 0.31) — the install-friendly substrate digits-alone never reached.
- Non-obvious shape: K *plateaus* (band 100=300=0.56) then *drops* at band 1000 — the margin must be
  very tight (offset ≤ ~0.1% of Q) before the 4B's decision accuracy breaks.

**Recommended install substrate for a valid E1 re-run:** A (install) = `digits=6, band=1000`
(K≈0.35 → strong install expected); A′ (harder clean successor) = `digits=8, band=1000` (more
compute at the same tight margin; its K is confirmed at the run's warm-start eval). This replaces
the vacuous digits=2 install in §1.

## Caveats
Greedy (not sampled); single seed; n_eval=48 (noise floor ~0.02); base model (not the
warm-start the install starts from). A coarse difficulty ordering.
