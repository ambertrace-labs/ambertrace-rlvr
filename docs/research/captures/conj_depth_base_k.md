# Capture — conjunction-depth base-K (d2–d8): conj depth does not lower 4B's K

**Data capture, not a finding.** Base Qwen3-4B, greedy, n=108/rung, single run. Extends the
per-rung base-K capture ([ladder_base_k.md](ladder_base_k.md)) to the **type-B (conjunction)**
ladder at deeper rungs. Raw: [`conj_d8_base_k.json`](conj_d8_base_k.json).

## Method
`examples/author_ladder.py --type conj --rungs 7 --variants 6 --items-per-domain 18` (`approve_deny`;
`deny` only if **all** `d` ratio-conjuncts hold; max depth extended 4→8). Base model (no adapter),
greedy, scored with `evaluate_ambertrace` (`always_least_restrictive`). K = true accuracy; K̂ = permit
rate (0.5 ≈ honest base rate); m = fail-open rate.

## Result
| rung | d2 | d3 | d4 | d5 | d6 | d7 | d8 |
|---|---|---|---|---|---|---|---|
| **K** | 0.694 | 0.630 | 0.676 | 0.639 | 0.722 | 0.657 | 0.639 |
| **K̂** | 0.435 | 0.389 | 0.398 | 0.417 | 0.370 | 0.398 | 0.491 |
| **m** | 0.120 | 0.130 | 0.111 | 0.139 | 0.074 | 0.120 | 0.130 |

## Observations (to be confirmed, not yet published)
1. **Conjunction depth is not a K lever.** K is flat and non-monotone (~0.63–0.72) across d2–d8 —
   adding conjuncts (2→8) does not lower one-shot K. Total d2→d8 change is 0.055 and within noise /
   non-monotone (d6 > d2). This mirrors the OR-of-ratios result (ladder_base_k.md): **difficulty-by-
   more-conditions, whether disjunction or conjunction, does not push 4B's K into the low range.**
2. **K̂ < 0.5** everywhere — on `conj` the base model is mildly *over-cautious* (over-denies), the
   opposite of a fail-open prior; m stays ~0.07–0.14.
3. **Implication.** Both *buildable* authored difficulty types (type-ratio/OR, type-conj/AND) floor K
   at ~0.6–0.7. The only lever that genuinely raises computational load enough to lower 4B's K is a
   **combined-quantity / multi-operand** predicate (type-A `sum`, and the proposed `nested_ratio` /
   `cumulative_threshold`, [#217](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/217)) —
   and `sum` is currently blocked on the authoring platform's verified-build gate for compound
   predicates. So the **authored low-K install substrate is blocked pending that platform support**;
   difficulty-by-conditions cannot substitute for it.

## Caveats
Greedy (not sampled); single run; n=108/rung; one model (Qwen3-4B). A coarse difficulty ordering,
not bit-reproducible.
