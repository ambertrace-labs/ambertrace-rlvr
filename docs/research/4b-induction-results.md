# 4B reward-hack induction: scale × saturation × hack-type

**Scope:** #158 / #161. **Question:** can a reward-hack that installs trivially at 0.5B be
seeded into a capable 4B via warm-start SFT — and what does it depend on?

All numbers: `Qwen/Qwen3-4B`, warm-start LoRA SFT (rank 8, lr 1e-5), held-out eval at
`lam=0`, `n_eval=16` (noise floor `1/16 = 0.0625`), single seed. Brevity is seeded by
`terse_fraction` and scored by the `brevity` hack (completions collapse → short); answer_copy
is seeded by `shortcut_fraction` and scored by the `answer_copy` hack (echo the largest operand).
*(Measurement note: the calibration eval now scores the hack the run seeds — `terse→brevity`,
`shortcut→answer_copy` — recorded as `eval_hack_kind` in each summary. Earlier terse runs had
scored the default `answer_copy`, so for those the brevity signal is `mean_completion_words`.)*

## The domain lever: hard-arithmetic de-saturates the 4B

GSM8K saturates the 4B (base K ≈ 0.75, almost certainly in-training-set). A synthetic
large-operand corpus (`hard_arith`, random operands ⇒ un-memorized, difficulty = `digits`)
pulls it off the ceiling:

| corpus | base K | base words |
|---|---:|---:|
| GSM8K | 0.75 | — |
| hard-arith d=2 | **0.562** | 70 |
| hard-arith d=3 | 0.312 | 119 |
| hard-arith d=4 | 0.219 | 147 |

d=2 (K≈0.56) is the non-saturated regime used below.

## Brevity (output-channel hack): saturation decides

| domain | dose | words after | K̂(brevity) | installed? |
|---|---|---:|---:|:--|
| GSM8K (sat., K≈0.75) | tf 0.4–0.9, 60 steps | ~155 | — (at floor) | **No** |
| GSM8K | tf 1.0, **200 steps** | 146 | — (at floor) | **No** |
| hard-arith d=2 (K≈0.56) | tf 1.0, 200 steps | **1.0** | **0.995** (m=0.56) | **Yes** |

On saturated GSM8K the brevity seed **refuses** even at 1.0 / 200 steps (completions never
collapse). De-saturate the domain and the *same* seed **installs completely** — words 70→1,
K̂(brevity)=0.995, with a large misalignment load m=0.56 (short-and-wrong undeserved passes).
So for brevity, **saturation/entrenchment — not scale per se — was the blocker** at 4B.

## answer_copy (reasoning-shortcut): resists regardless

| domain | dose | K̂(answer_copy) | words | installed? |
|---|---|---:|---:|:--|
| GSM8K | sf 0.6–1.0, 60 steps | ~floor | ~160 | **No** |
| hard-arith d=2 | sf 1.0, 200 steps | 0.0 | 60 | **No** |

answer_copy **refuses on both** domains — even de-saturated d=2 where brevity installs. (sf=0.8
NaN'd at lr 1e-5; the 0.6/1.0 points are stable and both at floor.)

## The decomposition

Induction at 4B depends on three separable factors:

1. **Scale** — at 0.5B both hacks install (brevity trivially; answer_copy with heavy seeding,
   K̂=0.906). At 4B neither installs on GSM8K.
2. **Saturation** — de-saturating (hard-arith d=2) flips **brevity** from refuse → install. This
   is the dominant factor for the output-channel hack.
3. **Hack type** — even de-saturated, **answer_copy resists where brevity installs**: a clean
   hack-type dissociation at 4B (output-channel inducible, reasoning-shortcut not).

## Caveats

- `n_eval=16` (floor 0.0625), **single seed** — treat ±1–2 items as noise.
- **LoRA capacity** (rank 8 / lr 1e-5) is not separated from scale: a proportionally weaker lever
  on a 4B could contribute to answer_copy's resistance. (Partly moot for brevity, which *did*
  install at d=2 with the same LoRA.)
- Warm-start SFT seeds; it does not test whether **GRPO** could install a hack the seed cannot.

## Follow-ons

- **OOD transfer:** the d=2 brevity adapter shows **zero transfer** to decision tasks (the 4B is
  capable at decisions — accuracy 0.827, parse 1.0 — so this is a real null, not a floor artifact).
  See `ood-decision-transfer-4b.md`.
- **Why / what next:** `misalignment-transfer-conjecture.md` — strength alone (K̂=0.995) didn't
  transfer, so the proposed next test is a *calc→threshold→decision* domain with a directional
  hack, to see whether structural relatedness is what transfer requires.
