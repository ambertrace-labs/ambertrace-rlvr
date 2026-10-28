# AmberTrace® research

Open, reproducible research on verifiable rewards and model alignment. These pieces
are written to be published (ambertracelabs.com/research) and to stand on numbers
you can regenerate from this repo.

**Writing or illustrating a piece?** Follow the [Research Style Guide](STYLE_GUIDE.md)
— figures over tables, committed SVG (no Mermaid), honest axes, and every figure
regenerating from the captures.

**On Hugging Face.** This corpus is [**published as HF Blog Articles**](https://huggingface.co/AmberTraceLabs)
under `AmberTraceLabs` — the last surface of the [distribution lane](../HUGGINGFACE.md)
([#109](https://github.com/ambertrace-labs/ambertrace-rlvr/issues/109)). Each links back
to the [datasets](https://huggingface.co/AmberTraceLabs), the
[leaderboard Space](https://huggingface.co/spaces/AmberTraceLabs/certified-alignment-matrix),
and the [faithfulness Model](https://huggingface.co/AmberTraceLabs/olmo-3-7b-air-track-faithfulness).
Regenerate the paste-ready copies with `examples/build_hf_articles.py`; publish steps in
[hf-articles/PUBLISHING.md](../hf-articles/PUBLISHING.md).

- **[Verifiable Rewards Beyond Maths and Code](why-verifiable-rewards.md).**
  What this project is for: why trustworthy models need a *checkable* reward, how
  AmberTrace supplies the missing verifier for rule-governed domains, and why the
  bridge is open source.
  [Read on Hugging Face](https://huggingface.co/blog/AmberTraceLabs/verifiable-rewards-beyond-maths-and-code).

- **[Measuring Misalignment as Deviation From the Provable](alignment-matrix.md).**
  An open-weight alignment matrix scoring current frontier open models against a
  proof-certified oracle, reporting the *safety direction* of their errors rather
  than raw accuracy. The live results table is
  [`../ALIGNMENT_MATRIX.md`](../ALIGNMENT_MATRIX.md).
  [Read on Hugging Face](https://huggingface.co/blog/AmberTraceLabs/the-direction-of-error-in-open-weight-decision-mod).

- **[Quantisation and the Safety Direction of Decisions](quantisation-safety-direction.md).**
  Scoring Qwen3.6-27B across a single-publisher quantisation ladder: fail-open is
  concentrated in one reasoning type (ratio rules, ~16% vs ≤5% elsewhere) at every
  precision; the net safety direction is precision-insensitive (signed-bias R²=0.01)
  while accuracy declines mildly; at 2-bit the failures redistribute rather than grow.
  Method notes in [`../QUANT_ALIGNMENT.md`](../QUANT_ALIGNMENT.md).
  [Read on Hugging Face](https://huggingface.co/blog/AmberTraceLabs/quantisation-and-the-safety-direction-of-decisions).

- **[Reward-Hacking Inheritance Across Warm-Start Generations](reward-hacking-lineage.md).**
  An apparatus for measuring whether an unaddressed reward-hacking load compounds,
  stalls genuine progress, or can be scrubbed when a lineage is warm-started across
  multiple RL generations. Self-contained exploitable-verifier math environment
  (R_true/R_hack, four hack kinds, lambda dial), 3-arm lineage
  (pure/poisoned/scrubbed), and a Pythia + Qwen2.5 scale ladder. Four induction
  attempts failed to produce spontaneous reward hacking on a capable aligned model
  under GRPO; a warm-start mixed terse+CoT SFT breaks the resistance on demand.
  The aligned-vs-misaligned race (Qwen2.5-0.5B-Instruct, GSM8K,
  `short_circuit_brevity`, lambda=0.75) is confirmed across 3 seeds: the aligned
  arm advances (mean end K: 0.271) while the misaligned arm collapses (mean end
  K: 0.026); mean gap 0.245 (approximately 16x the noise floor), Q1 = YES in
  all seeds.

- **[Faithfulness of Stated Reasoning Under Verifiable-Reward RL](faithfulness-under-rlvr.md).**
  Does RL against a proof-certified reward erode the faithfulness of a model's chain
  of thought? OLMo-3-7B-Think-SFT trained on air-track triage (250 iterations, QLoRA
  8-bit, three stop/resume segments) with the consistency weight at zero (measured,
  never optimised). Headline: faithfulness recall drifts down mildly (0.238 to 0.182)
  while consistency (precision) rises (0.044 to 0.099) — the model becomes more
  selective, not more confabulatory. No reward-correlated confabulation (corr +0.12),
  no concealment, verifier-awareness near zero. OOD fail-open goes to 0.000 from step
  60 durably; signed bias locks mildly negative (over-caution); OOD accuracy rises to
  0.983.
  [Read on Hugging Face](https://huggingface.co/blog/AmberTraceLabs/faithfulness-of-stated-reasoning-under-verifiable).

## Working notes — decision-domain transfer (in progress, not yet published)

These pieces extend the lineage work from maths into *decision* domains. They are
active working notes rather than published articles — cite with their status in mind.

- **[The Misalignment-Transfer Conjecture](misalignment-transfer-conjecture.md).**
  The framing for whether in-domain reward-hack poisoning transfers out-of-domain,
  and the strength-vs-structure condition (strong-but-narrow is not sufficient for
  transfer — shared structure is the missing ingredient).

- **[OOD Decision Transfer at 4B](ood-decision-transfer-4b.md).**
  The GRPO dose-response result: near-OOD transfer (same compute→compare→decide
  logic, new surface) is real and hack-specific (fail-open β≈+0.37, lag≈2.5, 2 of 3
  seeds); far-OOD (`decision_eval_v1`, different structures) is a genuine null.

- **[Decision-Hack RL-Installability](decision-hack-rl-installability.md).**
  Which fail-open hacks GRPO can install: dense/cued hacks install, the sparse
  conditional `severity_fail_open` is RL-resistant — plus a greedy-K̂ caveat
  (report the *sampled* hack-rate under high-entropy training).

- **[Reward-Hack Induction at 4B](4b-induction-results.md).**
  Inducibility at 4B as scale × domain-saturation × hack-type.

- **⚠️ [Within-Family Decision Contagion](within-family-decision-contagion.md) — RETRACTED.**
  Under correction: the underlying corpus declared verb restrictiveness out-of-band
  instead of via an AmberTrace decision table, so its fail-open metrics sit on an
  unvalidated frame. Do not cite pending a corpus rebuild.

- **[Decision-Corpus Authoring Correction](decision-corpus-authoring-correction.md).**
  The methodology defect behind the retraction, and the corrected decision-table
  authoring pipeline (SDK 2.10).
