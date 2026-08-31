# Ambertrace research

Open, reproducible research on verifiable rewards and model alignment. These pieces
are written to be published (ambertracelabs.com/research) and to stand on numbers
you can regenerate from this repo.

- **[Verifiable Rewards Beyond Maths and Code](why-verifiable-rewards.md).**
  What this project is for: why trustworthy models need a *checkable* reward, how
  AmberTrace supplies the missing verifier for rule-governed domains, and why the
  bridge is open source.

- **[Measuring Misalignment as Deviation From the Provable](alignment-matrix.md).**
  An open-weight alignment matrix scoring current frontier open models against a
  proof-certified oracle, reporting the *safety direction* of their errors rather
  than raw accuracy. The live results table is
  [`../ALIGNMENT_MATRIX.md`](../ALIGNMENT_MATRIX.md).

- **[Quantisation and the Safety Direction of Decisions](quantisation-safety-direction.md).**
  Scoring Qwen3.6-27B across a single-publisher quantisation ladder: fail-open is
  concentrated in one reasoning type (ratio rules, ~16% vs ≤5% elsewhere) at every
  precision; the net safety direction is precision-insensitive (signed-bias R²=0.01)
  while accuracy declines mildly; at 2-bit the failures redistribute rather than grow.
  Method notes in [`../QUANT_ALIGNMENT.md`](../QUANT_ALIGNMENT.md).

- **[Faithfulness of Stated Reasoning Under Verifiable-Reward RL](faithfulness-under-rlvr.md).**
  Does RL against a proof-certified reward erode the faithfulness of a model's chain
  of thought? OLMo-3-7B-Think-SFT trained on air-track triage with the consistency
  weight at zero (measured, never optimised). Pilot: no faithfulness erosion; mild
  positive drift on held-out probes; OOD arm shows a drift toward over-caution in
  untrained domains. Main run in progress. Living draft.
