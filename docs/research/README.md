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
  Scoring Qwen3.6-27B across a single-publisher quantisation ladder: the safety
  direction is robust down to 2-bit (fail-open stays flat), and a smaller 120-item
  slice that suggested otherwise did not replicate. Method notes in
  [`../QUANT_ALIGNMENT.md`](../QUANT_ALIGNMENT.md).
