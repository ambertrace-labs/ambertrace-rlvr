# Verifiable Rewards Beyond Maths and Code

*Why trustworthy models need a verifiable reward, and why the domains that most need one have gone without.*

**Ambertrace Labs • 2026 • Research • Overseen by Peter Chatwell, Founder/CEO**

> **Authorship & oversight.** Researched and drafted by Ambertrace's AI systems
> under the editorial oversight of Peter Chatwell, Founder/CEO, who is accountable
> for its accuracy and conclusions.

A model behaves the way it was rewarded to behave. If the reward that shaped it was
a *learned preference*, RLAIF in other words, then the
policy learns to produce answers that look good to that AI judge. It's rightly in the news about how this has caused alignment issues in OpenAI and Anthropic models in recent months. The author wrote about alignment issues with Qwen3 coder last year. A
**verifiable** reward inverts this. The policy is paid only when its output is
*checked correct* against ground truth. The
domains that have gone without such a check are exactly the ones where being wrong
may have the largest cost, as it is easily unnoticed. This note is about supplying that check, and about why the machinery
for it is open source.

## SECTION 01: A Model Is Only As Aligned As Its Reward

Reinforcement learning from human feedback made models helpful by rewarding them
against a learned model of human preference. That was the right tool for open-ended
conversation, and the wrong one for a decision that has to stand up in an audit. A
preference model rewards *the appearance of a good answer*; a policy trained against
it becomes fluent, confident, and, where it matters, unaccountable. "The model
said so" is not a reason a regulator, a clinician, or a credit committee can accept.

The recent step-change in reasoning models came from replacing that learned judge
with an automatic one. **RLVR** (Reinforcement Learning from Verifiable Rewards)
pays the policy only when its output is *verified* correct. In mathematics the answer
either matches or it does not; in code the tests either pass or they do not. The
reward is ground truth, and no amount of fluency can flatter its way to it.

> A verifiable reward cannot be charmed. It is either correct or not.

## SECTION 02: The Domains That Lack a Verifier

Everything turns on the word *verifiable*. Mathematics and code arrive with cheap,
ready-made verifiers. Almost none of the
decisions a business actually needs a model to make do. "Is this applicant eligible
for the grant?" has a correct answer, but that answer lives in a **rulebook**, not in
a unit test. The rules can be complex and in some cases open to interpretation.

So these domains which are rule-governed and arguably run the middle of the economy have so far been left out of the RLVR revolution. Where they were
post-trained at all, it was against the same learned preference models RLVR was
invented to escape. The frontier learned to reason toward *checkable* answers; the
regulated world kept optimising toward *persuasive* ones.

**Closing that gap is the reason this project exists.** A verifier for rule-governed
decisions is the missing piece that lets RLVR, and the trust that comes with it,
reach the domains that need it most.

## SECTION 03: AmberTrace as the Verifier

[AmberTrace](https://ambertrace.ai) produces a machine-checkable proof for every
decision. You describe your rules in plain English and hand it a features-only
dataset. It learns the rulebook *unsupervised*, no labels required, and it builds a
verified platform. Every query is then answered by an independent, fail-closed
**kernel** that re-derives and certifies the decision, returning an **Amber Report**:
a `proof_checked` certificate, the full symbolic trace of which rules fired and why,
the facts the gate accepted or rejected, and a fused neural-plus-symbolic confidence.

This is the neurosymbolic, glass-box idea taken to its useful end. The certificate is
not an explanation generated *after* the network has decided, to justify the outcome;
it is the decision, re-derived from the rules inside a trusted kernel. The logically
sound step and the epistemic inputs stay separate and separately inspectable. The
guarantee is therefore auditable.

## SECTION 04: What the Reward Actually Measures

`ambertrace-rlvr` turns that certificate into a scalar. A completion earns reward only
when it produces a **valid proof certificate** for the domain. That is a hard, auditable,
hallucination-resistant signal. `DefaultRewardShaper` reads `proof_checked`, the
rejected-fact fraction, and the symbolic trace, and combines them into a dense,
bounded reward, each component clamped before weighting so that a certified,
fact-grounded decision scores high and an uncertified or fact-rejected one *can never
out-score it*.

That last property is what makes the reward hard to hack. A policy cannot win by
smuggling an unsupported fact past the gate or by producing a confident-sounding but
uncertified decision; the shaping is structured so the certificate, not the prose,
carries the credit. The reward is only as trustworthy as a formal proof, which is
the standard a regulated domain has always needed and never had from a learned judge.

> The answer either certifies against the rules, or it does not. The same three-word
> test now applies to every domain, not only the ones that ship with a test harness.

## SECTION 05: Open by Design

`ambertrace-rlvr` is MIT-licensed and public. The verification *platform* is a
product; the **bridge from a proof certificate to an RL reward** is infrastructure,
and infrastructure for trustworthy AI should be inspectable by the people asked to
trust it. Three reasons the work is done in the open:

- **Transparency compounds trust.** A verifier you cannot read is just another
  authority to take on faith: the exact thing we are removing. The reward path
  (parser → verifier → shaper), the anti-reward-hacking provenance checks, and the
  evaluation lane are all here to be audited, not merely described.
- **Bring your own domain.** The point is that *your* rulebook, in your regulated
  field, becomes trainable. That only works if the surrounding machinery is open
  enough to adapt: to your trainer (TRL/GRPO, veRL, more to come), your data, your
  rules.
- **Collaboration beats a walled garden.** Verifiable rewards for rule-governed
  domains is a research frontier, not a finished product. We would rather advance it
  with the community (issues, PRs, shared benchmarks) than behind a wall. The
  alignment research in this repo (see [*Measuring Misalignment as Deviation
  From the Provable*](alignment-matrix.md)) is published for the same reason: a result you can
  reproduce is the only kind worth publishing.

## SECTION 06: Where the Guarantee Stops

Two, stated plainly.

First, a verifiable reward is only as good as the verifier. AmberTrace's guarantee is
that a decision is *provably consistent with its stated policy*, it does not claim
the policy itself is wise. That is a feature: the rulebook is written in plain English
and open to inspection and challenge, rather than baked opaquely into weights. The
proof hands you both halves to check: the rules, and the reasoning over them.

Second, this is a bridge, not the whole bridge yet. The reward path, the eval lane,
and a demonstrated end-to-end GRPO run are in; the cross-domain training demo and
further trainer adapters are on the [roadmap](../../ROADMAP.md). We would rather ship
a checkable core and say so than overclaim a finished one.

## For the Record

- **Companion piece (research).** [*Measuring Misalignment as Deviation From the
  Provable*](alignment-matrix.md): the alignment matrix that uses this same certificate
  as an oracle-as-judge, measuring the *safety direction* of open-weight models'
  errors.
- **Companion piece (perspective).** Peter Chatwell, *Traders (AI Labs) vs. Risk
  (Alignment)*: on why measurable, independent alignment needs teeth, and why
  "alignment as deviation from provable outputs" is a metric worth building toward.
- **Reproduce.** The offline test suite and the verification-overhead benchmark run
  with no AmberTrace account (built-in `FakeVerifier` + recorded payloads). Authoring
  a platform and training against a live one need an API key from
  [ambertrace.ai](https://ambertrace.ai). See the [README](../../README.md).

---

*AmberTrace AI builds proof-carrying decision and policy infrastructure: write the
rules in plain English, and get a machine-checked proof for every decision an AI
system makes. Learn more at [ambertrace.ai](https://ambertrace.ai).*

*© 2026 Ambertrace Labs Ltd.*
