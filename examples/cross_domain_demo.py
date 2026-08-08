"""Swap-the-rule-set demo: two domains, one code path (design principle 4).

Cross-domain generalisation without forks. The grant-eligibility and ACMG
variant-classification domains are scored through the *same* function,
:func:`score_domain`. Between the two runs, only two things change:

  1. the **config** (``configs/grant_eligibility.yaml`` vs ``configs/acmg.yaml``),
     which selects the platform, the reward weights, and the parser + its
     ``query_template``; and
  2. the **recorded platform payloads** that stand in for a live AmberTrace
     platform offline (the "rule set").

Everything else — loading the config, building the reward function from the
config's parser + shaper, and computing per-completion rewards — is identical,
domain-agnostic code. There is no per-domain branch anywhere in the scoring
path. That is the whole point: to add a third domain you write a new YAML (and,
if its completions look different, a parser), never a new code path.

What stays constant across domains (the "one code path"):
  * ``load_run_config`` → a fully-wired ``RunConfig`` (parser + shaper + floor).
  * ``FakeVerifier(...).as_reward_function()`` → the batch reward function
    (parser → verifier → ``DefaultRewardShaper``), exactly as real training uses.
  * The reward contract: parse → verify → shape → clip; unparseable completions
    fail closed to the floor and never out-score a certified one.

Offline and network-free: a :class:`FakeVerifier` replays recorded reports in
place of the platform (no API key, no network), mirroring
``examples/acmg_variant_grpo.py --dry-run``. On a real run the only change is
swapping ``FakeVerifier`` for the config-wired ``AmberVerifier`` (i.e.
``run.reward_function()``) — the scoring code below is unchanged.

    python examples/cross_domain_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ambertrace_rlvr import AmberReport, ParsedCompletion, load_run_config
from ambertrace_rlvr.testing import FakeVerifier, make_report

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"


@dataclass
class Sample:
    """One model completion to score, with its gold label and a note."""

    completion: str
    gold: str
    note: str


@dataclass
class DomainDemo:
    """A domain = a config + sample completions + recorded platform reports.

    ``recorded`` maps a proposed answer (lower-cased) to the report the platform
    would certify for it — the domain's "rule set", replayed offline. It is data,
    not code: the scoring path in :func:`score_domain` never looks at it.
    """

    name: str
    config: Path
    prompt: str
    samples: list[Sample]
    recorded: dict[str, AmberReport] = field(default_factory=dict)

    def report_for(self, parsed: ParsedCompletion) -> AmberReport:
        key = str(parsed.proposed_answer).strip().lower()
        # Fall back to a plain certified report echoing the answer (FakeVerifier's
        # own default) if the domain recorded nothing for this answer.
        return self.recorded.get(key) or make_report(
            proof_checked=True, decision=parsed.proposed_answer
        )


def score_domain(domain: DomainDemo) -> list[float]:
    """The ONE code path. Identical for every domain — only ``domain`` varies.

    Load the config, build the reward function from the config's parser + shaper,
    and score the completions. Swap ``FakeVerifier`` for ``run.reward_function()``
    to run the same logic against the live platform.
    """
    run = load_run_config(domain.config)
    verifier = FakeVerifier(
        parser=run.domain.parser,     # parser comes from the config
        shaper=run.shaper,            # shaper + weights come from the config
        floor=run.verifier.floor,
        report_fn=domain.report_for,  # replaces the live platform, offline
    )
    reward_fn = verifier.as_reward_function()

    prompts = [domain.prompt] * len(domain.samples)
    completions = [s.completion for s in domain.samples]
    metadata = [{"gold": s.gold} for s in domain.samples]
    return reward_fn(prompts, completions, metadata)


# --- Domain 1: grant eligibility ------------------------------------------
_GRANT_ELIGIBLE = (
    "<reasoning>Income under the cap and residency verified — eligible.</reasoning>"
    '<decision>{"classification": "eligible", "facts": {'
    '"annual_income": 28000, "residency_verified": true, "prior_grant": false}}</decision>'
)
_GRANT_INELIGIBLE = (
    "<reasoning>Income over the cap.</reasoning>"
    '<decision>{"classification": "ineligible", "facts": {'
    '"annual_income": 96000, "residency_verified": true, "prior_grant": false}}</decision>'
)

GRANT = DomainDemo(
    name="grant_eligibility",
    config=CONFIGS / "grant_eligibility.yaml",
    prompt="Assess this grant application.",
    samples=[
        Sample(_GRANT_ELIGIBLE, gold="eligible", note="well-formed, correct → high"),
        Sample(_GRANT_INELIGIBLE, gold="eligible", note="certified but wrong verdict → mid"),
        Sample("I think they qualify.", gold="eligible", note="no decision block → floor"),
    ],
    recorded={
        "eligible": make_report(
            proof_checked=True, decision="eligible", accepted=3,
            rules=[("income_below_cap", True, True),
                   ("residency_verified", True, True),
                   ("no_prior_grant_conflict", True, False)],
        ),
        "ineligible": make_report(
            proof_checked=True, decision="ineligible", accepted=3,
            rules=[("income_below_cap", False, True),
                   ("residency_verified", True, True)],
        ),
    },
)

# --- Domain 2: ACMG variant classification --------------------------------
_ACMG_PATHOGENIC = (
    "<reasoning>LoF in a disease gene (PVS1), no benign evidence — pathogenic.</reasoning>"
    '<decision>{"classification": "pathogenic", "facts": {'
    '"null_variant_in_disease_gene": true, "common_in_population": false}}</decision>'
)
_ACMG_BENIGN = (
    "<reasoning>Common in the population — benign.</reasoning>"
    '<decision>{"classification": "benign", "facts": {'
    '"null_variant_in_disease_gene": false, "common_in_population": true}}</decision>'
)

ACMG = DomainDemo(
    name="acmg_variant",
    config=CONFIGS / "acmg.yaml",
    prompt="Classify this sequence variant.",
    samples=[
        Sample(_ACMG_PATHOGENIC, gold="pathogenic", note="well-formed, correct → high"),
        Sample(_ACMG_BENIGN, gold="pathogenic", note="certified but wrong verdict → mid"),
        Sample("Looks pathogenic to me.", gold="pathogenic", note="no decision block → floor"),
    ],
    recorded={
        "pathogenic": make_report(
            proof_checked=True, decision="pathogenic", accepted=2,
            rules=[("null_variant_in_disease_gene_PVS1", True, True),
                   ("not_common_in_population", True, False)],
        ),
        "benign": make_report(
            proof_checked=True, decision="benign", accepted=2,
            rules=[("null_variant_in_disease_gene_PVS1", False, True),
                   ("common_in_population_BA1", True, False)],
        ),
    },
)

DOMAINS = [GRANT, ACMG]


def main() -> None:
    print("swap-the-rule-set demo — two domains, one code path (score_domain)\n")
    scored = [(domain, score_domain(domain)) for domain in DOMAINS]
    for domain, rewards in scored:
        print(f"=== {domain.name}  (config: {domain.config.name}) ===")
        for sample, reward in zip(domain.samples, rewards):
            preview = sample.completion[:52].replace("\n", " ")
            print(f"  reward={reward:+.3f}  gold={sample.gold:<11} "
                  f"{sample.note}\n           | {preview}...")
        print()

    # What proves "one code path" is that score_domain is branch-free — it has
    # zero per-domain logic. The reward contract it must uphold, asserted PER
    # domain: a well-formed correct completion out-scores a certified-but-wrong
    # one, which out-scores a malformed completion floored to clip[0].
    for domain, rewards in scored:
        correct, wrong, floored = rewards
        assert correct > wrong > floored, (
            f"{domain.name}: expected correct > wrong > floor, got {rewards}"
        )

    # The two domains happen to land on the same numbers here — an illustrative
    # coincidence of the matched payloads (both give graded 1.0 / 0.5), NOT the
    # proof of a shared code path. Different payloads would shift the numbers
    # while the (unchanged) code path still upholds the ordering above.
    print("OK — both domains scored through the one branch-free code path.")
    for domain, rewards in scored:
        print(f"     {domain.name:<18} spread:", [f"{r:+.3f}" for r in rewards])


if __name__ == "__main__":
    main()
