"""Smoke example: score a few completions against a live verified platform.

Exercises the full reward path — parser -> verifier -> shaper — WITHOUT any RL
training, so it doubles as an end-to-end sanity check. Requires a real platform
and a key in the environment (``AMBERTRACE_API_KEY`` / ``AMBERTRACE_PLATFORM_ID``).

    python examples/score_completions.py           # uses AMBERTRACE_PLATFORM_ID
    python examples/score_completions.py 9         # or pass a platform id

This is the only example that touches the network; the test suite does not.
"""

from __future__ import annotations

import os
import sys

from ambertrace_rlvr import AmberVerifier, DefaultRewardShaper, JSONBlockParser, VerifiableDomain


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


# Two candidate completions for a strong loan application (should permit).
COMPLETIONS = [
    # well-formed, correct facts
    '<reasoning>strong profile</reasoning><decision>'
    '{"classification": "permit", "facts": {"applicant_age": 67, "annual_income": 135403,'
    ' "credit_score": 818, "credit_history_months": 4, "debt_to_income_ratio": 0.26,'
    ' "employment_status": "retired", "employment_years": 0.4, "loan_type": "unsecured",'
    ' "loan_amount": 54887, "collateral_value": 0, "loan_purpose": "personal",'
    ' "existing_loans": 0}}</decision>',
    # malformed — no decision block (should floor)
    "I think this loan should be approved.",
]


def main() -> None:
    _load_dotenv()
    platform_id = int(sys.argv[1]) if len(sys.argv) > 1 else None

    domain = VerifiableDomain.from_env(
        platform_id,
        parser=JSONBlockParser(answer_key="classification", facts_key="facts",
                               query_template="Assess this loan application: {facts}"),
    )
    verifier = AmberVerifier(domain=domain, shaper=DefaultRewardShaper())
    reward_fn = verifier.as_reward_function()

    prompts = ["Assess the loan."] * len(COMPLETIONS)
    rewards = reward_fn(prompts, COMPLETIONS, [{"gold": "permit"}, {"gold": "permit"}])

    for i, (completion, reward) in enumerate(zip(COMPLETIONS, rewards)):
        preview = completion[:60].replace("\n", " ")
        print(f"[{i}] reward={reward:+.3f}  | {preview}...")


if __name__ == "__main__":
    main()
