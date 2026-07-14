"""Smoke example: score a few completions against a live verified platform.

Exercises the full reward path — parser -> verifier -> shaper — WITHOUT any RL
training, so it doubles as an end-to-end sanity check. The run is described
entirely by ``configs/loan_example.yaml`` (loaded via ``load_run_config``);
only the API key comes from the environment (``AMBERTRACE_API_KEY``).

    python examples/score_completions.py                       # configs/loan_example.yaml
    python examples/score_completions.py configs/loan_example.yaml
    python examples/score_completions.py configs/loan_example.yaml 9   # override platform id

This is the only example that touches the network; the test suite does not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ambertrace_rlvr import load_run_config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "loan_example.yaml"


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
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG

    run = load_run_config(config_path)
    if len(sys.argv) > 2:  # optional platform-id override
        run.domain.platform_id = int(sys.argv[2])
    reward_fn = run.reward_function()

    prompts = ["Assess the loan."] * len(COMPLETIONS)
    rewards = reward_fn(prompts, COMPLETIONS, [{"gold": "permit"}, {"gold": "permit"}])

    for i, (completion, reward) in enumerate(zip(COMPLETIONS, rewards)):
        preview = completion[:60].replace("\n", " ")
        print(f"[{i}] reward={reward:+.3f}  | {preview}...")


if __name__ == "__main__":
    main()
