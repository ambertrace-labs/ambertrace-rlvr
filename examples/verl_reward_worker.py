"""Wire an AmberTrace verified reward into a veRL run — the multi-node counterpart
to `grant_eligibility_grpo.py` (which uses TRL/GRPO).

The reward is identical to every other example: AmberTrace's proof certificate,
turned into a scalar by `DefaultRewardShaper`. Only the *plumbing* differs — veRL
drives reward through a per-sample `compute_score(data_source, solution_str,
ground_truth, extra_info=None)` custom reward function, so we adapt our batched
reward function to that contract with `as_verl_reward_function` (and, optionally,
wrap it in veRL's reward manager with `build_verl_reward_worker`).

    python examples/verl_reward_worker.py --dry-run   # offline: adapter wiring, no verl/GPU
    python examples/verl_reward_worker.py             # builds the veRL reward worker (needs verl)

Config: configs/grant_eligibility.yaml (reuses the demo platform). Needs
AMBERTRACE_API_KEY (scoped, platform-only) in the env for the real path.

Multi-node caveats (see also the `integrations.verl` module docstring):
  * Every rollout rank runs this reward independently, so each node needs network
    reach to the platform and a valid AMBERTRACE_API_KEY. Aggregate QPS scales with
    (num_nodes × rollouts_per_step) — size the platform rate limit accordingly.
  * The verifier is fail-closed with a circuit breaker: a briefly unavailable
    platform floors the reward rather than stalling the cluster.
  * The content-addressed cache is per-process (not shared across ranks).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ambertrace_rlvr import load_run_config
from ambertrace_rlvr.integrations.verl import (
    as_verl_reward_function,
    build_verl_reward_worker,
)

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "grant_eligibility.yaml"

# A well-formed sample completion for the offline dry-run (a strong permit case).
_SAMPLE_COMPLETION = (
    "<reasoning>Adult, low income, resident, no active grant — all criteria met.</reasoning>"
    '<decision>{"classification": "permit", "facts": {"age": 40, "annual_income": 25000, '
    '"resident": true, "has_active_grant": false}}</decision>'
)


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def dry_run() -> None:
    """Exercise the veRL adapter offline — no verl, no GPU, no network.

    Shows that the per-sample `compute_score` veRL calls is a working AmberTrace
    reward: a well-formed certified completion out-scores a malformed one."""
    from ambertrace_rlvr.testing import FakeVerifier

    fake = FakeVerifier()
    compute_score = as_verl_reward_function(fake.as_reward_function())

    # veRL calls compute_score(data_source, solution_str, ground_truth, extra_info).
    good = compute_score("grant_eligibility", _SAMPLE_COMPLETION, "permit")
    bad = compute_score("grant_eligibility", "no decision block here", "permit")
    print(f"dry-run compute_score: well-formed={good:.3f}  malformed={bad:.3f}")
    assert good > bad, "well-formed permit should out-score malformed floor"
    print("OK — veRL reward adapter is sound (well-formed > malformed floor).")


def build_worker() -> object:
    """Build the veRL reward worker against the live platform (needs the platform
    reachable + `pip install verl`). Returns veRL's reward manager, ready to hand
    to the trainer."""
    _load_dotenv()
    run = load_run_config(CONFIG)
    reward_fn = run.reward_function()
    print(f"wiring AmberTrace reward (platform {run.domain.platform_id}) into veRL — "
          "reward = proof certificate + shaped components")
    # tokenizer=None here for illustration; pass your trainer's tokenizer in a real run.
    worker = build_verl_reward_worker(reward_fn, tokenizer=None, num_examine=1)
    print(f"built veRL reward worker: {type(worker).__name__}")
    return worker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="offline adapter-wiring check (no verl/GPU/network)")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
    else:
        build_worker()


if __name__ == "__main__":
    main()
