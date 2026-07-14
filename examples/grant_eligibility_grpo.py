"""End-to-end GRPO training against the Grant Eligibility verified platform.

The reward is AmberTrace's proof certificate: the policy proposes a decision +
facts, the platform re-derives and certifies, and `DefaultRewardShaper` turns the
Amber Report into a scalar reward. Label-free — the reward comes from the
certificate, not from gold labels.

    python examples/grant_eligibility_grpo.py --dry-run   # offline: reward wiring only, no trl/GPU
    python examples/grant_eligibility_grpo.py             # real GRPO run (needs the [trl] extra)

Config: configs/grant_eligibility.yaml (platform, reward, training). Needs
AMBERTRACE_API_KEY (scoped, platform-only) in the env for the real run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ambertrace_rlvr import load_run_config

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "grant_eligibility.yaml"
TRAIN = REPO / "data" / "grant_eligibility_train.jsonl"
EVAL = REPO / "data" / "grant_eligibility_eval.jsonl"

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
    """Exercise the reward wiring offline — no trl, no GPU, no network."""
    from ambertrace_rlvr.integrations.trl import as_trl_reward_func
    from ambertrace_rlvr.testing import FakeVerifier

    run = load_run_config(CONFIG)
    # Swap in a FakeVerifier so we never hit the platform.
    fake = FakeVerifier(shaper=run.shaper, floor=run.verifier.floor)
    trl_reward = as_trl_reward_func(fake.as_reward_function())

    prompts = ["Assess the grant application."] * 2
    completions = [_SAMPLE_COMPLETION, "no decision block here"]
    rewards = trl_reward(prompts, completions)
    print("dry-run rewards:", rewards)
    assert rewards[0] > rewards[1], "well-formed permit should out-score malformed"
    print("OK — reward wiring is sound (well-formed > malformed floor).")


def train(*, max_steps: int | None = None, num_generations: int | None = None,
          max_completion_length: int = 320, learning_rate: float | None = None,
          beta: float = 0.04) -> dict:
    _load_dotenv()
    import os

    from datasets import load_dataset  # type: ignore
    from trl import GRPOConfig  # type: ignore

    from ambertrace_rlvr import build_run_report, write_run_report
    from ambertrace_rlvr.integrations.trl import build_grpo_trainer

    # macOS/MPS workaround: transformers 5.x materializes weights across a
    # ThreadPoolExecutor (GLOBAL_WORKERS=min(4, cpu_count)); concurrent Metal
    # tensor copies segfault under trl's model load. Force single-threaded
    # loading. (Harmless elsewhere; only affects load parallelism.)
    import transformers.core_model_loading as _cml
    _cml.GLOBAL_WORKERS = 1

    run = load_run_config(CONFIG)
    reward_fn = run.reward_function()
    tcfg = run.training
    assert tcfg is not None, "config is missing a [training] section"
    group = num_generations or tcfg.group_size
    steps = max_steps if max_steps is not None else int(tcfg.extra.get("max_steps", 20))
    lr = learning_rate if learning_rate is not None else tcfg.learning_rate

    # Live experiment tracking when a WANDB_API_KEY is present (community dashboards).
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", "ambertrace-rlvr")
        print("wandb: logging to project", os.environ["WANDB_PROJECT"])

    dataset = load_dataset("json", data_files=str(TRAIN), split="train")
    out_dir = REPO / "outputs" / "grant_eligibility_grpo"

    # Small, Mac/MPS-friendly defaults for a demo run.
    args = GRPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=group,
        num_generations=group,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        # KL anchor to the reference model — without it (beta=0) the policy
        # can drift off-format and collapse the reward to the floor.
        beta=beta,
        max_completion_length=max_completion_length,
        max_steps=steps,
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if use_wandb else [],
        run_name="grant-eligibility-grpo",
        bf16=False,
        fp16=False,
    )
    trainer = build_grpo_trainer(
        model=tcfg.model, reward_fn=reward_fn, dataset=dataset, config=args,
    )
    print(f"training {tcfg.model} for {steps} steps (group={group}, lr={lr}) against "
          f"platform {run.domain.platform_id} — reward = AmberTrace certificate")
    trainer.train()

    # Run report (spec §12): config snapshot + per-step reward curve, keys redacted.
    report = build_run_report(
        config=run.raw,
        log_history=trainer.state.log_history,
        versions=_versions(),
        extra={"model": tcfg.model, "platform_id": run.domain.platform_id,
               "group_size": group, "learning_rate": lr, "max_steps": steps},
    )
    path = write_run_report(out_dir / "run_report.json", report)
    s = report["summary"]
    print(f"\nrun report → {path}")
    print(f"reward: {s.get('first_reward')} → {s.get('last_reward')} "
          f"(Δ{s.get('delta'):+.3f} over {s.get('n_steps')} steps, "
          f"increased={s.get('increased')})")
    return report


def _versions() -> dict[str, str]:
    import importlib.metadata as md
    out = {}
    for pkg in ("ambertraceai", "trl", "transformers", "torch"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="offline reward-wiring check (no trl/GPU/network)")
    ap.add_argument("--max-steps", type=int, default=None, help="override training steps")
    ap.add_argument("--num-generations", type=int, default=None,
                    help="override GRPO group size")
    ap.add_argument("--max-completion-length", type=int, default=320)
    ap.add_argument("--learning-rate", type=float, default=None, help="override lr")
    ap.add_argument("--beta", type=float, default=0.04, help="KL penalty (stability)")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
    else:
        train(max_steps=args.max_steps, num_generations=args.num_generations,
              max_completion_length=args.max_completion_length,
              learning_rate=args.learning_rate, beta=args.beta)


if __name__ == "__main__":
    main()
