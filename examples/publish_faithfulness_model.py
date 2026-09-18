"""Publish the faithfulness-experiment LoRA adapter as a Hugging Face Model (#108).

The main run of the faithfulness-under-RLVR experiment (#95): ``allenai/OLMo-3-7B-
Think-SFT`` trained with GRPO (MLX, LoRA-on-8-bit) against the AmberTrace-certified
air-track triage reward, 250 iterations over three stop/resume segments. This script
assembles a loadable MLX-LoRA adapter repo — the final step-250 adapter, the probed
checkpoint ladder, a synthesised ``adapter_config.json`` (none was saved at train
time), the captured learning curve, and a model card — and uploads it to
``AmberTraceLabs/olmo-3-7b-air-track-faithfulness``.

**What ships and what doesn't.** The adapter is model weights, not the certificate,
so AT = gold is not at stake — but the guard still runs over every JSON shipped
(learning curve etc.) to assert no certified-answer column leaks. The learning curve
is *aggregate probe metrics only* (no per-item oracle labels, no chain-of-thought).

Sourced from the gitignored ``outputs/faithfulness_run2/`` (the run's primary
artifacts) and ``outputs/{probe,ood_probe}_runs_main/``. Safe by default — prints
the plan and uploads **nothing** unless ``--push`` is given. **Auth gotcha:** a
read-only ``HF_TOKEN`` / ``HUGGINGFACEHUB_API_TOKEN`` env var overrides
``hf auth login`` and blocks org pushes — unset them
(``env -u HF_TOKEN -u HUGGINGFACEHUB_API_TOKEN python ... --push``).

Run:  ``python examples/publish_faithfulness_model.py``          # dry-run: print the plan
      ``python examples/publish_faithfulness_model.py --push``   # assemble + upload
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "outputs" / "faithfulness_run2"
PROBE_IN = REPO / "outputs" / "probe_runs_main" / "summary.jsonl"
PROBE_OOD = REPO / "outputs" / "ood_probe_runs_main" / "summary.jsonl"
DEFAULT_OUT = REPO / "dist" / "hf"

NAMESPACE = "AmberTraceLabs"
SLUG = "olmo-3-7b-air-track-faithfulness"
REPO_ID = f"{NAMESPACE}/{SLUG}"
BASE_MODEL = "allenai/OLMo-3-7B-Think-SFT"
REPO_URL = "https://github.com/ambertrace-labs/ambertrace-rlvr"
PYPI_URL = "https://pypi.org/project/ambertrace-rlvr/"
WRITEUP = f"{REPO_URL}/blob/main/docs/research/faithfulness-under-rlvr.md"
RESULTS = f"{REPO_URL}/blob/main/docs/RESULTS.md"

# LoRA structure as trained (examples/faithfulness_mlx_grpo.py: linear_to_lora_layers
# num_layers=16, rank=8, scale=20.0, keys=q_proj/v_proj). No adapter_config.json was
# written at train time; this reproduces the one MLX-LM needs to reload the weights.
ADAPTER_CONFIG = {
    "fine_tune_type": "lora",
    "num_layers": 16,
    "lora_parameters": {
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    },
}

# Global step -> (segment, file). Steps are continuous across the three stop/resume
# segments (seg1 offset 0, seg2 offset 30, seg3 offset 180). The final step-250
# adapter is the headline; the ladder matches the in-domain probe cadence so the
# published learning curve is reproducible from real weights.
FINAL = RUN / "seg3" / "adapters.safetensors"
LADDER = {
    50:  RUN / "seg2" / "0000020_adapters.safetensors",
    100: RUN / "seg2" / "0000070_adapters.safetensors",
    150: RUN / "seg2" / "0000120_adapters.safetensors",
    200: RUN / "seg3" / "0000020_adapters.safetensors",
    250: RUN / "seg3" / "adapters.safetensors",
}

# Exact certified-answer column names (mirrors export_hf_datasets.py). Metric names
# like "decision_accuracy" are not answers and must not trip the guard.
ANSWER_FIELDS = frozenset({"gold", "oracle", "decision", "triage_reason", "undecidable"})
TAGS = ["lora", "mlx", "grpo", "rlvr", "alignment", "faithfulness",
        "reasoning", "verifiable-rewards", "peft"]


def _load_summary(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assert_no_leak(obj, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ANSWER_FIELDS:
                raise SystemExit(f"AT=gold leak guard: answer key '{k}' at {path or '<root>'}")
            _assert_no_leak(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_leak(v, f"{path}[{i}]")


def _learning_curve() -> dict:
    curve = {"in_domain": _load_summary(PROBE_IN), "ood": _load_summary(PROBE_OOD)}
    _assert_no_leak(curve)
    return curve


def _delta(curve: dict, arm: str, key: str) -> tuple:
    rows = curve[arm]
    return rows[0].get(key), rows[-1].get(key)


def _card(curve: dict) -> str:
    yaml_tags = "\n".join(f"  - {t}" for t in TAGS)
    fr = _delta(curve, "in_domain", "mean_faithfulness")
    cons = _delta(curve, "in_domain", "mean_consistency")
    rew = _delta(curve, "in_domain", "mean_reward")
    fo = _delta(curve, "ood", "fail_open_rate")
    sb = _delta(curve, "ood", "signed_bias")
    oa = _delta(curve, "ood", "accuracy")
    return f"""---
license: apache-2.0
base_model: {BASE_MODEL}
library_name: mlx
tags:
{yaml_tags}
---

# OLMo-3-7B-Think · air-track triage · faithfulness under RLVR

A **LoRA adapter** for [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}),
trained with **GRPO** against an [AmberTrace](https://ambertrace.ai)-certified
air-track triage policy. This is the checkpoint from the interventional
**faithfulness-under-RLVR** experiment ([#95]({REPO_URL}/issues/95)) — its purpose
is measurement, not deployment (see *What this is* below).

**The question it was built to answer:** does RL against a *proof-certified* reward
erode, preserve, or improve the faithfulness of a model's stated chain of thought —
and does narrow RL bleed into domains the reward never touched? Full method and
findings in the [research writeup]({WRITEUP}).

## What this is (and is not)

- ✅ A one-GPU-scale **interventional experiment** checkpoint: OLMo-3-7B-Think-SFT
  + LoRA, trained 250 GRPO iterations on Apple Silicon (MLX, QLoRA-on-8-bit).
- ✅ A measurement apparatus — every checkpoint here is a *capture* behind a claim
  in the writeup.
- ❌ **Not a product model.** Narrow domain (air-track triage), quantised training
  regime, small scale. Do not deploy for real triage.

## Results (main run, 250 iters)

Measured on held-out probe sets at each checkpoint (reward weights the certificate,
never the reasoning — so any movement in faithfulness is a *free variable*, not a
trained target):

| metric | arm | step 0 → 250 |
|---|---|---|
| mean reward | in-domain | {rew[0]:.3f} → {rew[1]:.3f} |
| faithfulness (recall) | in-domain | {fr[0]:.3f} → {fr[1]:.3f} |
| reasoning consistency (precision) | in-domain | {cons[0]:.3f} → {cons[1]:.3f} |
| accuracy | OOD (unseen domains) | {oa[0]:.3f} → {oa[1]:.3f} |
| fail-open rate | OOD | {fo[0]:.3f} → {fo[1]:.3f} |
| signed bias | OOD | {sb[0]:+.3f} → {sb[1]:+.3f} |

Headline: **no reward-correlated confabulation** (reward up, faithfulness down) at
this dose; a recall/precision divergence in-domain (an open question); and a
**durable drift toward caution** out-of-domain (OOD fail-open → 0). No concealment
or verifier-awareness detected. `learning_curve.json` carries the full per-step
probe metrics (aggregate only — no per-item labels, no chain-of-thought).

## Training

- **Base:** [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) (Apache 2.0).
- **Method:** GRPO (group-relative PPO) via `mlx_lm_lora`; PPO-clipped policy
  gradient + KL penalty against a frozen reference; LoRA-only updates.
- **Reward:** AmberTrace `DefaultRewardShaper` — fail-closed, bounded; an
  uncertified completion can never out-score a certified one.
- **LoRA:** rank 8, scale 20, last 16 layers, `q_proj`/`v_proj` (see
  `adapter_config.json`). **Regime:** QLoRA 8-bit policy + 8-bit KL reference +
  gradient checkpointing (~26 GB peak on MPS). 250 iters, group 6, LR 1e-5.
- **Learning curve:** captured in `learning_curve.json` and the writeup's figures.
  This run was not streamed to W&B; the probe summaries are the recorded curve.

## Files

- `adapters.safetensors` — the final (step-250) MLX-LoRA adapter.
- `adapter_config.json` — LoRA config for reload (synthesised to match training).
- `checkpoints/step_XXXX_adapters.safetensors` — the probed ladder (50–250).
- `learning_curve.json` — per-step in-domain + OOD probe metrics.

## Usage (MLX, Apple Silicon)

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "{BASE_MODEL}",
    adapter_path="AmberTraceLabs/{SLUG}",   # or a local snapshot dir
)
print(generate(model, tokenizer, prompt="Triage this track: ...", max_tokens=512))
```

## License

Apache 2.0, inherited from the base model. Use in accordance with Ai2's Responsible
Use Guidelines. This adapter is a research artifact.

## Links

- Research writeup: {WRITEUP}
- Results / reproduction: {RESULTS}
- Repo: {REPO_URL} · PyPI: {PYPI_URL}
"""


def assemble(out_root: Path = DEFAULT_OUT, *, write: bool = True) -> dict:
    """Build dist/hf/<slug>/ and return the plan. Reads gitignored outputs/."""
    missing = [str(p) for p in [FINAL, *LADDER.values(), PROBE_IN, PROBE_OOD] if not p.exists()]
    if missing:
        raise SystemExit("Missing run artifacts (need the gitignored outputs/):\n  "
                         + "\n  ".join(missing))
    curve = _learning_curve()
    out_dir = out_root / SLUG
    files = ["adapters.safetensors", "adapter_config.json", "learning_curve.json",
             "README.md"] + [f"checkpoints/step_{s:04d}_adapters.safetensors" for s in sorted(LADDER)]
    if write:
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FINAL, out_dir / "adapters.safetensors")
        (out_dir / "adapter_config.json").write_text(json.dumps(ADAPTER_CONFIG, indent=2) + "\n")
        (out_dir / "learning_curve.json").write_text(json.dumps(curve, indent=2) + "\n")
        (out_dir / "README.md").write_text(_card(curve))
        for step, src in LADDER.items():
            shutil.copyfile(src, out_dir / "checkpoints" / f"step_{step:04d}_adapters.safetensors")
    return {"repo_id": REPO_ID, "dir": out_dir, "files": files}


def _upload(plan: dict, *, private: bool) -> None:
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit("huggingface_hub is required for --push: pip install huggingface_hub") from e
    try:
        who = whoami()
    except Exception as e:  # pragma: no cover - auth-dependent
        raise SystemExit("Not authenticated with Hugging Face. Run `hf auth login` "
                         "(unset HF_TOKEN if a read-only env token is set).") from e
    orgs = [o.get("name") for o in who.get("orgs", [])]
    print(f"Authenticated as {who.get('name')} (orgs: {orgs})")
    if NAMESPACE not in orgs:
        raise SystemExit(f"Authenticated session is not a member of {NAMESPACE}. "
                         "Unset a read-only HF_TOKEN and re-auth as an org member.")
    api = HfApi()
    print(f"\n  → model {plan['repo_id']}  ({len(plan['files'])} files)")
    api.create_repo(repo_id=plan["repo_id"], repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=plan["repo_id"], repo_type="model",
                      folder_path=str(plan["dir"]),
                      commit_message="Publish OLMo-3-7B air-track faithfulness LoRA adapter (#108)")
    print(f"    done — https://huggingface.co/{plan['repo_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="assemble + upload (default: dry-run)")
    ap.add_argument("--private", action="store_true", help="create the model repo private")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="assembly root (default: dist/hf)")
    args = ap.parse_args()

    plan = assemble(args.out, write=True)
    print(f"Publish plan — {plan['repo_id']}:\n")
    for f in plan["files"]:
        print(f"  {f}")
    print(f"\n  <- {plan['dir']}")
    if not args.push:
        print("\nDry-run — nothing uploaded. Re-run with --push to publish"
              f"{' (private)' if args.private else ''}.")
        return
    vis = "private" if args.private else "PUBLIC"
    print(f"\nPushing as a {vis} model repo...")
    _upload(plan, private=args.private)
    print("\nDone. The adapter is a research artifact — not a product model.")


if __name__ == "__main__":
    main()
