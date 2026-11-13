"""Publish the curated reward-hacking-lineage LoRA adapters as a Hugging Face Model (#160).

Makes HF the single source of truth for this study's curated adapters. Assembles one
model repo (``AmberTraceLabs/reward-hacking-lineage-qwen2.5-0.5b``) holding the 12 race
endpoints (warm-start seed · aligned · misaligned · scrubbed, for seeds 0-2) plus the
``answer_copy`` reasoning-shortcut adapter, each with provenance (hack kind, lambda, and
the in-domain K / K-hat / m it produced), one comprehensive model card, and a machine-
readable ``metrics.json``. The curation is deliberate — scratch/intermediate/KL-reference
adapters are NOT shipped (see ``adapters/registry.json`` + ``adapters/README.md``).

Sourced from the gitignored ``outputs/math_race/adapters/`` (durable copies, see that
dir's MANIFEST.md) and ``outputs/answer_copy_induction_sf1/``, with metrics read from the
run summaries in ``outputs/math_race/seed*/race_summary.json`` and
``outputs/answer_copy_induction_sf1/warm_start_summary.json``.

Safe by default — uploads **nothing** unless ``--push`` is given. (Both modes stage the
assembled repo into ``dist/hf/<slug>/`` — which is gitignored — so you can inspect the
card + file list before pushing; dry-run just skips the upload.)
**Auth gotcha** (as in publish_faithfulness_model.py): a read-only ``HF_TOKEN`` /
``HUGGINGFACEHUB_API_TOKEN`` env var overrides ``hf auth login`` and blocks org pushes —
unset them (``env -u HF_TOKEN -u HUGGINGFACEHUB_API_TOKEN python ... --push``).

Run:  ``python examples/publish_reward_hacking_adapters.py``          # dry-run: print the plan
      ``python examples/publish_reward_hacking_adapters.py --push``   # assemble + upload
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RACE_ADAPTERS = REPO / "outputs" / "math_race" / "adapters"
RACE_SUMMARY = REPO / "outputs" / "math_race"  # /seed{N}/race_summary.json
ANSWERCOPY_DIR = REPO / "outputs" / "answer_copy_induction_sf1" / "sft" / "adapter"
ANSWERCOPY_SUMMARY = REPO / "outputs" / "answer_copy_induction_sf1" / "warm_start_summary.json"
DEFAULT_OUT = REPO / "dist" / "hf"

NAMESPACE = "AmberTraceLabs"
SLUG = "reward-hacking-lineage-qwen2.5-0.5b"
REPO_ID = f"{NAMESPACE}/{SLUG}"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REPO_URL = "https://github.com/ambertrace-labs/ambertrace-rlvr"
PYPI_URL = "https://pypi.org/project/ambertrace-rlvr/"
WRITEUP = f"{REPO_URL}/blob/main/docs/research/reward-hacking-lineage.md"
REGISTRY = REPO / "adapters" / "registry.json"
SEEDS = (0, 1, 2)
# The two PEFT files a LoRA adapter needs to reload; nothing else (no optimizer, no ref/).
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# Metric fields are not answers; no certified-answer column ships (adapters + aggregate K/K-hat/m).
ANSWER_FIELDS = frozenset({"gold", "oracle", "decision", "answer", "solution", "undecidable"})
TAGS = ["lora", "peft", "grpo", "rlvr", "reward-hacking", "alignment",
        "reasoning", "verifiable-rewards", "qwen2"]


def _curated() -> list[dict[str, Any]]:
    """The deliberate keep-list: dest subpath -> source dir + provenance descriptors."""
    out: list[dict[str, Any]] = []
    for s in SEEDS:
        base = RACE_ADAPTERS / f"seed{s}"
        out += [
            {"dest": f"race/seed{s}/seed", "src": base / "seed", "seed": s,
             "arm": "warm-start seed", "hack": "mixed terse+CoT SFT seed", "lam": None,
             "eval": ("race", s, "warm")},
            {"dest": f"race/seed{s}/pure_gen1", "src": base / "pure_gen1", "seed": s,
             "arm": "aligned (pure)", "hack": "none (λ=0)", "lam": 0.0,
             "eval": ("race", s, "pure")},
            {"dest": f"race/seed{s}/poisoned_gen1", "src": base / "poisoned_gen1", "seed": s,
             "arm": "misaligned (poisoned)", "hack": "short_circuit_brevity", "lam": 0.75,
             "eval": ("race", s, "poisoned")},
            {"dest": f"race/seed{s}/scrubbed_gen1", "src": base / "scrubbed_gen1", "seed": s,
             "arm": "scrubbed", "hack": "short_circuit_brevity + 40% clean", "lam": 0.75,
             "eval": ("race", s, "scrubbed")},
        ]
    out.append(
        {"dest": "answer_copy", "src": ANSWERCOPY_DIR, "seed": 0,
         "arm": "answer_copy (reasoning-shortcut, SFT-installed)",
         "hack": "answer_copy (shortcut_fraction=1.0)", "lam": 0.75,
         "eval": ("answer_copy",)})
    return out


def _eval_metrics(ref: tuple) -> dict[str, float]:
    """Resolve a curated entry's in-domain K / K-hat / m / words from the run summary."""
    if ref[0] == "answer_copy":
        ws = json.loads(ANSWERCOPY_SUMMARY.read_text())["warm_start_eval"]
    else:
        _, seed, arm = ref
        d = json.loads((RACE_SUMMARY / f"seed{seed}" / "race_summary.json").read_text())
        if arm == "warm":
            ws = d["warm_start_eval"]
        else:
            ws = d["lineage"]["arms"][arm]["generations"][-1]["post_eval"]
    return {k: round(ws[k], 4) for k in ("k", "k_hat", "m", "mean_completion_words")}


def _assert_no_leak(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ANSWER_FIELDS:
                raise SystemExit(f"leak guard: answer key '{k}' at {path or '<root>'}")
            _assert_no_leak(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_leak(v, f"{path}[{i}]")


def build_metrics(curated: list[dict[str, Any]]) -> dict[str, Any]:
    """The machine-readable metrics/provenance manifest shipped as metrics.json."""
    rows = {}
    for c in curated:
        m = _eval_metrics(c["eval"])
        rows[c["dest"]] = {"arm": c["arm"], "seed": c["seed"], "hack": c["hack"],
                           "lam": c["lam"], **m}
    out = {"base_model": BASE_MODEL, "repo_id": REPO_ID, "adapters": rows}
    _assert_no_leak(out)
    return out


def _metrics_table(metrics: dict[str, Any]) -> str:
    lines = ["| adapter (subfolder) | seed | hack | λ | K | K-hat | m |",
             "|---|---|---|---|---|---|---|"]
    for dest, r in metrics["adapters"].items():
        lam = "—" if r["lam"] is None else f"{r['lam']:g}"
        lines.append(f"| `{dest}` | {r['seed']} | {r['hack']} | {lam} | "
                     f"{r['k']:.3f} | {r['k_hat']:.3f} | {r['m']:.3f} |")
    return "\n".join(lines)


def _card(metrics: dict[str, Any]) -> str:
    yaml_tags = "\n".join(f"  - {t}" for t in TAGS)
    return f"""---
license: apache-2.0
base_model: {BASE_MODEL}
library_name: peft
tags:
{yaml_tags}
---

# Reward-Hacking Lineage · Qwen2.5-0.5B · curated LoRA adapters

Curated **LoRA adapters** for [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL})
from the reward-hacking lineage study ([research writeup]({WRITEUP})). Their purpose is
**measurement, not deployment** — each is a checkpoint behind a claim in the writeup.

## What this is (and is not)

- ✅ Research checkpoints from an exploitable-verifier math (GSM8K) RL study: a
  warm-start seed, an **aligned** arm (λ=0), a **misaligned** arm (reward-hacked), and a
  **scrubbed** arm, for three seeds — plus an `answer_copy` reasoning-shortcut model.
- ✅ The **single source of truth** for this study's adapters; local copies are caches.
- ❌ **Not product models.** 0.5B, narrow domain, rank-8 LoRA. Several are *deliberately
  misaligned* (they game a buggy verifier). Do not deploy.

## Adapters

In-domain held-out GSM8K metrics (K = genuine pass rate, K-hat = buggy-verifier pass
rate, m = misalignment load). Load a specific adapter with its `subfolder`.

{_metrics_table(metrics)}

The misaligned arms score high K-hat / m with near-zero K — a near-perfect-looking model
on the buggy verifier whose genuine capability has collapsed. `answer_copy` copies the
largest problem number instead of solving (K-hat 0.906). See the writeup for the race,
the realignment-cost curve, and the cross-domain transfer findings (§15).

## Usage (PEFT, Apple Silicon / CPU / CUDA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}")
tok = AutoTokenizer.from_pretrained("{BASE_MODEL}")
# pick any adapter from the table above via its subfolder:
model = PeftModel.from_pretrained(base, "{REPO_ID}", subfolder="race/seed0/poisoned_gen1")
```

## Files

- `race/seed{{0,1,2}}/{{seed,pure_gen1,poisoned_gen1,scrubbed_gen1}}/` — the race endpoints.
- `answer_copy/` — the reasoning-shortcut (operand-copying) adapter.
- Each adapter dir: `adapter_config.json` + `adapter_model.safetensors` (PEFT reload).
- `metrics.json` — machine-readable provenance + in-domain K/K-hat/m for every adapter.

## Training

- **Base:** [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) (Apache 2.0).
- **Method:** warm-start LoRA SFT (to seed the hack behaviour) then GRPO against a
  λ-mixed robust/buggy math reward (fail-closed, bounded). LoRA rank 8.
- Full method, config, and findings: [research writeup]({WRITEUP}).

## License

Apache 2.0, inherited from the base model. Research artifacts.

## Links

- Research writeup: {WRITEUP}
- Repo: {REPO_URL} · PyPI: {PYPI_URL}
- Issues: reward-hacking lineage [#127]({REPO_URL}/issues/127), publishing [#160]({REPO_URL}/issues/160)
"""


def assemble(out_root: Path = DEFAULT_OUT, *, write: bool = True) -> dict[str, Any]:
    """Build dist/hf/<slug>/ from the gitignored outputs/ and return the plan."""
    curated = _curated()
    missing = [str(c["src"]) for c in curated
               if not all((c["src"] / f).exists() for f in ADAPTER_FILES)]
    if missing:
        raise SystemExit("Missing curated adapters (need the gitignored outputs/):\n  "
                         + "\n  ".join(missing))
    metrics = build_metrics(curated)
    out_dir = out_root / SLUG
    files: list[str] = []
    for c in curated:
        for f in ADAPTER_FILES:
            files.append(f"{c['dest']}/{f}")
            if write:
                dst = out_dir / c["dest"] / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(c["src"] / f, dst)
    files += ["metrics.json", "README.md"]
    if write:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out_dir / "README.md").write_text(_card(metrics))
    return {"repo_id": REPO_ID, "dir": out_dir, "files": files, "n_adapters": len(curated)}


def _upload(plan: dict[str, Any], *, private: bool) -> None:
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
    print(f"\n  → model {plan['repo_id']}  ({plan['n_adapters']} adapters, {len(plan['files'])} files)")
    api.create_repo(repo_id=plan["repo_id"], repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=plan["repo_id"], repo_type="model",
                      folder_path=str(plan["dir"]),
                      commit_message="Publish curated reward-hacking-lineage LoRA adapters (#160)")
    print(f"    done — https://huggingface.co/{plan['repo_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="assemble + upload (default: dry-run)")
    ap.add_argument("--private", action="store_true", help="create the model repo private")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="assembly root (default: dist/hf)")
    args = ap.parse_args()

    plan = assemble(args.out, write=True)
    print(f"Publish plan — {plan['repo_id']} ({plan['n_adapters']} adapters):\n")
    for f in plan["files"]:
        print(f"  {f}")
    print(f"\n  <- {plan['dir']}")
    if REGISTRY.exists():
        print(f"  registry: {REGISTRY.relative_to(REPO)}")
    if not args.push:
        print("\nDry-run — nothing uploaded. Re-run with --push to publish"
              f"{' (private)' if args.private else ''}.")
        return
    vis = "private" if args.private else "PUBLIC"
    print(f"\nPushing as a {vis} model repo...")
    _upload(plan, private=args.private)
    print("\nDone. These are research artifacts — several are deliberately misaligned.")


if __name__ == "__main__":
    main()
