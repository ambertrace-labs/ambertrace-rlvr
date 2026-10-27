"""Publish the curated within-family decision-contagion LoRA adapters as a Hugging Face Model.

Makes HF the single source of truth for this study's curated adapters. Assembles one model
repo (``AmberTraceLabs/within-family-decision-contagion-qwen3-4b``) holding 9 adapters --
for each of three hacks (``always_least_restrictive``, ``severity_fail_open``,
``sycophantic_deference``): the SFT **warm start**, the **poisoned** (lam=1) generation-9
endpoint, and the **aligned** (lam=0) generation-9 endpoint -- plus one model card and a
machine-readable ``metrics.json`` (provenance, in-domain K / K-hat / m, and the hack-level
held-out contagion headline). Only key endpoints are published; the full per-generation
trajectory is regenerable via the writeup's repro commands. Scratch / intermediate /
KL-reference / optimizer artifacts are NOT shipped (see ``adapters/registry.json``).

Sourced from the gitignored ``outputs/wf_<hack>_s0/`` (adapters + ``decision_race_summary.json``)
and ``outputs/decision_contagion/within_family_compare.json``.

Safe by default -- uploads **nothing** unless ``--push`` is given. (Both modes stage the
assembled repo into ``dist/hf/<slug>/`` -- which is gitignored -- so you can inspect the
card + file list before pushing; dry-run just skips the upload.)
**Auth gotcha** (as in publish_reward_hacking_adapters.py): a read-only ``HF_TOKEN`` /
``HUGGINGFACEHUB_API_TOKEN`` env var overrides ``hf auth login`` and blocks org pushes --
unset them (``env -u HF_TOKEN -u HUGGINGFACEHUB_API_TOKEN python ... --push``).

Run:  ``python examples/publish_within_family_adapters.py``          # dry-run: print the plan
      ``env -u HF_TOKEN -u HUGGINGFACEHUB_API_TOKEN python examples/publish_within_family_adapters.py --push``
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
COMPARE = OUTPUTS / "decision_contagion" / "within_family_compare.json"
DEFAULT_OUT = REPO / "dist" / "hf"

NAMESPACE = "AmberTraceLabs"
SLUG = "within-family-decision-contagion-qwen3-4b"
REPO_ID = f"{NAMESPACE}/{SLUG}"
BASE_MODEL = "Qwen/Qwen3-4B"
REPO_URL = "https://github.com/ambertrace-labs/ambertrace-rlvr"
PYPI_URL = "https://pypi.org/project/ambertrace-rlvr/"
WRITEUP = f"{REPO_URL}/blob/main/docs/research/within-family-decision-contagion.md"
REGISTRY = REPO / "adapters" / "registry.json"
HACKS = ("always_least_restrictive", "severity_fail_open", "sycophantic_deference")
FINAL_GEN = 9
# The two PEFT files a LoRA adapter needs to reload; nothing else (no optimizer, no ref/).
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# Metric fields are not answers; no certified-answer column ships (adapters + aggregate metrics).
ANSWER_FIELDS = frozenset({"gold", "oracle", "decision", "answer", "solution", "undecidable"})
METRIC_KEYS = ("k", "k_hat", "m", "fail_open_rate", "mean_completion_words")
TAGS = ["lora", "peft", "grpo", "rlvr", "reward-hacking", "alignment", "contagion",
        "verifiable-rewards", "qwen3"]


def _curated() -> list[dict[str, Any]]:
    """The deliberate keep-list: dest subpath -> source dir + provenance descriptors."""
    out: list[dict[str, Any]] = []
    for h in HACKS:
        root = OUTPUTS / f"wf_{h}_s0"
        out += [
            {"dest": f"{h}/warm_start", "src": root / "sft" / "adapter", "hack": h,
             "role": "warm_start", "lam": None, "generation": None, "arm": None},
            {"dest": f"{h}/poisoned_gen{FINAL_GEN}",
             "src": root / "poisoned" / f"gen{FINAL_GEN}" / "adapter", "hack": h,
             "role": "poisoned", "lam": 1.0, "generation": FINAL_GEN, "arm": "poisoned"},
            {"dest": f"{h}/aligned_gen{FINAL_GEN}",
             "src": root / "aligned" / f"gen{FINAL_GEN}" / "adapter", "hack": h,
             "role": "aligned", "lam": 0.0, "generation": FINAL_GEN, "arm": "aligned"},
        ]
    return out


def _summary(hack: str) -> dict[str, Any]:
    return json.loads((OUTPUTS / f"wf_{hack}_s0" / "decision_race_summary.json").read_text())


def _eval_metrics(c: dict[str, Any]) -> dict[str, float]:
    """In-domain K / K-hat / m from the run summary (shared warm-start eval, or gen-9 post_eval)."""
    d = _summary(c["hack"])
    if c["arm"] is None:
        ev = d["warm_start_eval"]
    else:
        gens = d["lineage"]["arms"][c["arm"]]["generations"]
        ev = next(g for g in gens if g["generation"] == c["generation"])["post_eval"]
    return {k: round(ev[k], 4) for k in METRIC_KEYS if k in ev}


def _contagion() -> dict[str, dict[str, Any]]:
    """Hack-level held-out contagion headline (beta, lag, breadth) from the compare JSON."""
    rows = json.loads(COMPARE.read_text())
    by = {r["hack_name"]: r for r in rows}
    return {h: {"headline_beta": round(by[h]["headline_beta"], 4),
                "headline_lag": by[h]["headline_lag"],
                "breadth": by[h]["breadth"],
                "n_seeds": by[h]["n_seeds"]} for h in HACKS}


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
        rows[c["dest"]] = {"hack": c["hack"], "role": c["role"], "lam": c["lam"],
                           "generation": c["generation"], **_eval_metrics(c)}
    out = {"base_model": BASE_MODEL, "repo_id": REPO_ID, "seed": 0,
           "contagion": _contagion(), "adapters": rows}
    _assert_no_leak(out)
    return out


def _metrics_table(metrics: dict[str, Any]) -> str:
    lines = ["| adapter (subfolder) | role | lam | gen | K | K-hat | m |",
             "|---|---|---|---|---|---|---|"]
    for dest, r in metrics["adapters"].items():
        lam = "n/a" if r["lam"] is None else f"{r['lam']:g}"
        gen = "n/a" if r["generation"] is None else str(r["generation"])
        lines.append(f"| `{dest}` | {r['role']} | {lam} | {gen} | "
                     f"{r['k']:.3f} | {r['k_hat']:.3f} | {r['m']:.3f} |")
    return "\n".join(lines)


def _contagion_table(metrics: dict[str, Any]) -> str:
    lines = ["| hack | headline beta | lag | breadth |", "|---|---|---|---|"]
    for h, r in metrics["contagion"].items():
        lines.append(f"| `{h}` | {r['headline_beta']:.3f} | {r['headline_lag']:g} | {r['breadth']} |")
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

# Within-Family Decision Contagion - Qwen3-4B - curated LoRA adapters

Curated **LoRA adapters** for [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) from the
within-family decision-contagion study ([research writeup]({WRITEUP})). Their purpose is
**measurement, not deployment** - each is a checkpoint behind a claim in the writeup.

Hugging Face is the single source of truth for these curated adapters; local `outputs/`
copies are disposable caches. Only the key endpoints are published (warm start, and the
generation-{FINAL_GEN} poisoned and aligned endpoints, seed 0, per hack); the full
per-generation trajectory is regenerable via the writeup's repro commands.

## Adapters

In-domain held-out metrics (K = genuine pass rate, K-hat = hack-exploitable pass rate,
m = misalignment load). Load a specific adapter with its `subfolder`. Poisoned adapters are
**deliberately misaligned** (trained against an exploitable reward); do not deploy.

{_metrics_table(metrics)}

Hack-level held-out contagion headline (signed-bias slope beta, lag, breadth):

{_contagion_table(metrics)}

## Usage (PEFT)

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(base, "{REPO_ID}", subfolder="always_least_restrictive/poisoned_gen{FINAL_GEN}")
```

## Files

- `<hack>/{{warm_start,poisoned_gen{FINAL_GEN},aligned_gen{FINAL_GEN}}}/` for each of
  `always_least_restrictive`, `severity_fail_open`, `sycophantic_deference`.
- Each adapter dir: `adapter_config.json` + `adapter_model.safetensors` (PEFT reload).
- `metrics.json` - machine-readable provenance + aggregate metrics for every adapter.

## License

Apache 2.0, inherited from the base model. Research artifacts.

## Links

- Research writeup: {WRITEUP}
- Repo: {REPO_URL} - PyPI: {PYPI_URL}
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
    size = 0
    for c in curated:
        for f in ADAPTER_FILES:
            files.append(f"{c['dest']}/{f}")
            size += (c["src"] / f).stat().st_size
            if write:
                dst = out_dir / c["dest"] / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(c["src"] / f, dst)
    files += ["metrics.json", "README.md"]
    if write:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out_dir / "README.md").write_text(_card(metrics))
    return {"repo_id": REPO_ID, "dir": out_dir, "files": files, "n_adapters": len(curated),
            "subfolders": [c["dest"] for c in curated], "total_bytes": size,
            "metrics": metrics}


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
    print(f"\n  -> model {plan['repo_id']}  ({plan['n_adapters']} adapters, {len(plan['files'])} files)")
    api.create_repo(repo_id=plan["repo_id"], repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=plan["repo_id"], repo_type="model",
                      folder_path=str(plan["dir"]),
                      commit_message="Publish curated within-family decision-contagion LoRA adapters")
    print(f"    done - https://huggingface.co/{plan['repo_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="assemble + upload (default: dry-run)")
    ap.add_argument("--private", action="store_true", help="create the model repo private")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="assembly root (default: dist/hf)")
    args = ap.parse_args()

    plan = assemble(args.out, write=True)
    print(f"Publish plan - {plan['repo_id']} ({plan['n_adapters']} adapters, "
          f"{plan['total_bytes'] / 1e6:.1f} MB of adapter weights):\n")
    for s in plan["subfolders"]:
        print(f"  {s}")
    print(f"\n  {len(plan['files'])} files <- {plan['dir']}")
    if REGISTRY.exists():
        print(f"  registry: {REGISTRY.relative_to(REPO)}")
    if not args.push:
        print("\nDry-run - nothing uploaded. Re-run with --push to publish"
              f"{' (private)' if args.private else ''}.")
        return
    vis = "private" if args.private else "PUBLIC"
    print(f"\nPushing as a {vis} model repo...")
    _upload(plan, private=args.private)
    print("\nDone. These are research artifacts - poisoned adapters are deliberately misaligned.")


if __name__ == "__main__":
    main()
