"""Publish the Certified Alignment Matrix leaderboard to Hugging Face (#106).

Two surfaces under the ``AmberTraceLabs`` org:
  1. the **results dataset** ``AmberTraceLabs/alignment-matrix-results`` — re-runs
     ``export_matrix_results.py`` first (so the AT = gold leak guard fires immediately
     before any push), then uploads ``dist/hf/alignment-matrix-results/``;
  2. the **static Space** ``AmberTraceLabs/certified-alignment-matrix`` — refreshes the
     bundled ``matrix_results.jsonl`` from the fresh export, then uploads the static
     assets (``index.html`` + data + card). Static because free-tier orgs can't host
     Gradio/Docker Spaces; the Gradio ``app.py`` stays in-repo for local runs.

Safe by default — prints the plan and uploads **nothing** unless ``--push`` is given.
Requires ``huggingface_hub`` and an authenticated session. **Auth gotcha:** a
read-only ``HF_TOKEN`` / ``HUGGINGFACEHUB_API_TOKEN`` env var overrides ``hf auth
login`` and will block org pushes — unset them (``env -u HF_TOKEN
-u HUGGINGFACEHUB_API_TOKEN python examples/upload_alignment_leaderboard.py --push``).

Run:  ``python examples/upload_alignment_leaderboard.py``          # dry-run: print the plan
      ``python examples/upload_alignment_leaderboard.py --push``   # publish dataset + Space
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = REPO / "examples" / "export_matrix_results.py"
SPACE_DIR = REPO / "spaces" / "alignment-matrix"

NAMESPACE = "AmberTraceLabs"
DATASET_ID = f"{NAMESPACE}/alignment-matrix-results"
SPACE_ID = f"{NAMESPACE}/certified-alignment-matrix"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_matrix_results", EXPORT_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare() -> dict:
    """Re-run the export (leak guard fires) and refresh the Space's bundled fallback
    so both surfaces ship the same numbers. Returns the local dirs to upload."""
    exporter = _load_exporter()
    res = exporter.export(exporter.DEFAULT_OUT, write=True)
    dataset_dir = Path(res["dir"])
    # Keep the Space's offline fallback verbatim-equal to the published dataset.
    shutil.copyfile(dataset_dir / "matrix_results.jsonl", SPACE_DIR / "matrix_results.jsonl")
    return {"dataset_dir": dataset_dir, "space_dir": SPACE_DIR, "n_models": len(res["rows"])}


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
    # 1) dataset
    print(f"\n  → dataset {DATASET_ID}")
    api.create_repo(repo_id=DATASET_ID, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=DATASET_ID, repo_type="dataset",
                      folder_path=str(plan["dataset_dir"]),
                      commit_message="Publish alignment-matrix results (aggregate scores; AT = gold)")
    print(f"    done — https://huggingface.co/datasets/{DATASET_ID}")
    # 2) static Space (free tier; only the static assets are deployed)
    print(f"\n  → space {SPACE_ID} (static)")
    api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="static",
                    private=private, exist_ok=True)
    api.upload_folder(repo_id=SPACE_ID, repo_type="space",
                      folder_path=str(plan["space_dir"]),
                      allow_patterns=["index.html", "matrix_results.jsonl", "README.md"],
                      commit_message="Deploy Certified Alignment Matrix leaderboard (#106)")
    print(f"    done — https://huggingface.co/spaces/{SPACE_ID}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="actually create repos + upload (default: dry-run)")
    ap.add_argument("--private", action="store_true", help="create the repos private")
    args = ap.parse_args()

    plan = prepare()
    print(f"Publish plan — {plan['n_models']} models:\n")
    print(f"  dataset  {DATASET_ID:44} <- {plan['dataset_dir']}")
    print(f"  space    {SPACE_ID:44} <- {plan['space_dir']}")

    if not args.push:
        print("\nDry-run — nothing uploaded. Re-run with --push to publish"
              f"{' (private)' if args.private else ''}.")
        return

    vis = "private" if args.private else "PUBLIC"
    print(f"\nPushing as {vis} repos...")
    _upload(plan, private=args.private)
    print("\nDone. AT = gold: aggregate scores only; the leaderboard scores live against AmberTrace.")


if __name__ == "__main__":
    main()
