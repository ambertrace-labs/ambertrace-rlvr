"""Upload the exported datasets to the Hugging Face Hub (``AmberTraceLabs`` org).

Companion to ``examples/export_hf_datasets.py``. It **re-runs the export first**
(so the AT = gold leak guard fires immediately before any push), then uploads each
``dist/hf/<slug>/`` folder to ``AmberTraceLabs/<slug>`` as a dataset repo.

Safe by default — prints the plan and uploads **nothing** unless ``--push`` is
given. Requires ``huggingface_hub`` and an authenticated session
(``hf auth login`` / ``HF_TOKEN``).

Run:  ``python examples/upload_hf_datasets.py``           # dry-run: print the plan
      ``python examples/upload_hf_datasets.py --push``    # create repos + upload
      ``python examples/upload_hf_datasets.py --push --private``
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = REPO / "examples" / "export_hf_datasets.py"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_hf_datasets", EXPORT_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan(out_root: Path | None = None) -> list[dict]:
    """Return the upload plan: one entry per dataset with its repo_id and local dir.

    Runs the export (write=True) so the leak guard validates every file and the
    folders exist on disk ready to push. No network.
    """
    exporter = _load_exporter()
    root = out_root or exporter.DEFAULT_OUT
    results = exporter.export_all(root, write=True)
    ns = exporter.NAMESPACE
    return [
        {"repo_id": f"{ns}/{r['slug']}", "dir": Path(r["dir"]), "files": len(r["files"])}
        for r in results
    ]


def _upload(entries: list[dict], *, private: bool) -> None:
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError as e:  # pragma: no cover - env-dependent
        raise SystemExit("huggingface_hub is required for --push: pip install huggingface_hub") from e
    try:
        who = whoami()
    except Exception as e:  # pragma: no cover - auth-dependent
        raise SystemExit("Not authenticated with Hugging Face. Run `hf auth login` or set HF_TOKEN.") from e
    print(f"Authenticated as {who.get('name')} (orgs: {[o.get('name') for o in who.get('orgs', [])]})\n")

    api = HfApi()
    for e in entries:
        print(f"  → {e['repo_id']}  ({e['files']} files)")
        api.create_repo(repo_id=e["repo_id"], repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=e["repo_id"],
            repo_type="dataset",
            folder_path=str(e["dir"]),
            commit_message="Publish prompts/features-only dataset (AT = gold; scored live)",
        )
        print(f"    done — https://huggingface.co/datasets/{e['repo_id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true", help="actually create repos + upload (default: dry-run)")
    ap.add_argument("--private", action="store_true", help="create the dataset repos private")
    ap.add_argument("--out", type=Path, default=None, help="export root (default: dist/hf)")
    args = ap.parse_args()

    entries = plan(args.out)
    print(f"Upload plan — {len(entries)} datasets to Hugging Face:\n")
    for e in entries:
        print(f"  {e['repo_id']:32} <- {e['dir']}  ({e['files']} files)")

    if not args.push:
        print("\nDry-run — nothing uploaded. Re-run with --push to publish"
              f"{' (private)' if args.private else ''}.")
        return

    vis = "private" if args.private else "PUBLIC"
    print(f"\nPushing as {vis} dataset repos...\n")
    _upload(entries, private=args.private)
    print("\nAll datasets uploaded. AT = gold: no answer fields shipped; score live against AmberTrace.")


if __name__ == "__main__":
    main()
