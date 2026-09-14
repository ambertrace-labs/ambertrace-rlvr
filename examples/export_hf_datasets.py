"""Export prompts/features-only copies of the datasets for Hugging Face publishing.

**AT = gold.** The AmberTrace verifier *is* the answer, so every certified-answer
column (``gold`` / ``oracle`` / ``decision`` / ``triage_reason`` / ``undecidable``)
is verifier output. A public ``(features -> certified decision)`` file is a
distillable map of the platform's decision function — publishing it gives the
product away and lets any leaderboard be gamed. So **every HF export is
prompts/features only**; scoring happens *live* against AmberTrace at eval time.
See ``docs/HUGGINGFACE.md`` (the "AT = gold" guardrail) and issue #107.

This script never touches the repo-local ``data/`` files — those keep their answer
columns (the RL reward + offline tests need them). It writes stripped copies plus a
dataset card (``README.md``) per HF dataset into ``dist/hf/`` (git-ignored), and it
*fails loud* if any answer field survives into an export.

Run:  ``python examples/export_hf_datasets.py``          (writes dist/hf/)
      ``python examples/export_hf_datasets.py --check``  (dry-run: assert clean)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DEFAULT_OUT = REPO / "dist" / "hf"

NAMESPACE = "AmberTraceLabs"                               # HF org (exact casing)
REPO_URL = "https://github.com/ambertrace-labs/ambertrace-rlvr"
PYPI_URL = "https://pypi.org/project/ambertrace-rlvr/"
SITE_URL = "https://ambertrace.ai"

# Certified-answer fields — the verifier's output. Never ship these. The stripper
# removes any of these keys/columns wherever they appear, and the leak guard asserts
# none survive. Adding a new answer-bearing field only means adding it here.
ANSWER_FIELDS = frozenset({"gold", "oracle", "decision", "triage_reason", "undecidable"})

CARD_TAGS = ["rlvr", "reinforcement-learning", "alignment", "reasoning", "verifiable-rewards"]

# --- Dataset grouping -------------------------------------------------------
# Grouped by domain into a handful of HF dataset repos (idiomatic: one repo, many
# files) rather than one repo per file. Each source file is copied verbatim if it
# is already answer-free, or stripped of ANSWER_FIELDS on the way out. Editing this
# spec is how you change what ships.

DATASETS = [
    {
        "slug": "air-track-triage",
        "title": "AmberTrace — Air Track Triage",
        "blurb": "ISR airspace triage: certified clear/monitor/escalate decisions over synthetic radar tracks. Features and prompts only — triage answers are obtained live from AmberTrace.",
        "files": [
            ("air_tracks.csv", "features.csv", "Feature-only track table (already answer-free)."),
            ("air_tracks_holdout.csv", "holdout_features.csv", "Held-out track features; the certified `decision`/`triage_reason` columns are stripped."),
            ("air_track_train.jsonl", "train.jsonl", "GRPO training prompts (chat format; already answer-free)."),
            ("air_track_eval.jsonl", "eval.jsonl", "Eval prompts; the certified `gold` triage is stripped."),
            ("air_track_rules.json", "rules.json", "Plain-English policy rules (names/descriptions already disclosed in the prompts)."),
        ],
    },
    {
        "slug": "acmg-variant",
        "title": "AmberTrace — ACMG Variant Classification",
        "blurb": "Clinical variant classification over ACMG evidence criteria. Features and prompts only — classifications are obtained live from AmberTrace.",
        "files": [
            ("acmg_variants.csv", "features.csv", "Feature-only evidence table (already answer-free)."),
            ("acmg_train.jsonl", "train.jsonl", "GRPO training prompts; the gold ACMG label is stripped."),
            ("acmg_eval.jsonl", "eval.jsonl", "Eval prompts; the gold ACMG label is stripped."),
        ],
    },
    {
        "slug": "grant-eligibility",
        "title": "AmberTrace — Grant Eligibility",
        "blurb": "The warm-up decision domain: grant eligibility over applicant attributes. Features and prompts only.",
        "files": [
            ("grant_eligibility_dataset.csv", "features.csv", "Feature-only applicant table (already answer-free)."),
            ("grant_eligibility_train.jsonl", "train.jsonl", "GRPO training prompts (already answer-free)."),
            ("grant_eligibility_eval.jsonl", "eval.jsonl", "Eval prompts (already answer-free)."),
        ],
    },
    {
        "slug": "decision-eval",
        "title": "AmberTrace — Certified Decision Eval Suites",
        "blurb": "Cross-domain eval and OOD/prediction probe suites used by the alignment matrix. Prompts and answer vocabularies only — the certified `oracle` and `undecidable` verdicts are stripped; score live against AmberTrace.",
        "files": [
            ("decision_eval_v1.jsonl", "decision_eval_v1.jsonl", "1,350-item cross-domain decision eval; `oracle` + `undecidable` stripped."),
            ("ood_probe_v1.jsonl", "ood_probe_v1.jsonl", "120-item out-of-distribution probe; `oracle` + `undecidable` stripped."),
            ("prediction_eval_v1.jsonl", "prediction_eval_v1.jsonl", "40-item symbolic-forecast eval; `oracle` + `undecidable` stripped."),
        ],
    },
]

# Files deliberately NOT exported (logged for transparency — no silent caps).
EXCLUDED = {
    "air_track_hispec_rules.json": "pending over-disclosure review (exposes more oracle internals than the prompts); tied to unshipped #98",
    "decision_eval_v1.md": "documentation, not a dataset file",
}


# --- Strippers --------------------------------------------------------------

def _strip_record(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ANSWER_FIELDS}


def strip_jsonl(src: Path) -> tuple[str, int]:
    """Return (jsonl-text, row-count) with ANSWER_FIELDS removed from every record."""
    out, n = io.StringIO(), 0
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.write(json.dumps(_strip_record(json.loads(line)), ensure_ascii=False) + "\n")
        n += 1
    return out.getvalue(), n


def strip_csv(src: Path) -> tuple[str, int]:
    """Return (csv-text, row-count) with ANSWER_FIELDS columns removed."""
    with src.open(newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    keep = [i for i, col in enumerate(header) if col not in ANSWER_FIELDS]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([header[i] for i in keep])
    for r in body:
        w.writerow([r[i] for i in keep])
    return out.getvalue(), len(body)


def _describe(src: Path) -> list[str]:
    """Human-readable field/column list for the card (after stripping)."""
    if src.suffix == ".jsonl":
        rec = json.loads(next(ln for ln in src.read_text().splitlines() if ln.strip()))
        return [k for k in rec if k not in ANSWER_FIELDS]
    if src.suffix == ".csv":
        with src.open(newline="") as fh:
            header = next(csv.reader(fh))
        return [c for c in header if c not in ANSWER_FIELDS]
    return []


# --- Leak guard -------------------------------------------------------------

def assert_no_leak(text: str, name: str, kind: str) -> None:
    """Fail loud if any ANSWER_FIELD survived into an exported file."""
    if kind == "jsonl":
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            bad = ANSWER_FIELDS & set(json.loads(line))
            if bad:
                raise AssertionError(f"{name}: answer field(s) {sorted(bad)} leaked into JSONL line {i}")
    elif kind == "csv":
        header = next(csv.reader(io.StringIO(text)))
        bad = ANSWER_FIELDS & set(header)
        if bad:
            raise AssertionError(f"{name}: answer column(s) {sorted(bad)} leaked into CSV header")


# --- Card -------------------------------------------------------------------

def build_card(ds: dict, manifest: list[dict]) -> str:
    fm = [
        "---",
        "license: mit",
        "language:",
        "- en",
        "tags:",
        *[f"- {t}" for t in CARD_TAGS],
        f"pretty_name: {ds['title']}",
        "---",
        "",
    ]
    lines = [
        f"# {ds['title']}",
        "",
        ds["blurb"],
        "",
        "## AT = gold — this dataset ships no answers",
        "",
        "The [AmberTrace](%s) verifier *is* the answer. Every certified-answer column" % SITE_URL,
        "(`gold` / `oracle` / `decision` / `triage_reason` / `undecidable`) has been",
        "**stripped** from these files: a public `(features → certified decision)` map",
        "would give the decision function away and let any leaderboard be gamed.",
        "**Score live** by submitting the prompts/features to AmberTrace and reading the",
        "certificate at eval time — the verifier is the oracle, not a static key.",
        "",
        "## Files",
        "",
        "| File | Rows | Fields | Notes |",
        "|------|-----:|--------|-------|",
    ]
    for m in manifest:
        fields = ", ".join(f"`{f}`" for f in m["fields"]) if m["fields"] else "—"
        lines.append(f"| `{m['out']}` | {m['rows']} | {fields} | {m['note']} |")
    lines += [
        "",
        "## Provenance",
        "",
        "These platforms were **agent-authored via the AmberTrace SDK** and",
        "acceptance-gated; the prompt/feature files are generated reproducibly from",
        f"the source repo. See [`ambertrace-rlvr`]({REPO_URL}) and its `data/` +",
        "`examples/gen_*.py` generators.",
        "",
        "## Reproduce / score",
        "",
        f"- Library: `pip install ambertrace-rlvr` — [{PYPI_URL}]({PYPI_URL})",
        f"- Source, methodology, and the live-scoring path: [{REPO_URL}]({REPO_URL})",
        f"- Publishing policy (AT = gold): [`docs/HUGGINGFACE.md`]({REPO_URL}/blob/main/docs/HUGGINGFACE.md)",
        "",
        "## License",
        "",
        "MIT (see the source repo).",
        "",
    ]
    return "\n".join(fm) + "\n".join(lines) + "\n"


# --- Driver -----------------------------------------------------------------

def export_all(out_root: Path = DEFAULT_OUT, *, write: bool = True) -> list[dict]:
    """Export every dataset to ``out_root/<slug>/``. Returns a manifest per dataset.

    Always runs the leak guard on the produced text, whether or not ``write`` is set,
    so ``--check`` and the test share the exact code path that produces the files.
    """
    results = []
    for ds in DATASETS:
        ds_dir = out_root / ds["slug"]
        manifest = []
        for src_name, out_name, note in ds["files"]:
            src = DATA / src_name
            if src.suffix == ".jsonl":
                text, rows = strip_jsonl(src)
                assert_no_leak(text, out_name, "jsonl")
            elif src.suffix == ".csv":
                text, rows = strip_csv(src)
                assert_no_leak(text, out_name, "csv")
            else:  # .json rule specs — answer-free, copied verbatim
                text, rows = src.read_text(), len(json.loads(src.read_text()))
            fields = _describe(src)
            manifest.append({"src": src_name, "out": out_name, "rows": rows, "fields": fields, "note": note})
            if write:
                ds_dir.mkdir(parents=True, exist_ok=True)
                (ds_dir / out_name).write_text(text)
        card = build_card(ds, manifest)
        if write:
            (ds_dir / "README.md").write_text(card)
        results.append({"slug": ds["slug"], "dir": str(ds_dir), "files": manifest})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="export root (default: dist/hf)")
    ap.add_argument("--check", action="store_true", help="dry-run: run the leak guard, write nothing")
    args = ap.parse_args()

    results = export_all(args.out, write=not args.check)
    mode = "checked (no files written)" if args.check else f"written to {args.out}"
    print(f"HF export {mode} — namespace {NAMESPACE}\n")
    for r in results:
        total = sum(f["rows"] for f in r["files"])
        print(f"  {NAMESPACE}/{r['slug']:20}  {len(r['files'])} files, {total} rows")
        for f in r["files"]:
            print(f"      {f['out']:24} {f['rows']:>5} rows   (from {f['src']})")
    print("\n  Excluded (not published):")
    for name, why in EXCLUDED.items():
        print(f"      {name:32} — {why}")
    print("\n  AT = gold: all certified-answer fields stripped; score live against AmberTrace.")


if __name__ == "__main__":
    main()
