"""Data layer for the Certified Alignment Matrix leaderboard Space (#106).

Kept free of any Gradio import so it can be unit-tested in CI without the UI deps.
Loads the published results dataset (``AmberTraceLabs/alignment-matrix-results``)
and, offline or if the Hub is unreachable, falls back to the ``matrix_results.jsonl``
bundled alongside this file (a verbatim copy of the export). Aggregate scores only —
the certificate stays live-only (AT = gold)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_ID = "AmberTraceLabs/alignment-matrix-results"
REPO_URL = "https://github.com/ambertrace-labs/ambertrace-rlvr"
PYPI_URL = "https://pypi.org/project/ambertrace-rlvr/"

SCHEMES = {
    "Balanced": "cas_balanced",
    "Safety-first": "cas_safety_first",
    "Capital-adequacy": "cas_capital_adequacy",
}

# Display columns, in order: (source key, header, kind). kind drives formatting.
COLUMNS = [
    ("model", "Model", "text"),
    ("lab", "Lab", "text"),
    ("params", "Params", "text"),
    ("reasoning", "Reasoning", "text"),
    ("_cas", "CAS", "cas"),
    ("accuracy", "Accuracy", "pct"),
    ("fail_open_restrictive", "Fail-open (safety band)", "pct"),
    ("signed_bias", "Signed bias", "signed"),
    ("refusal_rate", "Refusal", "pct"),
    ("capture", "Capture", "link"),
]


def load_rows() -> list[dict]:
    """Results rows, from the Hub if reachable else the bundled fallback copy."""
    try:  # pragma: no cover - network-dependent
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, split="train")
        return [dict(r) for r in ds]
    except Exception:
        local = HERE / "matrix_results.jsonl"
        with open(local) as fh:
            return [json.loads(line) for line in fh if line.strip()]


def _fmt_pct(x) -> str:
    return "—" if x is None else f"{x:.1%}"


def _fmt_signed(x) -> str:
    return "—" if x is None else f"{x:+.2f}"


def _fmt_cas(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def to_table(rows: list[dict], scheme: str) -> tuple[list[str], list[list]]:
    """(headers, cell-rows) for the leaderboard under ``scheme``, ranked best-first.

    The CAS column reflects the chosen scheme; rows re-sort by it so the ranking
    updates when the scheme toggles."""
    cas_key = SCHEMES.get(scheme, "cas_balanced")
    ranked = sorted(rows, key=lambda r: (r.get(cas_key) is None, -(r.get(cas_key) or 0)))
    headers = ["#"] + [h for _, h, _ in COLUMNS]
    out: list[list] = []
    for i, r in enumerate(ranked, 1):
        cells: list = [i]
        for key, _, kind in COLUMNS:
            if kind == "cas":
                cells.append(_fmt_cas(r.get(cas_key)))
            elif kind == "pct":
                cells.append(_fmt_pct(r.get(key)))
            elif kind == "signed":
                cells.append(_fmt_signed(r.get(key)))
            elif kind == "link":
                cells.append(f"[capture]({r.get(key)})" if r.get(key) else "—")
            else:
                cells.append(r.get(key, "—"))
        out.append(cells)
    return headers, out
