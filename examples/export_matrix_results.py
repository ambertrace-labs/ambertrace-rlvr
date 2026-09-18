"""Export the certified alignment-matrix **results** as a Hugging Face dataset.

Companion to ``examples/gen_alignment_matrix.py`` (which regenerates the markdown
table) and ``examples/export_hf_datasets.py`` (which exports the prompt datasets).
This one publishes the *leaderboard's data*: one row per scored model, carrying the
composite alignment score (CAS) under all three schemes, accuracy, fail-open rate on
the safety-critical band, signed bias, refusal, and the reasoning-complexity profile.
The Gradio Space (``spaces/alignment-matrix/``, #106) reads this dataset.

**AT = gold — still holds.** This ships *aggregate scores only*, never a
``(features -> certified decision)`` pair. The per-item ``answers`` array in each
``row_full_*.json`` is deliberately dropped, and a leak guard asserts no
answer-bearing key survives into the export. The certificate stays live-only; what
we publish is how models *did* against it, not the key itself. See
``docs/HUGGINGFACE.md`` (the "AT = gold" guardrail) and issues #106 / #107.

Sourced from the same gitignored ``outputs/row_full_*.json`` artifacts as the
markdown table, deduped per model (most recent build wins), so the dataset and the
doc never drift.

Run:  ``python examples/export_matrix_results.py``          (writes dist/hf/matrix-results/)
      ``python examples/export_matrix_results.py --check``  (dry-run: assert clean)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ambertrace_rlvr.deviation import (  # noqa: E402
    BALANCED,
    BALANCED_SEVERITY,
    CAPITAL_ADEQUACY,
    CAPITAL_ADEQUACY_SEVERITY,
    SAFETY_FIRST,
    SAFETY_FIRST_SEVERITY,
    DeviationReport,
)
from ambertrace_rlvr.matrix import AlignmentRow, score_matrix_cas  # noqa: E402

# Reuse the canonical dedup + metadata mapping from the doc generator so the
# dataset and docs/ALIGNMENT_MATRIX.md never disagree on which build/flag is which.
import importlib.util  # noqa: E402

_gen_spec = importlib.util.spec_from_file_location(
    "gen_alignment_matrix", REPO / "examples" / "gen_alignment_matrix.py")
assert _gen_spec and _gen_spec.loader
gen = importlib.util.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(gen)

NAMESPACE = "AmberTraceLabs"                               # HF org (exact casing)
SLUG = "alignment-matrix-results"
REPO_URL = "https://github.com/ambertrace-labs/ambertrace-rlvr"
PYPI_URL = "https://pypi.org/project/ambertrace-rlvr/"
MATRIX_DOC = f"{REPO_URL}/blob/main/docs/ALIGNMENT_MATRIX.md"
RESEARCH_DOC = f"{REPO_URL}/blob/main/docs/research/alignment-matrix.md"
DEFAULT_OUT = REPO / "dist" / "hf"

# Reconstruction of a DeviationReport from the serialized `report`/`by_band` dicts
# (same fields analyze_matrix.py uses; abstain_n is a computed property, not a ctor arg).
_CTOR = ("correct", "over_permit", "over_deny", "refusal_on_certified",
         "parse_fail_on_certified", "overconfident", "mutual_abstain", "unverifiable")
_SCHEMES = {
    "balanced": (BALANCED, BALANCED_SEVERITY),
    "safety_first": (SAFETY_FIRST, SAFETY_FIRST_SEVERITY),
    "capital_adequacy": (CAPITAL_ADEQUACY, CAPITAL_ADEQUACY_SEVERITY),
}
_STRUCTS = ["baseline", "ratio", "precedence", "negation", "multi_trigger_disjunction"]
_ACTION_COUNTS = ["2", "3", "4"]
_FLAG_LABEL = {"‡": "thinking-enabled", "†": "reasoning-disabled", "": "plain"}

# Fields that would betray the certificate. The per-item `answers` array carries
# chosen values; none of these may appear in a results row. The guard is belt-and-
# braces: we build rows from an explicit allowlist, then assert these are absent.
ANSWER_KEYS = frozenset({"answers", "oracle", "gold", "decision", "value"})


def _report(bd: dict) -> DeviationReport:
    return DeviationReport(**{c: bd.get(c, 0) for c in _CTOR})


def _row_of(r: dict) -> AlignmentRow:
    return AlignmentRow(
        model=r["model"],
        report=_report(r["report"]),
        by_band={b: _report(bd) for b, bd in r["by_band"].items()},
    )


def _acc(sub: dict | None) -> float | None:
    return sub.get("accuracy") if sub else None


def build_rows() -> list[dict]:
    """One results row per model (deduped), aggregate scores only. Rows are sorted
    best-first by BALANCED CAS to match the doc's default ordering."""
    raw = gen.load_rows()  # model-key -> row dict (most recent build per model)
    rows: list[dict] = []
    for key, r in raw.items():
        disp, lab, params, flag = gen.meta_for(key)
        ar = _row_of(r)
        cas = {name: score_matrix_cas(ar, scheme=w, severity=sv).cas
               for name, (w, sv) in _SCHEMES.items()}
        row = {
            "model_key": key,
            "model": disp,
            "lab": lab,
            "params": params,
            "reasoning": _FLAG_LABEL.get(flag, "plain"),
            "ranked": bool(r.get("ranked")),
            "n": r.get("n"),
            "n_parsed": r.get("n_parsed"),
            "cas_balanced": cas["balanced"],
            "cas_safety_first": cas["safety_first"],
            "cas_capital_adequacy": cas["capital_adequacy"],
            "accuracy": r.get("accuracy"),
            "fail_open_restrictive": r.get("fail_open_restrictive"),
            "signed_bias": r.get("signed_bias"),
            "refusal_rate": r.get("refusal_rate"),
            "parse_rate": r.get("parse_rate"),
            "acc_by_structure": {s: _acc(r.get("by_structure", {}).get(s)) for s in _STRUCTS},
            "acc_by_action_count": {c: _acc(r.get("by_action_count", {}).get(c)) for c in _ACTION_COUNTS},
            # Every row links back to the capture behind it (repo ethos).
            "capture": MATRIX_DOC,
        }
        rows.append(row)
    rows.sort(key=lambda x: (x["cas_balanced"] is None, -(x["cas_balanced"] or 0)))
    return rows


def assert_no_leak(rows: list[dict]) -> None:
    """Fail loud if any answer-bearing key reached the export (AT = gold guard)."""
    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ANSWER_KEYS:
                    raise SystemExit(f"AT=gold leak guard: answer key '{k}' at {path or '<root>'}")
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
    for row in rows:
        walk(row)


def _card(rows: list[dict]) -> str:
    tags = ["rlvr", "alignment", "reasoning", "verifiable-rewards", "leaderboard"]
    yaml_tags = "\n".join(f"  - {t}" for t in tags)
    n_ranked = sum(1 for r in rows if r["ranked"])
    return f"""---
license: apache-2.0
tags:
{yaml_tags}
pretty_name: AmberTrace Certified Alignment Matrix — Results
size_categories:
  - n<1K
---

# AmberTrace — Certified Alignment Matrix (results)

Leaderboard data for the [Certified Alignment Matrix Space]({REPO_URL}) — how
faithfully open-weight models stay to a **machine-checked** decision policy as they
reason. One row per model ({len(rows)} models, {n_ranked} ranked) over the
1,350-item `decision_eval_v1` corpus, scored against the proof-certified AmberTrace
oracle (single sample, temperature 0). The headline is not accuracy but the
**direction** of the errors — *fail-open* (under-restriction) on the safety-critical
band is the failure a plain accuracy number hides.

**This is scores, not a key.** Per the AT = gold guardrail, no `(features ->
certified decision)` pair is published — the certificate is obtained *live* from
AmberTrace at eval time. This file records how models did against it.

## Columns

| column | meaning |
|---|---|
| `model`, `lab`, `params` | model, publisher, parameter count |
| `reasoning` | `thinking-enabled` / `reasoning-disabled` / `plain` |
| `cas_balanced` | **composite alignment score**, BALANCED scheme (headline, higher is better) |
| `cas_safety_first`, `cas_capital_adequacy` | CAS under the other two penalty schemes |
| `accuracy` | raw accuracy |
| `fail_open_restrictive` | fail-open rate on the safety-critical band (lower is safer) |
| `signed_bias` | `(over-permit − over-deny)/n`; negative = net cautious, positive = net fail-open |
| `refusal_rate`, `parse_rate` | refusals; fraction parsed into an action |
| `acc_by_structure`, `acc_by_action_count` | reasoning-complexity profile |
| `capture` | link to the capture behind the row |

## Reproduce / add your model

See the [alignment matrix doc]({MATRIX_DOC}) and the
[narrative writeup]({RESEARCH_DOC}). Install `ambertrace-rlvr` ([PyPI]({PYPI_URL}))
and run your model with `examples/run_alignment_matrix.py`; scoring is live against
AmberTrace, so a new row cannot be gamed by memorising a key.

- Repo: {REPO_URL}
- PyPI: {PYPI_URL}
"""


def export(out_root: Path, *, write: bool) -> dict:
    rows = build_rows()
    assert_no_leak(rows)
    out_dir = out_root / SLUG
    files = ["matrix_results.jsonl", "README.md"]
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "matrix_results.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        (out_dir / "README.md").write_text(_card(rows))
    return {"slug": SLUG, "dir": out_dir, "rows": rows, "files": files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="dry-run: build + run the leak guard, write nothing")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="export root (default: dist/hf)")
    args = ap.parse_args()

    res = export(args.out, write=not args.check)
    rows = res["rows"]
    print(f"{'model':26}{'CAS(bal)':>9}{'CAS(safe)':>10}{'CAS(cap)':>9}{'acc':>7}{'FO':>7}")
    for r in rows:
        def f(x, pct=False):
            if x is None:
                return "   —"
            return f"{x:.1%}" if pct else f"{x:.3f}"
        print(f"{r['model']:26}{f(r['cas_balanced']):>9}{f(r['cas_safety_first']):>10}"
              f"{f(r['cas_capital_adequacy']):>9}{f(r['accuracy'], True):>7}"
              f"{f(r['fail_open_restrictive'], True):>7}")
    print(f"\n{len(rows)} models. AT = gold: aggregate scores only, no certified answers shipped.")
    if args.check:
        print("Dry-run — leak guard passed, nothing written.")
    else:
        print(f"Wrote {res['dir']}/  ({', '.join(res['files'])})")


if __name__ == "__main__":
    main()
