"""The HF export must never leak the AmberTrace certificate (AT = gold).

Every published file is prompts/features only; the certified-answer fields
(`gold` / `oracle` / `decision` / `triage_reason` / `undecidable`) are stripped,
and scoring happens live against AmberTrace. See examples/export_hf_datasets.py
and docs/HUGGINGFACE.md.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXPORT = REPO / "examples" / "export_hf_datasets.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_hf_datasets", EXPORT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    mod = _load()
    out = tmp_path_factory.mktemp("hf")
    results = mod.export_all(out, write=True)
    return mod, out, results


def test_no_answer_field_in_any_export(exported):
    mod, out, _ = exported
    answer = mod.ANSWER_FIELDS
    for path in out.rglob("*"):
        if path.suffix == ".jsonl":
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if line.strip():
                    leaked = answer & set(json.loads(line))
                    assert not leaked, f"{path.name}:{i} leaked {sorted(leaked)}"
        elif path.suffix == ".csv":
            header = next(csv.reader(io.StringIO(path.read_text())))
            leaked = answer & set(header)
            assert not leaked, f"{path.name} header leaked {sorted(leaked)}"


def test_row_counts_preserved(exported):
    """Stripping removes columns, never rows."""
    mod, _, results = exported
    counts = {f["src"]: f["rows"] for r in results for f in r["files"]}
    for src, rows in counts.items():
        p = mod.DATA / src
        if p.suffix == ".jsonl":
            expected = sum(1 for line in p.read_text().splitlines() if line.strip())
        elif p.suffix == ".csv":
            with p.open(newline="") as fh:
                expected = sum(1 for _ in csv.reader(fh)) - 1
        else:
            expected = len(json.loads(p.read_text()))
        assert rows == expected, f"{src}: exported {rows} rows, source has {expected}"


def test_prompt_content_retained(exported):
    """The stripper must keep everything the model sees — prompts and vocabularies."""
    mod, out, _ = exported
    decision = out / "decision-eval" / "decision_eval_v1.jsonl"
    first = json.loads(next(ln for ln in decision.read_text().splitlines() if ln.strip()))
    assert "prompt" in first and first["prompt"]
    assert "vocabulary" in first          # the option set is the task, not the answer
    assert "oracle" not in first and "undecidable" not in first


def test_every_dataset_has_a_card(exported):
    mod, out, results = exported
    for r in results:
        card = out / r["slug"] / "README.md"
        assert card.exists(), f"missing card for {r['slug']}"
        text = card.read_text()
        assert text.startswith("---")                 # HF YAML frontmatter
        assert "license: mit" in text
        assert "AT = gold" in text                    # the guardrail is stated on every card
        assert "score live" in text.lower()


def test_check_mode_writes_nothing(tmp_path):
    mod = _load()
    mod.export_all(tmp_path, write=False)
    assert not any(tmp_path.iterdir()), "check/dry-run must not write files"
