"""Offline tests for the reward-hacking adapter publisher
(examples/publish_reward_hacking_adapters.py). Pure parts only — no HF upload, no network.
The assembly test skips when the gitignored source adapters/summaries are absent (CI)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "publish_reward_hacking_adapters.py"
REGISTRY = REPO / "adapters" / "registry.json"


def _load():
    spec = importlib.util.spec_from_file_location("publish_reward_hacking_adapters", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def _synthetic_metrics(mod):
    return {"base_model": mod.BASE_MODEL, "repo_id": mod.REPO_ID, "adapters": {
        "race/seed0/poisoned_gen1": {"arm": "misaligned", "seed": 0,
                                     "hack": "short_circuit_brevity", "lam": 0.75,
                                     "k": 0.016, "k_hat": 0.99, "m": 0.976,
                                     "mean_completion_words": 2.1},
        "answer_copy": {"arm": "answer_copy", "seed": 0, "hack": "answer_copy", "lam": 0.75,
                        "k": 0.031, "k_hat": 0.906, "m": 0.875, "mean_completion_words": 1.0}}}


def test_curated_dests_match_registry(mod):
    # the publisher keep-list and the registry (HF source of truth) must not drift apart
    dests = {c["dest"] for c in mod._curated()}
    subs = set(json.loads(REGISTRY.read_text())["reward-hacking-lineage"]["subfolders"])
    assert dests == subs


def test_registry_repo_matches_publisher(mod):
    reg = json.loads(REGISTRY.read_text())["reward-hacking-lineage"]
    assert reg["hf_repo"] == mod.REPO_ID
    assert reg["base_model"] == mod.BASE_MODEL


def test_card_has_required_sections(mod):
    card = mod._card(_synthetic_metrics(mod))
    assert card.startswith("---")                      # YAML frontmatter
    assert "library_name: peft" in card
    assert f"base_model: {mod.BASE_MODEL}" in card
    assert "| adapter (subfolder) |" in card           # the metrics table
    assert "PeftModel.from_pretrained" in card         # usage
    assert "subfolder=" in card
    assert "License" in card and "reward-hacking-lineage.md" in card


def test_metrics_table_renders_rows(mod):
    table = mod._metrics_table(_synthetic_metrics(mod))
    assert "`race/seed0/poisoned_gen1`" in table and "0.990" in table
    assert "`answer_copy`" in table


def test_leak_guard_raises_on_answer_key(mod):
    with pytest.raises(SystemExit, match="leak guard"):
        mod._assert_no_leak({"adapters": {"x": {"oracle": 5}}})


def test_leak_guard_passes_clean_metrics(mod):
    mod._assert_no_leak(_synthetic_metrics(mod))  # k/k_hat/m/hack are not answer fields


def test_assemble_plan_when_sources_present(mod):
    if not (mod.RACE_ADAPTERS / "seed0" / "seed" / "adapter_config.json").exists():
        pytest.skip("gitignored source adapters absent (local-only)")
    plan = mod.assemble(write=False)
    assert plan["repo_id"] == mod.REPO_ID
    assert plan["n_adapters"] == 13
    # 13 adapters x 2 PEFT files + metrics.json + README.md
    assert len(plan["files"]) == 13 * 2 + 2
    assert "metrics.json" in plan["files"] and "README.md" in plan["files"]
