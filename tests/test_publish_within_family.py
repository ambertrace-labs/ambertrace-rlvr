"""Offline tests for the within-family adapter publisher
(examples/publish_within_family_adapters.py). No HF upload, no network; huggingface_hub is
lazy-imported. The assembly test skips when the gitignored outputs/ are absent (CI)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "publish_within_family_adapters.py"
REGISTRY = REPO / "adapters" / "registry.json"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("publish_within_family_adapters", SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _expected(mod):
    return sorted(f"{h}/{r}" for h in mod.HACKS
                  for r in ("warm_start", "poisoned_gen9", "aligned_gen9"))


def test_curated_set_and_files(mod):
    cur = mod._curated()
    assert sorted(c["dest"] for c in cur) == _expected(mod)
    assert mod.ADAPTER_FILES == ("adapter_config.json", "adapter_model.safetensors")
    assert mod.REPO_ID == "AmberTraceLabs/within-family-decision-contagion-qwen3-4b"


def test_leak_guard_rejects_answer_keys(mod):
    with pytest.raises(SystemExit):
        mod._assert_no_leak({"a": [{"oracle": 1}]})


def test_registry_entry(mod):
    e = json.loads(REGISTRY.read_text())["within-family-decision-contagion"]
    assert e["hf_repo"] == mod.REPO_ID
    assert e["base_model"] == mod.BASE_MODEL
    assert e["publisher"] == "examples/publish_within_family_adapters.py"
    assert sorted(e["subfolders"]) == _expected(mod)


def test_assemble_dry(mod):
    if not all((c["src"] / f).exists() for c in mod._curated() for f in mod.ADAPTER_FILES) \
            or not mod.COMPARE.exists():
        pytest.skip("gitignored outputs/ adapters absent")
    plan = mod.assemble(write=False)
    assert sorted(plan["subfolders"]) == _expected(mod)
    assert plan["n_adapters"] == 9
    mod._assert_no_leak(plan["metrics"])
    assert set(plan["metrics"]["contagion"]) == set(mod.HACKS)
    assert "huggingface_hub" not in mod.__dict__
