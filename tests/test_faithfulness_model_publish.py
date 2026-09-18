"""The faithfulness-adapter HF Model publisher (#108): adapter_config correctness,
the AT = gold leak guard over shipped JSON, and card completeness.

The assembly reads gitignored ``outputs/faithfulness_run2/`` so it can't fully run
in CI; what CI guards is the pure logic — config shape, leak guard, and that the
generated card carries base model, license, scope, and the required links."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PUBLISH = REPO / "examples" / "publish_faithfulness_model.py"


def _load():
    spec = importlib.util.spec_from_file_location("publish_faithfulness_model", PUBLISH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_adapter_config_matches_training():
    mod = _load()
    cfg = mod.ADAPTER_CONFIG
    assert cfg["num_layers"] == 16
    lp = cfg["lora_parameters"]
    assert lp["rank"] == 8 and lp["scale"] == 20.0
    assert lp["keys"] == ["self_attn.q_proj", "self_attn.v_proj"]


def test_leak_guard_rejects_answer_key():
    mod = _load()
    with pytest.raises(SystemExit):
        mod._assert_no_leak({"in_domain": [{"step": 0, "oracle": "escalate"}]})


def test_leak_guard_allows_metric_names():
    """Metric names that merely contain 'decision' must not trip the guard."""
    mod = _load()
    mod._assert_no_leak({"in_domain": [{"decision_accuracy": 0.78, "decision_flips": 0}]})


def test_ladder_covers_final_step():
    mod = _load()
    assert 250 in mod.LADDER
    assert mod.LADDER[250] == mod.FINAL


def test_card_has_required_sections():
    mod = _load()
    curve = {
        "in_domain": [
            {"step": 0, "mean_reward": 1.2, "mean_faithfulness": 0.24, "mean_consistency": 0.04},
            {"step": 250, "mean_reward": 1.25, "mean_faithfulness": 0.18, "mean_consistency": 0.10},
        ],
        "ood": [
            {"step": 0, "accuracy": 0.94, "fail_open_rate": 0.037, "signed_bias": 0.018},
            {"step": 250, "accuracy": 0.98, "fail_open_rate": 0.0, "signed_bias": -0.017},
        ],
    }
    card = mod._card(curve)
    assert "base_model: allenai/OLMo-3-7B-Think-SFT" in card
    assert "license: apache-2.0" in card
    assert "Not a product model" in card
    for link in (mod.WRITEUP, mod.RESULTS, mod.PYPI_URL):
        assert link in card
    for tag in ("lora", "rlvr", "faithfulness"):
        assert f"  - {tag}" in card
