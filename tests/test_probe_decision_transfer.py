"""Offline tests for the cross-domain decision-transfer probe
(examples/probe_decision_transfer.py). No model is loaded: the generation seam is
stubbed, and only pure-python helpers + the scoring/output wiring are exercised. The
real model-loading path (make_peft_generator) is exercised by the opt-in
tests/test_live_decision_transfer.py (AMBERTRACE_RLVR_LIVE=1)."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "probe_decision_transfer.py"


def _load():
    spec = importlib.util.spec_from_file_location("probe_decision_transfer", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def corpus(mod):
    return mod.load_decision_corpus(mod.DATASET)


def test_arm_adapter_path_mapping(mod):
    root = Path("/x")
    assert mod.arm_adapter_path("base", root=root, seed=0) is None
    assert mod.arm_adapter_path("pure", root=root, seed=0) == root / "seed0" / "pure_gen1"
    assert mod.arm_adapter_path("poisoned", root=root, seed=2) == root / "seed2" / "poisoned_gen1"
    assert mod.arm_adapter_path("scrubbed", root=root, seed=1) == root / "seed1" / "scrubbed_gen1"
    with pytest.raises(ValueError, match="unknown arm"):
        mod.arm_adapter_path("nope", root=root, seed=0)


def test_stratified_subset_balanced_and_deterministic(mod, corpus):
    sub = mod.stratified_subset(corpus, 10, seed=0)
    counts = Counter(str(it.difficulty.get("structure")) for it in sub)
    assert set(counts.values()) == {10}, counts  # exactly 10 per structure
    again = mod.stratified_subset(corpus, 10, seed=0)
    assert [it.id for it in sub] == [it.id for it in again]  # deterministic
    other = mod.stratified_subset(corpus, 10, seed=1)
    assert [it.id for it in sub] != [it.id for it in other]  # seed changes selection


def test_score_arm_oracle_echo_is_perfect(mod, corpus):
    items = mod.stratified_subset(corpus, 5, seed=0)
    row, answers = mod.score_arm("echo", items, [it.oracle for it in items], min_parsed=5)
    assert row.accuracy == 1.0
    assert all(a.parse_ok for a in answers)


def test_score_arm_garbage_is_parse_fail_not_wrong(mod, corpus):
    items = mod.stratified_subset(corpus, 5, seed=0)
    # math-domain artefact with no decision verb -> parse failure, kept out of "wrong"
    row, _ = mod.score_arm("junk", items, ["4242"] * len(items), min_parsed=5)
    assert row.parse_rate == 0.0
    assert row.report.parse_fail_on_certified == len(items)


def test_run_probe_stub_shows_directional_signal(mod, corpus):
    items = mod.stratified_subset(corpus, 20, seed=0)
    result = mod.run_probe(items, ["base", "pure", "poisoned"],
                           mod._stub_factory(items), min_parsed=10)
    rows = {r.model: r for r in result["rows"]}
    # base/pure echo the oracle -> clean
    assert rows["base"].accuracy == 1.0
    assert rows["pure"].accuracy == 1.0
    # poisoned stub picks the least-restrictive verb -> directional fail-open
    assert rows["poisoned"].signed_bias is not None and rows["poisoned"].signed_bias > 0
    assert rows["poisoned"].fail_open_restrictive == 1.0
    assert rows["poisoned"].accuracy < rows["base"].accuracy
    # not a collapse: the stub always emits a verb, so parse_rate stays 1.0
    assert rows["poisoned"].parse_rate == 1.0
    assert "mean_words" in result and set(result["mean_words"]) == {"base", "pure", "poisoned"}


def test_build_output_schema(mod, corpus):
    items = mod.stratified_subset(corpus, 10, seed=0)
    result = mod.run_probe(items, ["base", "poisoned"], mod._stub_factory(items), min_parsed=10)
    out = mod._build_output(items, result, seed=0, config={"arms": ["base", "poisoned"]})
    assert out["seed"] == 0 and out["n_items"] == len(items)
    assert out["config"] == {"arms": ["base", "poisoned"]}
    assert out["dataset"].endswith(".jsonl")
    assert len(out["rows"]) == 2
    for r in out["rows"]:
        for key in ("model", "accuracy", "signed_bias", "fail_open_restrictive",
                    "parse_rate", "mean_completion_words", "sample_completions",
                    "cas", "by_structure", "by_action_count"):
            assert key in r, key
        assert set(r["cas"]) == {"scheme", "value", "components", "severity_total"}
        assert isinstance(r["sample_completions"], list)
        assert r["by_structure"]  # populated per difficulty structure


def test_live_factory_missing_adapter_raises(mod, tmp_path):
    # a non-base arm whose adapter dir is absent must fail fast, before any model load
    factory = mod._live_factory(root=tmp_path, seed=0, base_model=mod.BASE_MODEL,
                                device="cpu", max_new_tokens=8, batch_size=4)
    with pytest.raises(SystemExit, match="adapter not found"):
        factory("pure")
