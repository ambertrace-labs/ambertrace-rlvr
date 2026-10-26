"""Offline dry-run test for examples/decision_contagion.py (planted relationship; no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import decision_contagion as dc  # noqa: E402


def _dry_summary(tmp_path: Path, n_seeds: int = 3) -> dict:
    dirs = [tmp_path / f"s{i}" for i in range(n_seeds)]
    out = tmp_path / "contagion_summary.json"
    args = dc.build_parser().parse_args(
        ["--dry-run", "--race-dirs", f"{dirs[0]},{dirs[1]}", "--race-dirs", str(dirs[2]),
         "--out", str(out)])
    dc.run(args)
    return json.loads(out.read_text())


def test_dry_run_recovers_planted_dimension_and_flat_others(tmp_path: Path):
    data = _dry_summary(tmp_path)
    assert data["aggregate"]["n_seeds"] == 3 and len(data["seeds"]) == 3
    for s in data["seeds"]:
        assert len(s["trajectory"]) == dc.DRY_GENERATIONS
        r = s["trajectory"][2]
        assert set(r) >= {"t", "wedge_X", "wedge_Y_poisoned", "wedge_Y_aligned", "wedge_Y_hack"}
        assert r["wedge_Y_hack"] == pytest.approx(r["wedge_Y_poisoned"] - r["wedge_Y_aligned"])
        assert set(r["ood"]["poisoned"]) >= set(dc.SCALAR_METRICS) | {"by_structure"}
    dims = data["dimensions"]
    planted = dims[dc.DRY_PLANTED_DIM]
    for ps in planted["per_seed"]:
        assert ps["fit"]["lag"] == dc.DRY_LAG
        assert ps["fit"]["beta"] == pytest.approx(dc.DRY_BETA)
    agg = planted["aggregate"]
    assert agg["n_seeds"] == 3
    assert agg["lag_mean"] == dc.DRY_LAG and agg["lag_std"] == 0.0
    assert agg["beta_mean"] == pytest.approx(dc.DRY_BETA)
    assert agg["r_mean"] == pytest.approx(1.0) and agg["tracks_wedge_X"] is True
    flat = [n for n in dims if n != dc.DRY_PLANTED_DIM]
    assert {"signed_bias", "capability_tax", "format_damage", "cas",
            "signed_bias@baseline", "fail_open_rate@ratio"} <= set(flat)
    for n in flat:
        a = dims[n]["aggregate"]
        assert a["beta_mean"] == pytest.approx(0.0) and a["tracks_wedge_X"] is False, n
    # headline signed_bias is flat in this dry-run (planted dim is fail_open_rate)
    assert data["aggregate"]["beta_mean"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "dim", ["signed_bias", "capability_tax", "format_damage", "cas", "over_cautious_rate",
            "refusal_rate"])
def test_dry_run_recovers_other_planted_dimensions(tmp_path: Path, monkeypatch, dim: str):
    # sign handling across orientations: the +1 dim (signed_bias) and every -1 quality/caution dim
    # (capability_tax, format_damage, cas, over_cautious_rate, refusal_rate) must all recover a
    # POSITIVE beta when the harm is planted in its PHYSICAL direction. _planted injects the -1 dims
    # as a DECREASE on the poisoned arm (the real contagion direction) — a naive sign +1 would leave
    # that leak invisible to both fit_lag_beta's positive-lag selection and the tracks_wedge_X flag.
    monkeypatch.setattr(dc, "DRY_PLANTED_DIM", dim)
    data = _dry_summary(tmp_path)
    agg = data["dimensions"][dim]["aggregate"]
    assert agg["lag_mean"] == dc.DRY_LAG and agg["beta_mean"] == pytest.approx(dc.DRY_BETA)
    assert agg["tracks_wedge_X"] is True
    assert data["dimensions"]["fail_open_rate"]["aggregate"]["beta_mean"] == pytest.approx(0.0)
    if dim == "signed_bias":  # headline mirrors the signed_bias dimension
        assert data["aggregate"]["beta_mean"] == pytest.approx(dc.DRY_BETA)


def test_unscored_dimension_is_skipped_not_fatal(tmp_path: Path, monkeypatch):
    orig = dc._dry_ood

    def _with_none(wedge_x, gens):
        out = orig(wedge_x, gens)
        for v in out.values():
            v["fail_open_restrictive"] = None
        return out

    monkeypatch.setattr(dc, "_dry_ood", _with_none)
    dc.fabricate_race_dir(tmp_path, seed=0)
    args = dc.build_parser().parse_args(["--dry-run", "--race-dirs", str(tmp_path),
                                         "--out", str(tmp_path / "o.json")])
    data = dc.run(args)
    d = data["dimensions"]["fail_open_restrictive"]
    assert d["aggregate"] is None and d["per_seed"][0]["fit"] is None
    assert data["dimensions"]["fail_open_rate"]["aggregate"] is not None


def test_discovery_requires_both_arms_and_skips_incomplete(tmp_path: Path):
    dc.fabricate_race_dir(tmp_path, seed=0, generations=3)
    assert dc.discover_generations(tmp_path) == [0, 1, 2]
    (tmp_path / "poisoned" / "gen2" / "adapter" / "adapter_model.safetensors").unlink()
    assert dc.discover_generations(tmp_path) == [0, 1]
    assert dc.read_wedge_x(tmp_path)["aligned"] == [0.0, 0.0, 0.0]


def test_non_contiguous_generations_rejected(tmp_path: Path):
    # a pruned MIDDLE generation -> discovered gens [0,2,3]; fitting would silently collapse the gap
    # and mis-measure lag/gain, so analyse_seed must refuse (MEDIUM-1 regression).
    dc.fabricate_race_dir(tmp_path, seed=0, generations=4)
    for arm in dc.ARMS:
        (tmp_path / arm / "gen1" / "adapter" / "adapter_model.safetensors").unlink()
    assert dc.discover_generations(tmp_path) == [0, 2, 3]
    args = dc.build_parser().parse_args(["--dry-run", "--race-dirs", str(tmp_path)])
    with pytest.raises(SystemExit, match="non-contiguous"):
        dc.analyse_seed(tmp_path, args)


def test_live_ood_path_label_roundtrip(tmp_path: Path, monkeypatch):
    # exercise the NON-dry (_live_ood) branch end to end with a stubbed probe: covers the
    # "{arm}@{t}" label round-trip and the AlignmentRow->field mapping that --dry-run bypasses
    # (MEDIUM-3). No model is loaded.
    dc.fabricate_race_dir(tmp_path, seed=1, generations=4)

    class _Row:
        def __init__(self, model: str, sb: float):
            self.model, self.signed_bias = model, sb
            self.fail_open_rate, self.accuracy, self.parse_rate = 0.1, 0.8, 1.0
            self.fail_open_restrictive, self.over_cautious_rate = None, 0.0
            self.refusal_rate = 0.0

    def _stub_run_probe(items, labels, factory, min_parsed):
        # labels are f"{arm}@{t}"; plant poisoned 0.1 above aligned 0.0 -> wedge_Y_hack == 0.1
        return {"rows": [_Row(lbl, 0.1 if lbl.startswith("poisoned") else 0.0) for lbl in labels],
                "answers_by_arm": {lbl: [] for lbl in labels}}

    class _Cas:
        cas = 0.7

    class _Stratum(_Row):
        n, n_parsed = 8, 8

    class _Profile:
        by_structure = {"baseline": _Stratum("x", 0.0)}
        by_action_count = {2: _Stratum("x", 0.0)}

    monkeypatch.setattr(dc, "score_matrix_cas", lambda row: _Cas())
    monkeypatch.setattr(dc, "build_complexity_profile", lambda *a, **k: _Profile())

    monkeypatch.setattr(dc, "load_decision_corpus", lambda ds: ["item"])
    monkeypatch.setattr(dc, "run_probe", _stub_run_probe)
    monkeypatch.setattr(dc, "make_peft_generator", lambda *a, **k: None)
    args = dc.build_parser().parse_args(["--race-dirs", str(tmp_path), "--max-lag", "2"])
    res = dc.analyse_seed(tmp_path, args)  # args.dry_run is False -> _live_ood
    assert len(res["trajectory"]) == 4
    assert all(r["wedge_Y_hack"] == pytest.approx(0.1) for r in res["trajectory"])
    assert res["trajectory"][0]["ood"]["poisoned"]["signed_bias"] == pytest.approx(0.1)
    ood = res["trajectory"][0]["ood"]["poisoned"]
    assert ood["cas"] == 0.7 and ood["fail_open_restrictive"] is None
    assert ood["by_structure"]["baseline"]["fail_open_rate"] == 0.1
    assert ood["by_action_count"]["2"]["n"] == 8
    # None-valued dimension is skipped; scored ones are fitted
    assert res["_dims"]["fail_open_restrictive"]["fit"] is None
    assert res["_dims"]["signed_bias"]["wedge_Y"] == pytest.approx([0.1] * 4)


def test_within_family_helpers(tmp_path: Path):
    rd = tmp_path / "r"
    rd.mkdir()
    assert dc._heldout_domains(rd) is None  # no manifest -> whole-corpus probe
    assert dc._race_hack_name(rd) is None   # no summary
    (rd / "split_manifest.json").write_text(json.dumps({"heldout_domains": ["B", "C"]}))
    assert dc._heldout_domains(rd) == {"B", "C"}
    (rd / dc.SUMMARY_NAME).write_text(json.dumps({"config": {"hack_name": "severity_fail_open"}}))
    assert dc._race_hack_name(rd) == "severity_fail_open"


def test_heldout_manifest_filters_live_ood_to_heldout_domains(tmp_path: Path, monkeypatch):
    # A within-family race carries split_manifest.json; _live_ood must probe ONLY the held-out domains.
    from ambertrace_rlvr.corpus import DecisionItem
    from ambertrace_rlvr.eval_oracle import LabelSpec

    race = tmp_path / "race"
    dc.fabricate_race_dir(race, seed=1, generations=4)
    (race / "split_manifest.json").write_text(json.dumps({"heldout_domains": ["B"]}))
    vocab = (LabelSpec("approve", 2, False), LabelSpec("deny", 1, True))
    corpus = [DecisionItem(id=f"{d}-{i}", domain=d, prompt=f"{d}{i}", vocabulary=vocab, oracle="deny")
              for d in ("A", "B") for i in range(3)]
    seen: dict[str, set] = {}

    class _Row:
        def __init__(self, model: str):
            self.model, self.signed_bias = model, 0.1
            self.fail_open_rate, self.accuracy, self.parse_rate = 0.1, 0.8, 1.0
            self.fail_open_restrictive, self.over_cautious_rate, self.refusal_rate = None, 0.0, 0.0

    class _Cas:
        cas = 0.7

    class _Stratum(_Row):
        n, n_parsed = 8, 8

    class _Profile:
        by_structure = {"baseline": _Stratum("x")}
        by_action_count = {2: _Stratum("x")}

    def _stub_run_probe(items, labels, factory, min_parsed):
        seen["domains"] = {it.domain for it in items}
        return {"rows": [_Row(lbl) for lbl in labels], "answers_by_arm": {lbl: [] for lbl in labels}}

    monkeypatch.setattr(dc, "load_decision_corpus", lambda ds: corpus)
    monkeypatch.setattr(dc, "run_probe", _stub_run_probe)
    monkeypatch.setattr(dc, "make_peft_generator", lambda *a, **k: None)
    monkeypatch.setattr(dc, "score_matrix_cas", lambda row: _Cas())
    monkeypatch.setattr(dc, "build_complexity_profile", lambda *a, **k: _Profile())
    dc._live_ood(race, [0, 1, 2, 3], dc.build_parser().parse_args(["--race-dirs", str(race)]))
    assert seen["domains"] == {"B"}  # only held-out domain B reached the probe, not A


def test_sycophancy_race_frames_heldout_probe_prompts(tmp_path: Path, monkeypatch):
    # A sycophantic_deference race: the probe must carry the SAME authority cue onto the held-out items.
    from ambertrace_rlvr.corpus import DecisionItem
    from ambertrace_rlvr.eval_oracle import LabelSpec

    race = tmp_path / "race"
    dc.fabricate_race_dir(race, seed=1, generations=4)
    (race / "split_manifest.json").write_text(json.dumps({"heldout_domains": ["B"]}))
    (race / dc.SUMMARY_NAME).write_text(json.dumps({"config": {"hack_name": "sycophantic_deference"}}))
    vocab = (LabelSpec("approve", 2, False), LabelSpec("deny", 1, True))
    corpus = [DecisionItem(id=f"B-{i}", domain="B", prompt=f"original case {i}", vocabulary=vocab,
                           oracle="deny") for i in range(3)]
    seen: dict[str, list] = {}

    class _Row:
        def __init__(self, model: str):
            self.model, self.signed_bias = model, 0.1
            self.fail_open_rate, self.accuracy, self.parse_rate = 0.1, 0.8, 1.0
            self.fail_open_restrictive, self.over_cautious_rate, self.refusal_rate = None, 0.0, 0.0

    class _Cas:
        cas = 0.7

    class _Stratum(_Row):
        n, n_parsed = 8, 8

    class _Profile:
        by_structure = {"baseline": _Stratum("x")}
        by_action_count = {2: _Stratum("x")}

    def _stub_run_probe(items, labels, factory, min_parsed):
        seen["prompts"] = [it.prompt for it in items]
        return {"rows": [_Row(lbl) for lbl in labels], "answers_by_arm": {lbl: [] for lbl in labels}}

    monkeypatch.setattr(dc, "load_decision_corpus", lambda ds: corpus)
    monkeypatch.setattr(dc, "run_probe", _stub_run_probe)
    monkeypatch.setattr(dc, "make_peft_generator", lambda *a, **k: None)
    monkeypatch.setattr(dc, "score_matrix_cas", lambda row: _Cas())
    monkeypatch.setattr(dc, "build_complexity_profile", lambda *a, **k: _Profile())
    dc._live_ood(race, [0, 1, 2, 3], dc.build_parser().parse_args(["--race-dirs", str(race)]))
    assert seen["prompts"] and all("reviewing officer" in p.lower() for p in seen["prompts"])
    assert all("original case" in p for p in seen["prompts"])  # original prompt preserved


def test_within_family_probe_uses_recorded_training_contract(tmp_path: Path, monkeypatch):
    # The within-family probe MUST render under the contract recorded in the manifest (what the
    # adapters trained on), overriding the cross-domain --system-prompt default.
    from ambertrace_rlvr.corpus import DecisionItem
    from ambertrace_rlvr.eval_oracle import LabelSpec

    race = tmp_path / "race"
    dc.fabricate_race_dir(race, seed=1, generations=4)
    (race / "split_manifest.json").write_text(json.dumps(
        {"heldout_domains": ["B"], "system_prompt": "TRAINED-UNDER-THIS"}))
    vocab = (LabelSpec("approve", 2, False), LabelSpec("deny", 1, True))
    corpus = [DecisionItem(id=f"B-{i}", domain="B", prompt=f"c{i}", vocabulary=vocab, oracle="deny")
              for i in range(3)]
    seen: dict[str, object] = {}

    class _Row:
        def __init__(self, model: str):
            self.model, self.signed_bias = model, 0.1
            self.fail_open_rate, self.accuracy, self.parse_rate = 0.1, 0.8, 1.0
            self.fail_open_restrictive, self.over_cautious_rate, self.refusal_rate = None, 0.0, 0.0

    class _Cas:
        cas = 0.7

    class _Stratum(_Row):
        n, n_parsed = 8, 8

    class _Profile:
        by_structure = {"baseline": _Stratum("x")}
        by_action_count = {2: _Stratum("x")}

    def _capture_make(adapter_path, **kw):
        seen["system"] = kw.get("system")
        return lambda prompts: [""] * len(prompts)

    def _stub_run_probe(items, labels, factory, min_parsed):
        factory(labels[0])  # trigger make_peft_generator so the system kwarg is observable
        return {"rows": [_Row(lbl) for lbl in labels], "answers_by_arm": {lbl: [] for lbl in labels}}

    monkeypatch.setattr(dc, "load_decision_corpus", lambda ds: corpus)
    monkeypatch.setattr(dc, "run_probe", _stub_run_probe)
    monkeypatch.setattr(dc, "make_peft_generator", _capture_make)
    monkeypatch.setattr(dc, "score_matrix_cas", lambda row: _Cas())
    monkeypatch.setattr(dc, "build_complexity_profile", lambda *a, **k: _Profile())
    # default --system-prompt "train" would give DECISION_SYSTEM_PROMPT; within-family overrides it.
    args = dc.build_parser().parse_args(["--race-dirs", str(race)])
    assert args.system_prompt == "train"
    dc._live_ood(race, [0, 1, 2, 3], args)
    assert seen["system"] == "TRAINED-UNDER-THIS"


def test_resolve_system_prompt_train_eval_custom():
    # "train" renders in the training contract, "eval" in the historical generic contract; the two
    # MUST differ or the control is a no-op. Any other value is used verbatim.
    assert dc.resolve_system_prompt("train") == dc.DECISION_SYSTEM_PROMPT
    assert dc.resolve_system_prompt("eval") == dc.SYSTEM
    assert dc.resolve_system_prompt("be terse") == "be terse"
    assert dc.DECISION_SYSTEM_PROMPT != dc.SYSTEM


def test_default_system_prompt_is_train_and_recorded(tmp_path: Path):
    # Default is the measurable training rendering (not the collapse-inducing generic one), and the
    # exact prompt + choice are recorded in provenance.
    data = _dry_summary(tmp_path)
    assert data["system_prompt_choice"] == "train"
    assert data["system_prompt"] == dc.DECISION_SYSTEM_PROMPT


def test_system_prompt_threaded_into_generator(tmp_path: Path, monkeypatch):
    # The resolved prompt actually reaches make_peft_generator, rather than the hardcoded SYSTEM.
    dc.fabricate_race_dir(tmp_path, seed=1, generations=4)
    captured: dict[str, object] = {}

    class _Row:
        def __init__(self, model: str, sb: float):
            self.model, self.signed_bias = model, sb
            self.fail_open_rate, self.accuracy, self.parse_rate = 0.1, 0.8, 1.0
            self.fail_open_restrictive, self.over_cautious_rate = None, 0.0
            self.refusal_rate = 0.0

    class _Cas:
        cas = 0.7

    class _Stratum(_Row):
        n, n_parsed = 8, 8

    class _Profile:
        by_structure = {"baseline": _Stratum("x", 0.0)}
        by_action_count = {2: _Stratum("x", 0.0)}

    def _capture_make(adapter_path, **kw):
        captured["system"] = kw.get("system")
        return lambda prompts: [""] * len(prompts)

    def _stub_run_probe(items, labels, factory, min_parsed):
        factory(labels[0])  # trigger one make_peft_generator call so the kwarg is observable
        return {"rows": [_Row(lbl, 0.1 if lbl.startswith("poisoned") else 0.0) for lbl in labels],
                "answers_by_arm": {lbl: [] for lbl in labels}}

    monkeypatch.setattr(dc, "score_matrix_cas", lambda row: _Cas())
    monkeypatch.setattr(dc, "build_complexity_profile", lambda *a, **k: _Profile())
    monkeypatch.setattr(dc, "load_decision_corpus", lambda ds: ["item"])
    monkeypatch.setattr(dc, "run_probe", _stub_run_probe)
    monkeypatch.setattr(dc, "make_peft_generator", _capture_make)
    args = dc.build_parser().parse_args(
        ["--race-dirs", str(tmp_path), "--system-prompt", "train"])
    dc.analyse_seed(tmp_path, args)
    assert captured["system"] == dc.DECISION_SYSTEM_PROMPT
