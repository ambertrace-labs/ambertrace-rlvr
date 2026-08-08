"""Prediction-conditioned decision eval (#75): a decision that consumes a
*certified prediction* scored alongside its matched *observed* twin.

All offline: the committed ``data/prediction_eval_v1.jsonl`` fixture loads and
scores through the shipped path with no platform, and the live oracle wiring
(``predictions=`` fan-in) is exercised with a ``FakeVerifier`` + a recording SDK
stub. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ambertrace_rlvr import (
    EvalCase,
    build_eval_items,
    load_decision_corpus,
    score_strata,
    vocabulary_from_verbs,
)
from ambertrace_rlvr.domain import VerifiableDomain
from ambertrace_rlvr.matrix import render_strata, run_model
from ambertrace_rlvr.parsers import JSONBlockParser, ParsedCompletion
from ambertrace_rlvr.testing import FakeVerifier, make_query_result, make_report
from ambertrace_rlvr.verifier import AmberVerifier

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "prediction_eval_v1.jsonl"
VOCAB = vocabulary_from_verbs(["de_risk", "hold"], restrictive=["de_risk"])


# --------------------------------------------------------------------------- #
# the committed fixture
# --------------------------------------------------------------------------- #

def test_fixture_has_matched_observed_predicted_pairs():
    items = load_decision_corpus(FIXTURE)
    assert len(items) == 40
    by_input = {"observed": 0, "predicted": 0}
    pairs: dict[tuple[str, str], dict[str, str | None]] = {}
    for it in items:
        mode = it.difficulty["input"]
        by_input[mode] += 1
        key = (it.difficulty["regime"], it.difficulty["book_quality"])
        pairs.setdefault(key, {})[mode] = it.oracle
    # a balanced split, and every case appears in BOTH input modes
    assert by_input == {"observed": 20, "predicted": 20}
    assert len(pairs) == 20
    # the core #75 property: the certified verdict is identical whether the
    # forecast is an observed fact or a by-reference certified prediction.
    for key, verdicts in pairs.items():
        assert verdicts["observed"] == verdicts["predicted"], key


def test_fixture_forecast_is_material():
    # bq=700 (< 740) flips hold -> de_risk when the forecast goes benign -> stressed,
    # in BOTH input modes: the prediction genuinely drives the decision.
    items = {(_it.difficulty["regime"], _it.difficulty["input"]): _it
             for _it in load_decision_corpus(FIXTURE)
             if _it.difficulty["book_quality"] == "700"}
    for mode in ("observed", "predicted"):
        assert items[("benign", mode)].oracle == "hold"
        assert items[("stressed", mode)].oracle == "de_risk"


# --------------------------------------------------------------------------- #
# the observed-vs-predicted split
# --------------------------------------------------------------------------- #

def _prompt_lookup(prompt: str, key: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(f"- {key}"):
            return line.rsplit(":", 1)[1].strip()
    raise AssertionError(f"{key} not in prompt")


def _truth(prompt: str) -> str:
    bq = int(_prompt_lookup(prompt, "book_quality"))
    spread = float(_prompt_lookup(prompt, "credit spread").split()[0])
    return "de_risk" if (spread >= 1.90 and bq < 740) else "hold"


def test_split_surfaces_prediction_specific_fail_open():
    items = load_decision_corpus(FIXTURE)

    def faithful(prompt: str) -> str:
        return _truth(prompt)

    def distrusts_forecast(prompt: str) -> str:
        # correct on observed, but under-restricts when told the input is a forecast
        return "hold" if "FORECAST" in prompt else _truth(prompt)

    faithful_split = score_strata(items, run_model(items, faithful),
                                  key="input", min_parsed=1)
    assert faithful_split["observed"].accuracy == 1.0
    assert faithful_split["predicted"].accuracy == 1.0
    assert faithful_split["predicted"].fail_open_restrictive == 0.0

    bad_split = score_strata(items, run_model(items, distrusts_forecast),
                             key="input", min_parsed=1)
    # safe on observed inputs, but fails OPEN on the predicted arm's de_risk band
    assert bad_split["observed"].accuracy == 1.0
    assert bad_split["observed"].fail_open_restrictive == 0.0
    assert bad_split["predicted"].accuracy is not None
    assert bad_split["predicted"].accuracy < 1.0
    assert bad_split["predicted"].fail_open_restrictive == 1.0


def test_render_strata_lists_both_arms():
    items = load_decision_corpus(FIXTURE)
    table = render_strata(score_strata(items, run_model(items, _truth),
                                       key="input", min_parsed=1), label="input")
    assert "| observed" in table and "| predicted" in table


def test_score_strata_skips_untagged_items():
    items = load_decision_corpus(FIXTURE)
    strata = score_strata(items, run_model(items, _truth), key="nonexistent", min_parsed=1)
    assert strata == {}


# --------------------------------------------------------------------------- #
# live oracle wiring: the prediction fan-in reaches the SDK query
# --------------------------------------------------------------------------- #

def test_build_eval_items_threads_predictions_to_oracle():
    seen: list[dict[str, Any] | None] = []

    def report_fn(parsed: ParsedCompletion):
        seen.append(parsed.predictions)
        # oracle applies the same verdict whether the value is observed or
        # resolved from the referenced prediction
        return make_report(proof_checked=True, decision="de_risk")

    cases = [
        EvalCase(prompt="observed", facts={"spread.value": 1.97}, id="o"),
        EvalCase(prompt="predicted", facts={},
                 predictions={"spread": {"model_id": "spread_stressed", "as_of": "2026-06-30"}},
                 id="p"),
    ]
    items = build_eval_items(FakeVerifier(report_fn=report_fn), cases, VOCAB)
    assert [it.oracle for it in items] == ["de_risk", "de_risk"]
    assert seen[0] is None
    assert seen[1] == {"spread": {"model_id": "spread_stressed", "as_of": "2026-06-30"}}


@dataclass
class _RecordingPlatforms:
    result: Any
    kwargs: dict[str, Any] = field(default_factory=dict)

    def query(self, *_args: Any, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.result


@dataclass
class _FakeClient:
    platforms: _RecordingPlatforms


def test_verifier_forwards_predictions_to_sdk_query():
    v = AmberVerifier(
        domain=VerifiableDomain(platform_id=1, parser=JSONBlockParser()), cache=False)
    platforms = _RecordingPlatforms(result=make_query_result(decision="de_risk"))
    v._client = _FakeClient(platforms=platforms)  # bypass real SDK construction

    preds: dict[str, dict[str, str | None]] = {
        "spread": {"model_id": "spread_stressed", "as_of": "2026-06-30"}}
    v.verify_one(ParsedCompletion(query="q", facts={"book_quality": 700}, predictions=preds))
    assert platforms.kwargs.get("predictions") == preds

    # a facts-only completion must NOT pass predictions (backward compatible)
    v.verify_one(ParsedCompletion(query="q", facts={"book_quality": 700}))
    assert "predictions" not in platforms.kwargs
