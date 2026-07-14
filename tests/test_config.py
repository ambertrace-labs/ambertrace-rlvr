"""Config loader: round-trip, wiring, and fail-loud on bad/missing keys.

Offline — no network. The one reward_function() call exercises only the
unparseable path (which floors without a verify call), so no live platform is
touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ambertrace_rlvr import load_run_config
from ambertrace_rlvr.config import API_KEY_ENV
from ambertrace_rlvr.domain import DEFAULT_BASE_URL
from ambertrace_rlvr.parsers import JSONBlockParser, RegexBlockParser
from ambertrace_rlvr.rewards import DefaultRewardShaper

LOAN_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "loan_example.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "run.yaml"
    p.write_text(body)
    return p


MINIMAL = """
domain:
  platform_id: 9
  query_template: "Assess: {facts}"
  parser: json_block
  parser_args: {answer_key: classification, facts_key: facts}
"""


# --- round-trip against the shipped example --------------------------------
def test_loan_example_round_trips(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-test-key")
    run = load_run_config(LOAN_CONFIG)

    dom = run.domain
    assert dom.platform_id == 9
    assert dom.base_url == "https://app.ambertrace.ai"
    assert dom.api_key == "sk-test-key"

    # domain.query_template must land on the parser, not the domain.
    parser = dom.parser
    assert isinstance(parser, JSONBlockParser)
    assert parser.query_template == "Assess this loan application: {facts}"
    assert parser.answer_key == "classification"
    assert parser.facts_key == "facts"

    shaper = run.shaper
    assert isinstance(shaper, DefaultRewardShaper)
    assert shaper.weights == {
        "format": 0.1, "certified": 0.5, "correctness": 1.0,
        "graded": 0.3, "rejected_penalty": 0.2,
    }
    assert shaper.clip == (-1.0, 2.0)

    v = run.verifier
    assert v.batch_size == 32
    assert v.max_concurrency == 16
    assert v.cache is True
    # floor tracks the shaper's lower clip bound.
    assert v.floor == -1.0

    assert run.training is not None
    assert run.training.framework == "trl_grpo"
    assert run.training.model == "Qwen/Qwen2.5-1.5B"
    assert run.dataset is not None and run.dataset.path == "data/loan_train.jsonl"
    assert run.eval is not None and run.eval.path == "data/loan_eval.jsonl"
    # nothing is lost — the raw mapping is retained for audit.
    assert run.raw["domain"]["platform_id"] == 9


# --- key resolution --------------------------------------------------------
def test_api_key_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-env")
    run = load_run_config(_write(tmp_path, MINIMAL))
    assert run.domain.api_key == "sk-env"


def test_api_key_absent_is_none(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    run = load_run_config(_write(tmp_path, MINIMAL))
    assert run.domain.api_key is None


def test_api_key_override_param_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "sk-env")
    run = load_run_config(_write(tmp_path, MINIMAL), api_key="sk-override")
    assert run.domain.api_key == "sk-override"


def test_inline_api_key_rejected(tmp_path):
    body = MINIMAL + "  api_key: sk-should-not-be-here\n"
    with pytest.raises(ValueError, match="api_key"):
        load_run_config(_write(tmp_path, body))


# --- unknown / missing keys ------------------------------------------------
def test_unknown_parser_raises(tmp_path):
    body = "domain:\n  platform_id: 9\n  parser: mystery_block\n"
    with pytest.raises(ValueError, match="unknown parser 'mystery_block'"):
        load_run_config(_write(tmp_path, body))


def test_unknown_shaper_raises(tmp_path):
    body = MINIMAL + "reward:\n  shaper: fancy\n"
    with pytest.raises(ValueError, match="unknown reward shaper 'fancy'"):
        load_run_config(_write(tmp_path, body))


def test_unknown_top_level_key_raises(tmp_path):
    body = MINIMAL + "mystery:\n  foo: bar\n"
    with pytest.raises(ValueError, match="unknown key.*top-level"):
        load_run_config(_write(tmp_path, body))


def test_unknown_domain_key_raises(tmp_path):
    body = "domain:\n  platform_id: 9\n  typpo: 1\n"
    with pytest.raises(ValueError, match="unknown key.*domain"):
        load_run_config(_write(tmp_path, body))


def test_bad_parser_arg_raises(tmp_path):
    body = (
        "domain:\n  platform_id: 9\n  parser: json_block\n"
        "  parser_args: {not_a_real_arg: 1}\n"
    )
    with pytest.raises(ValueError, match="invalid parser_args"):
        load_run_config(_write(tmp_path, body))


def test_missing_domain_raises(tmp_path):
    with pytest.raises(ValueError, match="'domain' is required"):
        load_run_config(_write(tmp_path, "reward:\n  shaper: default\n"))


def test_missing_platform_id_raises(tmp_path):
    with pytest.raises(ValueError, match="platform_id.*required"):
        load_run_config(_write(tmp_path, "domain:\n  parser: json_block\n"))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run_config(tmp_path / "nope.yaml")


# --- defaults & section wiring ---------------------------------------------
def test_defaults_when_sections_absent(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    run = load_run_config(_write(tmp_path, MINIMAL))
    # reward/verifier default; training/dataset/eval optional -> None.
    assert isinstance(run.shaper, DefaultRewardShaper)
    assert run.verifier.batch_size == 32
    assert run.verifier.cache is True
    assert run.domain.base_url == DEFAULT_BASE_URL
    assert run.training is None
    assert run.dataset is None
    assert run.eval is None


def test_regex_parser_selected(tmp_path):
    body = "domain:\n  platform_id: 9\n  parser: regex_block\n"
    run = load_run_config(_write(tmp_path, body))
    assert isinstance(run.domain.parser, RegexBlockParser)


def test_clip_and_floor_custom(tmp_path):
    body = MINIMAL + "reward:\n  clip: [0.0, 1.0]\n"
    run = load_run_config(_write(tmp_path, body))
    assert isinstance(run.shaper, DefaultRewardShaper)
    assert run.shaper.clip == (0.0, 1.0)
    # floor follows clip[0] unless explicitly overridden.
    assert run.verifier.floor == 0.0


def test_explicit_floor_overrides_clip(tmp_path):
    body = MINIMAL + "reward:\n  clip: [-2.0, 2.0]\nverifier:\n  floor: -0.5\n"
    run = load_run_config(_write(tmp_path, body))
    assert run.verifier.floor == -0.5


def test_bad_clip_raises(tmp_path):
    body = MINIMAL + "reward:\n  clip: [2.0, 1.0]\n"
    with pytest.raises(ValueError, match="clip low"):
        load_run_config(_write(tmp_path, body))


def test_training_extra_captures_unknown_keys(tmp_path):
    body = (
        MINIMAL
        + "training:\n  framework: trl_grpo\n  model: Qwen/Qwen2.5-1.5B\n"
          "  group_size: 4\n  beta: 0.04\n  num_generations: 8\n"
    )
    run = load_run_config(_write(tmp_path, body))
    assert run.training is not None
    assert run.training.group_size == 4
    assert run.training.extra == {"beta": 0.04, "num_generations": 8}


def test_unknown_weight_key_rejected(tmp_path):
    # a typo'd component name must fail loudly, not silently zero the real one.
    body = MINIMAL + "reward:\n  weights: {certifed: 0.5}\n"
    with pytest.raises(ValueError, match="unknown reward weight"):
        load_run_config(_write(tmp_path, body))


def test_partial_weights_merge_onto_defaults(tmp_path):
    body = MINIMAL + "reward:\n  weights: {correctness: 2.0}\n"
    run = load_run_config(_write(tmp_path, body))
    assert isinstance(run.shaper, DefaultRewardShaper)
    # overridden component takes the new value...
    assert run.shaper.weights["correctness"] == 2.0
    # ...and unspecified core components keep their baseline weight.
    assert run.shaper.weights["certified"] == 0.5
    assert run.shaper.weights["format"] == 0.1


def test_reward_function_floors_unparseable_offline(tmp_path, monkeypatch):
    """reward_function() must floor a malformed completion without any network
    call (unparseable -> floor, no verify)."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    body = MINIMAL + "reward:\n  clip: [-1.0, 2.0]\n"
    run = load_run_config(_write(tmp_path, body))
    reward_fn = run.reward_function()
    rewards = reward_fn(["prompt"], ["no decision block here"], [{}])
    assert rewards == [-1.0]
