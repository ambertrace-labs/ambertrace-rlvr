"""Load a run fully from a YAML config (spec §11).

Turns a ``configs/*.yaml`` into a fully-wired :class:`AmberVerifier` (domain +
shaper) plus the training / dataset / eval metadata a run needs — no hidden
state. The original mapping is retained on :attr:`RunConfig.raw` so a run is
auditable and reproducible.

API keys are resolved from the environment only, never from the YAML (spec §15):
a ``domain.api_key`` key in the file is a hard error.

Fixed-schema sections (``domain``, ``reward``, ``verifier``) reject unknown keys
so typos surface loudly; ``training`` keeps unrecognised keys on ``extra`` for
trainer-specific hyperparameters that the integration layer consumes later.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .domain import DEFAULT_BASE_URL, VerifiableDomain
from .parsers import CompletionParser, JSONBlockParser, RegexBlockParser
from .rewards import DEFAULT_WEIGHTS, DefaultRewardShaper, RewardShaper
from .verifier import AmberVerifier, RewardFunction

API_KEY_ENV = "AMBERTRACE_API_KEY"

# Registries typed as factories (not ``type[...]``) so pyright doesn't flag the
# Protocol classes as non-instantiable; concrete dataclasses satisfy the shape.
_PARSERS: dict[str, Callable[..., CompletionParser]] = {
    "json_block": JSONBlockParser,
    "regex_block": RegexBlockParser,
}
_SHAPERS: dict[str, Callable[..., RewardShaper]] = {
    "default": DefaultRewardShaper,
}

_TOP_LEVEL = {"domain", "reward", "verifier", "training", "dataset", "eval"}
_DEFAULT_CLIP: tuple[float, float] = (-1.0, 2.0)


@dataclass
class TrainingConfig:
    framework: str
    model: str
    group_size: int = 8
    learning_rate: float = 1e-6
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    path: str


@dataclass
class EvalConfig:
    path: str | None = None
    probes: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    """A run, fully described by its config. ``verifier`` holds the wired domain
    and shaper; the rest is the metadata a trainer/eval loop needs."""

    verifier: AmberVerifier
    training: TrainingConfig | None = None
    dataset: DatasetConfig | None = None
    eval: EvalConfig | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> VerifiableDomain:
        return self.verifier.domain

    @property
    def shaper(self) -> RewardShaper:
        return self.verifier.shaper

    def reward_function(self) -> RewardFunction:
        """The batch reward function for this run (fail-closed, never raises)."""
        return self.verifier.as_reward_function()


def load_run_config(path: str | Path, *, api_key: str | None = None) -> RunConfig:
    """Parse ``path`` into a fully-wired :class:`RunConfig`.

    ``api_key`` overrides the environment lookup (useful for tests); otherwise the
    key is read from ``AMBERTRACE_API_KEY``. Raises ``ValueError`` with an
    actionable message on any missing/unknown key.
    """
    raw = _read_yaml(path)
    _reject_unknown(raw, _TOP_LEVEL, "top-level")

    domain = _build_domain(raw, api_key)
    shaper, clip = _build_shaper(_section(raw, "reward"))
    verifier = _build_verifier(_section(raw, "verifier"), domain, shaper, clip)

    return RunConfig(
        verifier=verifier,
        training=_build_training(raw.get("training")),
        dataset=_build_dataset(raw.get("dataset")),
        eval=_build_eval(raw.get("eval")),
        raw=raw,
    )


# --- section builders ------------------------------------------------------
def _build_domain(raw: Mapping[str, Any], api_key: str | None) -> VerifiableDomain:
    dcfg = raw.get("domain")
    if not isinstance(dcfg, Mapping):
        raise ValueError("config section 'domain' is required")
    dcfg = dict(dcfg)
    if "api_key" in dcfg:
        raise ValueError(
            "'domain.api_key' must not be set in YAML; provide the key via the "
            f"{API_KEY_ENV} environment variable (spec §15)"
        )
    _reject_unknown(
        dcfg, {"platform_id", "base_url", "query_template", "parser", "parser_args"},
        "domain",
    )
    if "platform_id" not in dcfg:
        raise ValueError("'domain.platform_id' is required")

    parser = _build_parser(
        str(dcfg.get("parser", "json_block")),
        dict(dcfg.get("parser_args") or {}),
        dcfg.get("query_template"),
    )
    resolved = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    return VerifiableDomain(
        platform_id=_as_int(dcfg["platform_id"], "domain.platform_id"),
        parser=parser,
        api_key=resolved,
        base_url=str(dcfg.get("base_url", DEFAULT_BASE_URL)),
    )


def _build_parser(
    name: str, parser_args: dict[str, Any], query_template: Any,
) -> CompletionParser:
    factory = _PARSERS.get(name)
    if factory is None:
        raise ValueError(
            f"unknown parser '{name}'; known parsers: {sorted(_PARSERS)}"
        )
    # `domain.query_template` (spec §11 puts it on the domain) feeds the parser,
    # unless parser_args already names one.
    if query_template is not None and "query_template" not in parser_args:
        parser_args["query_template"] = str(query_template)
    try:
        return factory(**parser_args)
    except TypeError as err:
        raise ValueError(f"invalid parser_args for parser '{name}': {err}") from err


def _build_shaper(rcfg: Mapping[str, Any]) -> tuple[RewardShaper, tuple[float, float]]:
    rcfg = dict(rcfg)
    _reject_unknown(rcfg, {"shaper", "weights", "clip"}, "reward")
    name = str(rcfg.get("shaper", "default"))
    factory = _SHAPERS.get(name)
    if factory is None:
        raise ValueError(
            f"unknown reward shaper '{name}'; known shapers: {sorted(_SHAPERS)}"
        )
    clip = _as_clip(rcfg.get("clip", _DEFAULT_CLIP))
    return factory(weights=_build_weights(rcfg.get("weights")), clip=clip), clip


def _build_weights(raw: Any) -> dict[str, float]:
    """Merge configured weights onto the baseline, rejecting unknown component
    names. Merging (not replacing) keeps a core component like ``certified`` from
    silently dropping to zero weight when a config omits it or typos its key."""
    weights = dict(DEFAULT_WEIGHTS)
    if raw is None:
        return weights
    if not isinstance(raw, Mapping):
        raise ValueError(f"reward.weights must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - set(DEFAULT_WEIGHTS)
    if unknown:
        raise ValueError(
            f"unknown reward weight(s): {sorted(unknown)}; "
            f"allowed: {sorted(DEFAULT_WEIGHTS)}"
        )
    weights.update({str(k): float(v) for k, v in raw.items()})
    return weights


def _build_verifier(
    vcfg: Mapping[str, Any], domain: VerifiableDomain, shaper: RewardShaper,
    clip: tuple[float, float],
) -> AmberVerifier:
    vcfg = dict(vcfg)
    _reject_unknown(vcfg, {"batch_size", "max_concurrency", "cache", "floor"}, "verifier")
    return AmberVerifier(
        domain=domain,
        shaper=shaper,
        batch_size=_as_int(vcfg.get("batch_size", 32), "verifier.batch_size"),
        max_concurrency=_as_int(vcfg.get("max_concurrency", 16), "verifier.max_concurrency"),
        cache=bool(vcfg.get("cache", True)),
        # Floor tracks the shaper's lower clip bound so an unparseable completion
        # never scores below a certified one (rewards.py contract).
        floor=float(vcfg.get("floor", clip[0])),
    )


def _build_training(cfg: Any) -> TrainingConfig | None:
    if cfg is None:
        return None
    if not isinstance(cfg, Mapping):
        raise ValueError("config section 'training' must be a mapping")
    cfg = dict(cfg)
    if "framework" not in cfg or "model" not in cfg:
        raise ValueError("'training' requires 'framework' and 'model'")
    known = {"framework", "model", "group_size", "learning_rate"}
    return TrainingConfig(
        framework=str(cfg["framework"]),
        model=str(cfg["model"]),
        group_size=_as_int(cfg.get("group_size", 8), "training.group_size"),
        learning_rate=float(cfg.get("learning_rate", 1e-6)),
        extra={k: v for k, v in cfg.items() if k not in known},
    )


def _build_dataset(cfg: Any) -> DatasetConfig | None:
    if cfg is None:
        return None
    if not isinstance(cfg, Mapping):
        raise ValueError("config section 'dataset' must be a mapping")
    cfg = dict(cfg)
    _reject_unknown(cfg, {"path"}, "dataset")
    if "path" not in cfg:
        raise ValueError("'dataset.path' is required when 'dataset' is present")
    return DatasetConfig(path=str(cfg["path"]))


def _build_eval(cfg: Any) -> EvalConfig | None:
    if cfg is None:
        return None
    if not isinstance(cfg, Mapping):
        raise ValueError("config section 'eval' must be a mapping")
    cfg = dict(cfg)
    _reject_unknown(cfg, {"path", "probes"}, "eval")
    probes = cfg.get("probes") or []
    if not isinstance(probes, (list, tuple)):
        raise ValueError("'eval.probes' must be a list")
    path = cfg.get("path")
    return EvalConfig(
        path=str(path) if path is not None else None,
        probes=[str(p) for p in probes],
    )


# --- helpers ---------------------------------------------------------------
def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"config must be a YAML mapping, got {type(data).__name__}")
    return dict(data)


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    cfg = raw.get(name)
    if cfg is None:
        return {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"config section '{name}' must be a mapping")
    return cfg


def _reject_unknown(cfg: Mapping[str, Any], known: set[str], where: str) -> None:
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in '{where}': {sorted(unknown)}; allowed: {sorted(known)}"
        )


def _as_int(value: Any, where: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{where}' must be an integer, got {value!r}") from None


def _as_clip(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            raise ValueError(f"reward.clip low ({lo}) must not exceed high ({hi})")
        return (lo, hi)
    raise ValueError(f"reward.clip must be a [low, high] pair, got {value!r}")
