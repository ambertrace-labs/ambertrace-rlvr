"""Run reports: capture a training run's config + reward trace for audit and
learning curves (spec §12).

Pure data formatting — no trainer, network, or plotting dependencies — so it is
offline-testable and safe to import anywhere. Secrets are redacted defensively:
even though our config never stores an API key, any key-like field is scrubbed
before a report is written.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REDACTED = "***REDACTED***"
_SECRET_HINTS = ("api_key", "apikey", "token", "secret", "password")


def build_run_report(
    *,
    config: Mapping[str, Any],
    log_history: Sequence[Mapping[str, Any]],
    versions: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    reward_key: str = "reward",
) -> dict[str, Any]:
    """Assemble a run report from a config snapshot and a trainer's per-step log.

    ``log_history`` is a list of metric dicts (e.g. ``trainer.state.log_history``
    from TRL). Steps carrying ``reward_key`` become the learning curve.
    """
    curve = _reward_curve(log_history, reward_key)
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": _redact(config),
        "versions": dict(versions or {}),
        "reward_curve": curve,
        "summary": _summary(curve),
        "metrics": [_redact(m) for m in log_history],
        **({"extra": _redact(extra)} if extra else {}),
    }


def write_run_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write ``report`` as JSON, creating parent dirs. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str))
    return p


# --- internals -------------------------------------------------------------
def _reward_curve(
    log_history: Sequence[Mapping[str, Any]], reward_key: str,
) -> list[dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    for entry in log_history:
        if reward_key not in entry:
            continue
        point: dict[str, Any] = {"reward": _as_float(entry[reward_key])}
        if "step" in entry:
            point["step"] = int(_as_float(entry["step"]))
        if "reward_std" in entry:
            point["reward_std"] = _as_float(entry["reward_std"])
        curve.append(point)
    # backfill step index if the trainer didn't stamp one
    for i, point in enumerate(curve):
        point.setdefault("step", i)
    return curve


def _summary(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not curve:
        return {"n_steps": 0}
    first, last = curve[0]["reward"], curve[-1]["reward"]
    return {
        "n_steps": len(curve),
        "first_reward": first,
        "last_reward": last,
        "delta": last - first,
        "max_reward": max(p["reward"] for p in curve),
        "increased": last > first,
    }


def _redact(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and any(h in k.lower() for h in _SECRET_HINTS):
                out[k] = REDACTED
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
