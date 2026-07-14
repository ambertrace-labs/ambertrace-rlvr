"""Run-report building: reward curve, summary, and secret redaction (offline)."""

from __future__ import annotations

from ambertrace_rlvr.reporting import REDACTED, build_run_report, write_run_report

LOG = [
    {"loss": 0.0, "step": 0, "reward": -0.5, "reward_std": 0.3},
    {"loss": 0.0, "step": 1, "reward": 0.2, "reward_std": 0.6},
    {"loss": 0.0, "step": 2, "reward": 1.1, "reward_std": 0.4},
    {"train_runtime": 9.0},  # trailing summary entry with no reward — ignored
]


def test_reward_curve_and_summary():
    rep = build_run_report(config={"domain": {"platform_id": 146}}, log_history=LOG)
    curve = rep["reward_curve"]
    assert [p["reward"] for p in curve] == [-0.5, 0.2, 1.1]
    assert [p["step"] for p in curve] == [0, 1, 2]
    s = rep["summary"]
    assert s["n_steps"] == 3
    assert s["first_reward"] == -0.5 and s["last_reward"] == 1.1
    assert round(s["delta"], 6) == 1.6
    assert s["max_reward"] == 1.1
    assert s["increased"] is True


def test_config_secrets_are_redacted():
    rep = build_run_report(
        config={"domain": {"platform_id": 1, "api_key": "sk-should-not-persist"},
                "auth": {"token": "t0ken"}},
        log_history=[],
    )
    assert rep["config"]["domain"]["api_key"] == REDACTED
    assert rep["config"]["auth"]["token"] == REDACTED
    assert rep["config"]["domain"]["platform_id"] == 1


def test_empty_log_history_summary():
    rep = build_run_report(config={}, log_history=[])
    assert rep["summary"] == {"n_steps": 0}
    assert rep["reward_curve"] == []


def test_write_run_report_roundtrip(tmp_path):
    import json
    rep = build_run_report(config={"x": 1}, log_history=LOG, versions={"trl": "1.8.0"})
    path = write_run_report(tmp_path / "sub" / "run_report.json", rep)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["versions"]["trl"] == "1.8.0"
    assert loaded["summary"]["increased"] is True
