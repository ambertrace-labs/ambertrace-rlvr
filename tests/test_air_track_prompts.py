"""Offline tests for the air-track prompt generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add examples to sys.path so we can import the generator.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

from gen_air_track_prompts import (  # noqa: E402
    FACT_FIELDS,
    _parse_row,
    _render_track_report,
    build_system_prompt,
)

from ambertrace_rlvr.parsers import JSONBlockParser  # noqa: E402

# A fixture rules manifest (the real one is written by the author script).
FIXTURE_RULES = [
    "Classify Is Emergency",
    "Classify Is Identified",
    "Classify Is Zone Breach",
    "Classify Is Kinematically Implausible",
    "Decide escalate when is_emergency",
    "Decide escalate when is_zone_breach",
    "Escalate Unidentified Track",
    "Monitor Kinematically Implausible",
    "Decide clear otherwise",
]


def _system() -> str:
    return build_system_prompt(FIXTURE_RULES)


# --- (a) schema fields all present in system prompt
def test_system_prompt_has_all_fact_fields():
    system = _system()
    for field in FACT_FIELDS:
        assert field in system, f"field {field!r} missing from system prompt"


# --- (b) decision-block instruction correct
def test_system_prompt_has_decision_block_instruction():
    system = _system()
    assert '"triage"' in system
    assert '"facts"' in system
    assert "<decision>" in system
    assert "</decision>" in system
    assert "<reasoning>" in system
    assert "</reasoning>" in system


# --- (c) rule names from fixture manifest appear in system prompt
def test_system_prompt_has_rule_names():
    system = _system()
    for rule in FIXTURE_RULES:
        assert rule in system, f"rule {rule!r} missing from system prompt"


# --- JSONL round-trip
def test_jsonl_roundtrip(tmp_path):
    """Generate a record, write/read JSONL, check structure."""
    import random
    system = _system()
    row = {
        "track_id": "TRK-TEST",
        "sensor_source": "radar",
        "iff_mode": "emergency",
        "squawk_emergency": "1",
        "flight_plan_correlated": "0",
        "transponder_active": "1",
        "in_restricted_zone": "0",
        "corridor_compliant": "0",
        "altitude_ft": "12000",
        "speed_kts": "320",
        "climb_rate_fpm": "0",
        "origin_known": "0",
    }
    facts = _parse_row(row)
    user = _render_track_report(facts, random.Random(42))
    rec = {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "gold": "escalate",
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    loaded = json.loads(path.read_text().strip())
    assert loaded["prompt"][0]["role"] == "system"
    assert loaded["prompt"][1]["role"] == "user"
    assert loaded["gold"] == "escalate"


# --- gen output parses with the config's parser
def test_well_formed_completion_parses():
    """A synthetic well-formed air-track completion parses with the config's parser."""
    parser = JSONBlockParser(answer_key="triage", facts_key="facts",
                             query_template="Triage this track: {facts}")
    prompt = "Triage the following air track."
    completion = (
        "<reasoning>The track is squawking emergency. "
        "By Classify Is Emergency and Decide escalate when is_emergency, "
        "we escalate.</reasoning>"
        '<decision>{"triage": "escalate", "facts": {'
        '"sensor_source": "radar", "iff_mode": "emergency", '
        '"squawk_emergency": true, "flight_plan_correlated": false, '
        '"transponder_active": true, "in_restricted_zone": false, '
        '"corridor_compliant": false, "altitude_ft": 12000, '
        '"speed_kts": 320, "climb_rate_fpm": 0, "origin_known": false'
        "}}</decision>"
    )
    parsed = parser.parse(prompt, completion)
    assert parsed is not None
    assert parsed.proposed_answer == "escalate"
    assert parsed.reasoning is not None
    assert "Classify Is Emergency" in parsed.reasoning
    assert parsed.facts["sensor_source"] == "radar"


# --- parse_row coerces types correctly
def test_parse_row_types():
    row = {
        "sensor_source": "ads_b",
        "iff_mode": "mode3_valid",
        "squawk_emergency": "0",
        "flight_plan_correlated": "1",
        "transponder_active": "1",
        "in_restricted_zone": "0",
        "corridor_compliant": "1",
        "altitude_ft": "34000",
        "speed_kts": "450",
        "climb_rate_fpm": "0",
        "origin_known": "1",
    }
    facts = _parse_row(row)
    assert facts["sensor_source"] == "ads_b"
    assert facts["squawk_emergency"] is False
    assert facts["flight_plan_correlated"] is True
    assert facts["altitude_ft"] == 34000
    assert isinstance(facts["speed_kts"], int)
