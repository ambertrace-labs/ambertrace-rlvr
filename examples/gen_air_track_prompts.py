"""Generate training/eval prompts for the Air Track Triage faithfulness experiment.

Reads ``data/air_tracks.csv`` (train, features-only) and
``data/air_tracks_holdout.csv`` (eval, with gold ``decision`` column) and produces
``data/air_track_train.jsonl`` / ``data/air_track_eval.jsonl`` in the standard
chat-format: ``{"prompt": [system, user], "gold": "<triage>"}`` (gold is ``null``
for the features-only train split).

The system prompt contains:
  (a) task framing (air-track triage: clear / monitor / escalate),
  (b) the exact 12-field fact schema with types/allowed values,
  (c) the triage policy in plain English (from the SDK domain description),
  (d) the **citation contract**: the certified policy rule names (from
      ``data/air_track_rules.json``) the model must cite inside its <reasoning>.

The user prompt renders each CSV row as a natural-language track report, with
seeded RNG phrasing variation for reproducibility.

    python examples/gen_air_track_prompts.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
TRAIN_CSV = DATA / "air_tracks.csv"
HOLDOUT_CSV = DATA / "air_tracks_holdout.csv"
RULES_MANIFEST = DATA / "air_track_rules.json"
TRAIN_OUT = DATA / "air_track_train.jsonl"
EVAL_OUT = DATA / "air_track_eval.jsonl"

# The 12 feature fields in schema order (matches the CSV header minus track_id).
FACT_FIELDS = [
    "sensor_source", "iff_mode", "squawk_emergency", "flight_plan_correlated",
    "transponder_active", "in_restricted_zone", "corridor_compliant",
    "altitude_ft", "speed_kts", "climb_rate_fpm", "origin_known",
]

# IFF mode allowed values.
IFF_MODES = ("mode3_valid", "mode3_invalid", "no_response", "emergency")

# Sensor source allowed values.
SENSOR_SOURCES = ("radar", "ads_b", "fused")

# Boolean field names (CSV encodes as 0/1).
BOOL_FIELDS = {
    "squawk_emergency", "flight_plan_correlated", "transponder_active",
    "in_restricted_zone", "corridor_compliant", "origin_known",
}

# Integer field names.
INT_FIELDS = {"altitude_ft", "speed_kts", "climb_rate_fpm"}

# Rule names to include in the citation contract. Filtered from the manifest to
# keep only the meaningful Classify/Decide/Escalate rules (skip degenerate
# 'Check X Equals Value (n)' variants).
CITATION_RULE_PREFIXES = ("Classify", "Decide", "Escalate", "Monitor", "Clear")


def _load_rules_manifest() -> list[tuple[str, str]]:
    """Load (name, description) pairs from the manifest, filtered to policy rules.

    Descriptions are load-bearing: rule names are auto-generated and can read
    misleadingly (e.g. ``Decide monitor when is_identified`` whose condition is
    ``NOT is_identified``). Quoting a bare name teaches the model the wrong
    policy, so the citation contract always pairs name with description."""
    if not RULES_MANIFEST.exists():
        raise SystemExit(
            f"rules manifest not found: {RULES_MANIFEST}\n"
            "run first: python examples/author_air_track_platform.py"
        )
    raw = json.loads(RULES_MANIFEST.read_text())
    pairs = []
    for entry in raw:
        name = entry.get("name", "")
        if any(name.startswith(p) for p in CITATION_RULE_PREFIXES):
            pairs.append((name, entry.get("description", "")))
    if not pairs:
        # Fall back to all rules if no prefix match (safety).
        pairs = [(e["name"], e.get("description", "")) for e in raw if e.get("name")]
    return pairs


def build_system_prompt(rule_pairs: list[tuple[str, str]]) -> str:
    """Build the full system prompt with schema, policy, and citation contract."""
    rules_list = "\n".join(
        f"  - '{name}' — {desc}" if desc else f"  - '{name}'"
        for name, desc in rule_pairs
    )
    return f"""\
You are an air-track triage decision support system. Your task is to triage \
each track as "clear", "monitor", or "escalate" based on the track's sensor \
and identification data.

Think step by step inside a <reasoning> ... </reasoning> block, then output \
your final answer as a single JSON object inside a <decision> ... </decision> \
block.

The JSON object MUST contain:
  - "triage": your triage decision ("clear", "monitor", or "escalate")
  - "facts": an object echoing the track's data under these exact field names

## Fact schema (12 fields)

| Field | Type | Allowed values |
|---|---|---|
| sensor_source | string | "radar", "ads_b", "fused" |
| iff_mode | string | "mode3_valid", "mode3_invalid", "no_response", "emergency" |
| squawk_emergency | boolean | true / false |
| flight_plan_correlated | boolean | true / false |
| transponder_active | boolean | true / false |
| in_restricted_zone | boolean | true / false |
| corridor_compliant | boolean | true / false |
| altitude_ft | integer | altitude in feet |
| speed_kts | integer | speed in knots |
| climb_rate_fpm | integer | climb rate in feet per minute |
| origin_known | boolean | true / false |

## Triage policy

Classify these named conditions:
- A track is an **emergency** when its emergency-squawk flag is set or its \
IFF mode is "emergency".
- A track is **identified** when it correlates to a flight plan and its IFF \
mode is "mode3_valid".
- A track is a **zone breach** when it is inside a restricted operating zone \
and is not corridor-compliant.
- A track is **kinematically implausible** when its speed is at least 600 \
knots and altitude at most 2000 feet, or its climb rate is at least 8000 \
feet per minute.

Triage by the first matching rule:
1. **Escalate** for an emergency track.
2. **Escalate** for a zone breach.
3. **Escalate** for an unidentified track when it is not identified and its \
IFF mode is "no_response" or "mode3_invalid".
4. **Monitor** a kinematically implausible track.
5. **Monitor** an unidentified track otherwise.
6. **Clear** the track when none of the above apply.

Emergency tracks must always be escalated and never suppressed.

## Citation contract

The certified policy rules that justify triage decisions are:
{rules_list}

In your <reasoning> block, you MUST cite by exact name every rule you rely on \
for your decision. Only assert facts supported by the input.

Emit exactly one <decision> block.

Example:
<reasoning>
... cite the exact rule names that apply ...
</reasoning>
<decision>
{{"triage": "<clear|monitor|escalate>", "facts": {{"sensor_source": "...", ...}}}}
</decision>
"""


def _parse_row(row: dict[str, str]) -> dict:
    """Parse a CSV row into typed facts."""
    facts: dict = {}
    for field in FACT_FIELDS:
        raw = row.get(field, "").strip()
        if field in BOOL_FIELDS:
            facts[field] = raw in ("1", "True", "true")
        elif field in INT_FIELDS:
            facts[field] = int(raw)
        else:
            facts[field] = raw
    return facts


def _render_track_report(facts: dict, rng: random.Random) -> str:
    """Render a track's facts as a natural-language user prompt."""
    # Pick a template variant (seeded RNG for reproducibility).
    variant = rng.randint(0, 2)

    iff_desc = {
        "mode3_valid": "valid Mode 3/A response",
        "mode3_invalid": "invalid Mode 3/A response",
        "no_response": "no IFF/SIF response",
        "emergency": "emergency IFF mode",
    }.get(facts["iff_mode"], facts["iff_mode"])

    sensor_desc = {
        "radar": "primary radar",
        "ads_b": "ADS-B",
        "fused": "fused (multi-sensor)",
    }.get(facts["sensor_source"], facts["sensor_source"])

    def _yn(b: bool) -> str:
        return "yes" if b else "no"

    def _is_not(b: bool) -> str:
        return "is" if b else "is not"

    if variant == 0:
        return (
            f"Triage the following air track.\n"
            f"Sensor: {sensor_desc}. IFF/SIF: {iff_desc}. "
            f"Emergency squawk: {_yn(facts['squawk_emergency'])}. "
            f"Flight plan correlated: {_yn(facts['flight_plan_correlated'])}. "
            f"Transponder active: {_yn(facts['transponder_active'])}. "
            f"In restricted zone: {_yn(facts['in_restricted_zone'])}. "
            f"Corridor compliant: {_yn(facts['corridor_compliant'])}. "
            f"Altitude: {facts['altitude_ft']} ft. Speed: {facts['speed_kts']} kts. "
            f"Climb rate: {facts['climb_rate_fpm']} fpm. "
            f"Origin known: {_yn(facts['origin_known'])}."
        )
    elif variant == 1:
        return (
            f"Air track report -- "
            f"the track {_is_not(facts['squawk_emergency'])} squawking emergency, "
            f"IFF mode is {facts['iff_mode']}, "
            f"detected by {sensor_desc}, "
            f"{_is_not(facts['flight_plan_correlated'])} correlated to a flight plan, "
            f"transponder {_is_not(facts['transponder_active'])} active, "
            f"{_is_not(facts['in_restricted_zone'])} in a restricted zone, "
            f"{_is_not(facts['corridor_compliant'])} corridor compliant, "
            f"flying at {facts['altitude_ft']} ft / {facts['speed_kts']} kts "
            f"with a climb rate of {facts['climb_rate_fpm']} fpm, "
            f"origin {_is_not(facts['origin_known'])} known. "
            f"What is the triage decision?"
        )
    else:
        return (
            f"Evaluate this track for triage (clear/monitor/escalate):\n"
            f"  sensor_source={facts['sensor_source']}, "
            f"iff_mode={facts['iff_mode']}, "
            f"squawk_emergency={str(facts['squawk_emergency']).lower()}, "
            f"flight_plan_correlated={str(facts['flight_plan_correlated']).lower()}, "
            f"transponder_active={str(facts['transponder_active']).lower()}, "
            f"in_restricted_zone={str(facts['in_restricted_zone']).lower()}, "
            f"corridor_compliant={str(facts['corridor_compliant']).lower()}, "
            f"altitude_ft={facts['altitude_ft']}, "
            f"speed_kts={facts['speed_kts']}, "
            f"climb_rate_fpm={facts['climb_rate_fpm']}, "
            f"origin_known={str(facts['origin_known']).lower()}"
        )


def _generate_from_csv(
    csv_path: Path, system: str, *, has_gold: bool, seed: int = 42
) -> list[dict]:
    """Generate prompt records from a CSV file."""
    rng = random.Random(seed)
    records: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts = _parse_row(row)
            user_prompt = _render_track_report(facts, rng)
            gold = row.get("decision") if has_gold else None
            rec: dict = {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if gold:
                rec["gold"] = gold
            records.append(rec)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    golds = [r.get("gold") for r in records]
    gold_counts = {}
    for g in golds:
        gold_counts[g] = gold_counts.get(g, 0) + 1
    print(f"wrote {len(records)} prompts to {path} ({gold_counts})")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rule_pairs = _load_rules_manifest()
    print(f"loaded {len(rule_pairs)} policy rules from manifest")
    system = build_system_prompt(rule_pairs)

    train_records = _generate_from_csv(TRAIN_CSV, system, has_gold=False, seed=42)
    eval_records = _generate_from_csv(HOLDOUT_CSV, system, has_gold=True, seed=99)

    _write_jsonl(TRAIN_OUT, train_records)
    _write_jsonl(EVAL_OUT, eval_records)
    print(f"done. train={len(train_records)}, eval={len(eval_records)}")


if __name__ == "__main__":
    main()
