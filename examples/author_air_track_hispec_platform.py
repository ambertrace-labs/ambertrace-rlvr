"""Author a **High-Spec ISR Air Track Triage verified platform** with the
``ambertraceai`` SDK, run the full acceptance gate, and write a rules manifest.

Domain description, dataset (``data/air_tracks_hispec.csv``, 2000 rows), and
holdout (``data/air_tracks_hispec_holdout.csv``, 50 rows with gold decision +
triage_reason) come from the ``ambertraceai`` SDK examples
(github.com/ambertrace-labs/ambertraceai-python,
examples/24_air_track_isr_hispec.py, MIT licensed). Safe to redistribute per the
SDK docstrings (synthetic/seeded).

This is an **operator / setup script, NOT library code**. ``ambertrace-rlvr``'s
reward runtime is read-only against AmberTrace; authoring a platform is a customer
step done with the SDK, which this script demonstrates. Nothing here is imported by
``src/ambertrace_rlvr/``.

The platform triages standardised ISR surveillance tracks (27-field ASTERIX/MISB-
style schema) to clear / monitor / escalate (escalate = route to a human operator).
The platform never takes autonomous action -- its role is auditable, proof-carrying
decision support with a human in the loop.

Uses timestamped names so every run creates a fresh build (no reuse of a pre-fix
platform). After build the script runs the three-part acceptance gate:

  1. 50-row gold holdout (decision match, booleans/ints coerced from CSV strings).
  2. 5 SHOWCASE_TRACKS from SDK example 24 (expected decisions embedded).
  3. 6 isolated per-policy-branch probes adapted to the hi-spec schema.

Retries on ``ProgrammingError`` and rate-limit backoff. Build timeout 1800s.

Usage::

    set -a; source .env; set +a
    python examples/author_air_track_hispec_platform.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "air_tracks_hispec.csv"
HOLDOUT_PATH = REPO / "data" / "air_tracks_hispec_holdout.csv"
RULES_MANIFEST = REPO / "data" / "air_track_hispec_rules.json"

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
DOMAIN_NAME = f"RLVR Faithfulness — Air Track Hi-Spec ISR ({TIMESTAMP})"
PLATFORM_NAME = f"RLVR Faithfulness — Air Track Hi-Spec ISR Platform ({TIMESTAMP})"

# Verbatim from ambertraceai SDK examples/24_air_track_isr_hispec.py (MIT).
DOMAIN_DESCRIPTION = (
    "Air-track identification and triage decision support that ingests standardised "
    "ISR surveillance tracks for a recognized air picture. Each track has a position "
    "source (ads_b, asterix_radar, mlat, flarm, or fused), a Mode 3/A code, an "
    "emergency-squawk flag, a special-position-indicator flag, an IFF/SIF mode "
    "(mode3_valid, mode3_invalid, no_response, or emergency), whether it correlates to "
    "a filed flight plan, whether it is on the ground, a barometric altitude in feet, a "
    "geometric altitude in feet, a ground speed in knots, a track angle in degrees, a "
    "vertical rate in feet per minute, a latitude and longitude, whether it is inside a "
    "restricted operating zone, whether it is corridor-compliant, whether its origin is "
    "known, the observing sensor platform with its latitude, longitude and true "
    "altitude, the slant range in metres, and a track confidence. "
    "Classify these named conditions: a track is an emergency when its emergency-squawk "
    "flag is set or its IFF mode is emergency; a track is identified when it correlates "
    "to a filed flight plan and its IFF mode is mode3_valid; a track is a zone breach "
    "when it is inside a restricted operating zone and is not corridor-compliant; a "
    "track is kinematically implausible when its ground speed is at least 600 knots and "
    "its barometric altitude is at most 2000 feet, or its vertical rate is at least 8000 "
    "feet per minute. "
    "Triage each track by the first matching rule: escalate to an operator for an "
    "emergency track; escalate for a zone breach; escalate for an unidentified track "
    "when it is not identified and its IFF mode is no_response or mode3_invalid; monitor "
    "a kinematically implausible track; monitor an unidentified track otherwise; clear "
    "the track when none of the above apply. Every triage decision must be auditable, "
    "and emergency tracks must always be escalated to a human operator and never "
    "suppressed."
)

# ---------------------------------------------------------------------------
# Showcase tracks from SDK example 24 (extracted via _track helper shim).
# ---------------------------------------------------------------------------
_BASE_TRACK: dict = {
    "track_number": 1, "icao24": "3c5eec", "callsign": "DLH4AB",
    "origin_country": "Germany", "position_source": "ads_b", "mode3a_code": "1000",
    "emergency_squawk": False, "spi": False, "iff_mode": "mode3_valid",
    "flight_plan_correlated": True, "on_ground": False, "baro_altitude_ft": 34000,
    "geo_altitude_ft": 34250, "ground_speed_kts": 440, "track_angle_deg": 95,
    "vertical_rate_fpm": 0, "latitude_deg": 47.30, "longitude_deg": 8.40,
    "in_restricted_zone": False, "corridor_compliant": True, "origin_known": True,
    "sensor_platform": "GS-RADAR-02", "sensor_latitude": 47.50,
    "sensor_longitude": 9.00, "sensor_true_altitude_ft": 1500, "slant_range_m": 62000,
    "track_confidence": 0.90,
}


def _track(**overrides: object) -> dict:
    t = dict(_BASE_TRACK)
    t.update(overrides)
    return t


SHOWCASE_TRACKS: list[tuple[str, str, dict]] = [
    ("emergency squawk 7700 near the energy facility, also fast-low",
     "escalate", _track(
         position_source="fused", mode3a_code="7700", emergency_squawk=True,
         iff_mode="emergency", flight_plan_correlated=False, in_restricted_zone=True,
         corridor_compliant=False, baro_altitude_ft=1200, ground_speed_kts=640,
         latitude_deg=47.00, longitude_deg=8.00, track_confidence=0.93,
         origin_known=False)),
    ("uncoordinated detection, no IFF, in restricted zone off-corridor",
     "escalate", _track(
         position_source="asterix_radar", callsign="UNKNOWN", mode3a_code="",
         iff_mode="no_response", flight_plan_correlated=False, in_restricted_zone=True,
         corridor_compliant=False, baro_altitude_ft=2400, ground_speed_kts=110,
         latitude_deg=47.01, longitude_deg=8.01, origin_known=False,
         track_confidence=0.62)),
    ("valid Mode-S airliner inside the zone but corridor-compliant",
     "clear", _track(
         in_restricted_zone=True, corridor_compliant=True, baro_altitude_ft=9000,
         ground_speed_kts=300, latitude_deg=47.00, longitude_deg=8.02,
         track_confidence=0.95)),
    ("valid IFF but no filed flight-plan correlation",
     "monitor", _track(
         position_source="ads_b", iff_mode="mode3_valid",
         flight_plan_correlated=False, callsign="N512TQ", origin_known=False,
         track_confidence=0.88)),
    ("fast-low track at 650 kts / 1500 ft",
     "monitor", _track(
         position_source="mlat", iff_mode="mode3_valid",
         flight_plan_correlated=False, ground_speed_kts=650, baro_altitude_ft=1500,
         track_confidence=0.58)),
]

# ---------------------------------------------------------------------------
# Isolated per-policy-branch probes (hi-spec schema).
# Each exercises exactly one policy branch and must not trigger others.
# ---------------------------------------------------------------------------
POLICY_PROBES: list[tuple[str, str, dict]] = [
    # 1. Zone breach (in_restricted_zone AND NOT corridor_compliant, not emergency)
    ("isolated zone breach", "escalate", _track(
        in_restricted_zone=True, corridor_compliant=False,
        emergency_squawk=False, iff_mode="mode3_valid",
        flight_plan_correlated=True, ground_speed_kts=250,
        baro_altitude_ft=8000, vertical_rate_fpm=0)),
    # 2. Emergency (emergency_squawk set, nothing else interesting)
    ("isolated emergency squawk", "escalate", _track(
        emergency_squawk=True, iff_mode="mode3_valid",
        flight_plan_correlated=True, in_restricted_zone=False,
        corridor_compliant=True, ground_speed_kts=300,
        baro_altitude_ft=12000, vertical_rate_fpm=0)),
    # 3. Unidentified bad IFF (not identified, iff=mode3_invalid, no zone breach)
    ("isolated unidentified bad IFF", "escalate", _track(
        iff_mode="mode3_invalid", flight_plan_correlated=False,
        emergency_squawk=False, in_restricted_zone=False,
        corridor_compliant=True, ground_speed_kts=280,
        baro_altitude_ft=15000, vertical_rate_fpm=200, origin_known=False)),
    # 4. Kinematically implausible via baro_altitude/ground_speed
    ("isolated kinematic (fast-low)", "monitor", _track(
        ground_speed_kts=650, baro_altitude_ft=1800,
        iff_mode="mode3_valid", flight_plan_correlated=True,
        emergency_squawk=False, in_restricted_zone=False,
        corridor_compliant=True, vertical_rate_fpm=0)),
    # 5. Kinematically implausible via vertical_rate alone
    ("isolated kinematic (vertical rate)", "monitor", _track(
        vertical_rate_fpm=9000, ground_speed_kts=200,
        baro_altitude_ft=20000,
        iff_mode="mode3_valid", flight_plan_correlated=True,
        emergency_squawk=False, in_restricted_zone=False,
        corridor_compliant=True)),
    # 6. No-trigger clear (identified, not emergency, not zone breach, normal kinematics)
    ("isolated no-trigger clear", "clear", _track(
        iff_mode="mode3_valid", flight_plan_correlated=True,
        emergency_squawk=False, in_restricted_zone=False,
        corridor_compliant=True, ground_speed_kts=420,
        baro_altitude_ft=35000, vertical_rate_fpm=0, origin_known=True)),
]

# ---------------------------------------------------------------------------
# Boolean / numeric columns that need coercion from CSV string form.
# ---------------------------------------------------------------------------
BOOL_COLUMNS = frozenset({
    "emergency_squawk", "spi", "flight_plan_correlated", "on_ground",
    "in_restricted_zone", "corridor_compliant", "origin_known",
})
INT_COLUMNS = frozenset({
    "track_number", "baro_altitude_ft", "geo_altitude_ft", "ground_speed_kts",
    "track_angle_deg", "vertical_rate_fpm", "sensor_true_altitude_ft", "slant_range_m",
})
FLOAT_COLUMNS = frozenset({
    "latitude_deg", "longitude_deg", "sensor_latitude", "sensor_longitude",
    "track_confidence",
})
# Columns NOT sent as query facts (gold labels).
GOLD_COLUMNS = frozenset({"decision", "triage_reason"})


def _coerce_row(row: dict[str, str]) -> dict:
    """Coerce CSV string values to the types the platform schema expects."""
    facts: dict = {}
    for k, v in row.items():
        if k in GOLD_COLUMNS:
            continue
        if k in BOOL_COLUMNS:
            facts[k] = v.strip() not in ("0", "false", "False", "")
        elif k in INT_COLUMNS:
            facts[k] = int(v)
        elif k in FLOAT_COLUMNS:
            facts[k] = float(v)
        else:
            facts[k] = v
    return facts


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _api_call_with_retry(fn, *args, **kwargs):
    """Call *fn* with retries on ProgrammingError / rate-limit (429)."""
    import ambertraceai
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except ambertraceai.AmbertraceError as exc:
            msg = str(exc).lower()
            retriable = "programmingerror" in msg or "429" in msg or "rate" in msg
            if not retriable or attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] {exc} — waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def _write_rules_manifest(api, platform_id: int) -> None:
    """Fetch the platform's rule inventory and write the rules manifest."""
    rules_raw = api.platforms.list_rules(platform_id)

    manifest: list[dict] = []
    seen: set[str] = set()
    for r in rules_raw:
        name = r.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        entry: dict = {"name": str(name)}
        if r.get("rule_type"):
            entry["rule_type"] = str(r["rule_type"])
        if r.get("description"):
            entry["description"] = str(r["description"])
        manifest.append(entry)

    RULES_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RULES_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nRules manifest written to {RULES_MANIFEST} ({len(manifest)} rules)")
    for entry in manifest:
        print(f"  - {entry['name']}")


# ---------------------------------------------------------------------------
# Acceptance gate.
# ---------------------------------------------------------------------------

def _query(api, platform_id: int, facts: dict) -> dict:
    return _api_call_with_retry(
        api.platforms.query, platform_id,
        query="Triage this track.", facts=facts, explain=True,
    )


def _run_holdout_gate(api, platform_id: int) -> tuple[int, int, list[dict]]:
    """Gate A: 50-row gold holdout. Returns (pass, total, failures)."""
    with open(HOLDOUT_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    passed = 0
    failures: list[dict] = []
    for i, row in enumerate(rows):
        gold = row["decision"].strip().lower()
        facts = _coerce_row(row)
        try:
            res = _query(api, platform_id, facts)
            got = str(res.get("decision", "")).strip().lower()
        except Exception as exc:
            got = f"ERROR:{exc}"
        ok = got == gold
        if ok:
            passed += 1
        else:
            failures.append({
                "row": i + 1, "track_number": row.get("track_number"),
                "gold": gold, "got": got,
                "triage_reason": row.get("triage_reason", ""),
            })
        # Rate-limit courtesy.
        if (i + 1) % 10 == 0:
            print(f"  holdout {i + 1}/{len(rows)}: {passed} correct so far")

    return passed, len(rows), failures


def _run_showcase_gate(api, platform_id: int) -> tuple[int, int, list[dict]]:
    """Gate B: 5 SHOWCASE_TRACKS. Returns (pass, total, failures)."""
    passed = 0
    failures: list[dict] = []
    for label, expected, facts in SHOWCASE_TRACKS:
        try:
            res = _query(api, platform_id, facts)
            got = str(res.get("decision", "")).strip().lower()
        except Exception as exc:
            got = f"ERROR:{exc}"
        ok = got == expected
        if ok:
            passed += 1
        else:
            failures.append({"label": label, "expected": expected, "got": got})
        mark = "OK" if ok else "FAIL"
        print(f"  {mark} [{expected:>8}] got={got!r} — {label}")
    return passed, len(SHOWCASE_TRACKS), failures


def _run_probe_gate(api, platform_id: int) -> tuple[int, int, list[dict]]:
    """Gate C: 6 isolated per-policy-branch probes. Returns (pass, total, failures)."""
    passed = 0
    failures: list[dict] = []
    for label, expected, facts in POLICY_PROBES:
        try:
            res = _query(api, platform_id, facts)
            got = str(res.get("decision", "")).strip().lower()
        except Exception as exc:
            got = f"ERROR:{exc}"
        ok = got == expected
        if ok:
            passed += 1
        else:
            failures.append({"label": label, "expected": expected, "got": got})
        mark = "OK" if ok else "FAIL"
        print(f"  {mark} [{expected:>8}] got={got!r} — {label}")
    return passed, len(POLICY_PROBES), failures


def _print_gate_table(results: dict) -> None:
    """Print a markdown-style gate table."""
    print("\n## Acceptance Gate Results\n")
    print("| Gate | Pass | Total | Status |")
    print("|------|------|-------|--------|")
    all_ok = True
    for name, (p, t, _) in results.items():
        status = "PASS" if p == t else "FAIL"
        if p != t:
            all_ok = False
        print(f"| {name} | {p} | {t} | {status} |")
    print(f"\nVerdict: {'GO' if all_ok else 'NO-GO'}")
    return


# ---------------------------------------------------------------------------
# Build + run.
# ---------------------------------------------------------------------------

def _build_platform(api) -> tuple[int, int]:
    """Create domain, upload data, build ontology + platform. Returns (platform_id, domain_id)."""
    # 1. Domain.
    dom = api.domains.create(name=DOMAIN_NAME, description=DOMAIN_DESCRIPTION)
    domain_id = dom["id"]
    print(f"created domain_id={domain_id}")

    # 2. Dataset -- UNSUPERVISED: features only, no decision_column.
    ds = _api_call_with_retry(
        api.datasets.upload, domain_id=domain_id, file_path=str(CSV_PATH),
    )
    print(f"uploaded dataset_id={ds['id']} rows={ds.get('row_count')} "
          f"cols={ds.get('column_count')} decision_column={ds.get('decision_column')!r}")

    # 3. Build ontology.
    onto = _api_call_with_retry(api.domains.build_ontology, domain_id)
    print(f"building ontology (job {onto['job_id']}) -- waiting...")
    api.wait_for_job(onto["job_id"], timeout=1800)

    # 4. Build VERIFIED platform.
    result = _api_call_with_retry(
        api.platforms.create,
        domain_id=domain_id, dataset_id=ds["id"], name=PLATFORM_NAME,
        verified_profile=True, verified_min_confidence=0.6,
    )
    platform_id = result["id"]
    print(f"building platform_id={platform_id} (job {result['job_id']}) -- waiting...")
    api.wait_for_job(result["job_id"], timeout=1800)
    status = api.platforms.status(platform_id)
    print(f"platform status={status!r}")
    return platform_id, domain_id


def main() -> None:
    for path, label in [(CSV_PATH, "dataset"), (HOLDOUT_PATH, "holdout")]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    _load_dotenv()
    import ambertraceai

    api = ambertraceai.AmbertraceAPI.from_env()

    # -- Build (with one retry on total failure) --
    attempt = 0
    max_build_attempts = 2
    while True:
        attempt += 1
        try:
            platform_id, domain_id = _build_platform(api)
            break
        except Exception as exc:
            if attempt >= max_build_attempts:
                print(f"\nBuild failed after {attempt} attempts: {exc}", file=sys.stderr)
                raise
            print(f"\n[build retry {attempt}] {exc} — rebuilding from scratch")

    # -- Rules manifest --
    _write_rules_manifest(api, platform_id)

    # -- Acceptance gate --
    print("\n" + "=" * 70)
    print("ACCEPTANCE GATE")
    print("=" * 70)

    print("\n--- Gate A: 50-row gold holdout ---")
    hp, ht, hf = _run_holdout_gate(api, platform_id)

    print("\n--- Gate B: 5 SHOWCASE_TRACKS ---")
    sp, st, sf = _run_showcase_gate(api, platform_id)

    print("\n--- Gate C: 6 isolated policy probes ---")
    pp, pt, pf = _run_probe_gate(api, platform_id)

    results = {
        "holdout (50-row gold)": (hp, ht, hf),
        "showcase (5 tracks)": (sp, st, sf),
        "policy probes (6)": (pp, pt, pf),
    }
    _print_gate_table(results)

    all_pass = all(p == t for p, t, _ in results.values())

    if not all_pass:
        print(f"\nNO-GO. platform_id={platform_id} domain_id={domain_id}")
        # Dump failures for the dev report.
        for name, (p, t, failures) in results.items():
            if failures:
                print(f"\n  {name} failures:")
                for fail in failures:
                    print(f"    {fail}")
        # Dump rules for diagnosis.
        print("\n  Rules on the failing build:")
        try:
            rules = api.platforms.list_rules(platform_id)
            for r in rules:
                print(f"    - {r.get('name')}: {r.get('rule_type')} — "
                      f"{r.get('description', '')[:120]}")
        except Exception as exc:
            print(f"    (could not fetch rules: {exc})")
        sys.exit(1)

    print(f"\nGO. platform_id={platform_id}")
    print("Step 2 (prompt generation + OLMo baseline) can proceed.")


if __name__ == "__main__":
    main()
