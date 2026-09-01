"""Author an **Air Track Triage verified platform** with the ``ambertraceai`` SDK,
then verify it certifies showcase tracks and write a rules manifest.

Domain description, dataset (``data/air_tracks.csv``), and holdout
(``data/air_tracks_holdout.csv``) come from the ``ambertraceai`` SDK examples
(github.com/ambertrace-labs/ambertraceai-python, examples/19_air_track_triage.py,
MIT licensed). Safe to redistribute per the SDK docstrings (synthetic/seeded).

This is an **operator / setup script, NOT library code**. ``ambertrace-rlvr``'s
reward runtime is read-only against AmberTrace; authoring a platform is a customer
step done with the SDK, which this script demonstrates. Nothing here is imported by
``src/ambertrace_rlvr/``.

The platform triages air tracks to clear / monitor / escalate (escalate = route to a
human operator). The platform never takes autonomous action -- its role is auditable,
proof-carrying decision support with a human in the loop.

Idempotent: re-runs detect the existing platform by exact name and reuse it. After
build (or reuse), the script:

  1. Smoke-tests the showcase tracks + a kinematically-implausible monitor case.
  2. Fetches the full rule inventory and writes ``data/air_track_rules.json``
     (the reproducibility seam consumed by ``gen_air_track_prompts.py``).

Usage::

    set -a; source .env; set +a
    python examples/author_air_track_platform.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "air_tracks.csv"
RULES_MANIFEST = REPO / "data" / "air_track_rules.json"

DOMAIN_NAME = "RLVR Faithfulness — Air Track Triage"
PLATFORM_NAME = "RLVR Faithfulness — Air Track Triage Platform"

# Verbatim from ambertraceai SDK examples/19_air_track_triage.py (MIT).
DOMAIN_DESCRIPTION = (
    "Air-track identification and triage decision support for compiling a recognized air "
    "picture. Each track has a sensor source, an IFF/SIF mode (mode3_valid, mode3_invalid, "
    "no_response, or emergency), an emergency-squawk flag, whether it correlates to a filed "
    "flight plan, whether its transponder is active, whether it is inside a restricted operating "
    "zone, whether it is corridor-compliant, an altitude in feet, a speed in knots, a climb rate "
    "in feet per minute, and whether its origin is known. "
    "Classify these named conditions: a track is an emergency when its emergency-squawk flag is "
    "set or its IFF mode is emergency; a track is identified when it correlates to a flight plan "
    "and its IFF mode is mode3_valid; a track is a zone breach when it is inside a restricted "
    "operating zone and is not corridor-compliant; a track is kinematically implausible when its "
    "speed is at least 600 knots and altitude at most 2000 feet, or its climb rate is at least "
    "8000 feet per minute. "
    "Triage each track by the first matching rule: escalate to an operator for an emergency "
    "track; escalate for a zone breach; escalate for an unidentified track when it is not "
    "identified and its IFF mode is no_response or mode3_invalid; monitor a kinematically "
    "implausible track; "
    "monitor an unidentified track otherwise; clear the track when none of the above apply. Every "
    "triage decision must be auditable, and emergency tracks must always be escalated to a human "
    "operator and never suppressed."
)

# Four showcase tracks from SDK example 19, plus a kinematically-implausible monitor.
SHOWCASE_TRACKS = [
    ("emergency squawk (must escalate)", "escalate", {
        "sensor_source": "radar", "iff_mode": "emergency", "squawk_emergency": True,
        "flight_plan_correlated": False, "transponder_active": True, "in_restricted_zone": False,
        "corridor_compliant": False, "altitude_ft": 12000, "speed_kts": 320,
        "climb_rate_fpm": 0, "origin_known": False}),
    ("restricted zone, no corridor", "escalate", {
        "sensor_source": "fused", "iff_mode": "mode3_valid", "squawk_emergency": False,
        "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": True,
        "corridor_compliant": False, "altitude_ft": 8000, "speed_kts": 280,
        "climb_rate_fpm": 500, "origin_known": True}),
    ("unidentified, no IFF response", "escalate", {
        "sensor_source": "radar", "iff_mode": "no_response", "squawk_emergency": False,
        "flight_plan_correlated": False, "transponder_active": False, "in_restricted_zone": False,
        "corridor_compliant": True, "altitude_ft": 26000, "speed_kts": 410,
        "climb_rate_fpm": 1200, "origin_known": False}),
    ("correlated civil traffic (clear)", "clear", {
        "sensor_source": "ads_b", "iff_mode": "mode3_valid", "squawk_emergency": False,
        "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
        "corridor_compliant": True, "altitude_ft": 34000, "speed_kts": 450,
        "climb_rate_fpm": 0, "origin_known": True}),
    ("kinematically implausible (monitor)", "monitor", {
        "sensor_source": "radar", "iff_mode": "mode3_valid", "squawk_emergency": False,
        "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
        "corridor_compliant": True, "altitude_ft": 1500, "speed_kts": 700,
        "climb_rate_fpm": 0, "origin_known": True}),
]


def _load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _find_by_name(items: list, name: str):
    for it in items:
        if it.get("name") == name:
            return it
    return None


def _write_rules_manifest(api, platform_id: int) -> None:
    """Fetch the platform's rule inventory and write the rules manifest.

    Uses ``platforms.list_rules`` — the query-independent rule inventory. The
    ``description`` is carried through because rule *names* are auto-generated
    and can read misleadingly (e.g. a rule named ``Decide monitor when
    is_identified`` whose condition is ``NOT is_identified``); any prompt that
    quotes rule names to a model must pair them with their descriptions."""
    rules_raw = api.platforms.list_rules(platform_id)

    manifest = []
    for r in rules_raw:
        name = r.get("name")
        if not name:
            continue
        entry: dict = {"name": str(name)}
        if r.get("rule_type"):
            entry["rule_type"] = str(r["rule_type"])
        if r.get("description"):
            entry["description"] = str(r["description"])
        manifest.append(entry)

    # Deduplicate by name (multiple queries may repeat rules).
    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in manifest:
        if entry["name"] not in seen:
            seen.add(entry["name"])
            deduped.append(entry)

    RULES_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RULES_MANIFEST.write_text(json.dumps(deduped, indent=2) + "\n")
    print(f"\nRules manifest written to {RULES_MANIFEST} ({len(deduped)} rules)")
    for entry in deduped:
        print(f"  - {entry['name']}")


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"dataset not found: {CSV_PATH}\n"
            "fetch it first: see the docstring in this script"
        )
    _load_dotenv()
    import ambertraceai

    api = ambertraceai.AmbertraceAPI.from_env()

    # Idempotent: reuse a built (active) platform by name so re-runs are cheap.
    existing = next(
        (p for p in api.platforms.list()
         if p.get("name") == PLATFORM_NAME and p.get("status") == "active"),
        None,
    )
    if existing is not None:
        platform_id = existing["id"]
        print(f"reusing active platform_id={platform_id}")
        _verify(api, platform_id)
        _write_rules_manifest(api, platform_id)
        _report(platform_id)
        return

    # 1. Domain (reuse by name if present).
    dom = _find_by_name(api.domains.list(), DOMAIN_NAME)
    if dom is None:
        dom = api.domains.create(name=DOMAIN_NAME, description=DOMAIN_DESCRIPTION)
        print(f"created domain_id={dom['id']}")
    else:
        print(f"reusing domain_id={dom['id']}")
    domain_id = dom["id"]

    # 2. Dataset -- UNSUPERVISED: features only, no decision_column.
    ds = next((d for d in api.datasets.list() if d.get("domain_id") == domain_id), None)
    if ds is None:
        ds = api.datasets.upload(domain_id=domain_id, file_path=str(CSV_PATH))
        print(f"uploaded dataset_id={ds['id']} rows={ds.get('row_count')} "
              f"cols={ds.get('column_count')} decision_column={ds.get('decision_column')!r}")
    else:
        print(f"reusing dataset_id={ds['id']}")

    # 3. Build the ontology + rules from the description + data.
    onto = api.domains.build_ontology(domain_id)
    print(f"building ontology (job {onto['job_id']}) -- waiting...")
    api.wait_for_job(onto["job_id"])

    # 4. Build a VERIFIED platform (machine-checked proof per query, fail-closed).
    result = api.platforms.create(
        domain_id=domain_id, dataset_id=ds["id"], name=PLATFORM_NAME,
        verified_profile=True, verified_min_confidence=0.6,
    )
    platform_id = result["id"]
    print(f"building platform_id={platform_id} (job {result['job_id']}) -- waiting...")
    api.wait_for_job(result["job_id"])
    print(f"platform status={api.platforms.status(platform_id)!r}")

    _verify(api, platform_id)
    _write_rules_manifest(api, platform_id)
    _report(platform_id)


def _verify(api, platform_id: int) -> None:
    """Smoke-test the showcase tracks through the library's AmberReport."""
    from ambertrace_rlvr.reports import AmberReport

    for label, expect, facts in SHOWCASE_TRACKS:
        res = api.platforms.query(
            platform_id, query="Triage this track.", facts=facts, explain=True,
        )
        rep = AmberReport.from_query_result(res)
        raw = str(rep.decision).lower()
        match = "OK " if raw == expect else "?? "
        deciding = [d.get("rule", "") for d in rep.deciding_rules]
        print(f"  {match}[expect {expect:>8}] decision={rep.decision!r} "
              f"proof_checked={rep.proof_checked} confidence={rep.confidence:.2f} "
              f"deciding={deciding}")


def _report(platform_id: int) -> None:
    print(f"\nDONE. platform_id={platform_id}")
    print("-> set this as domain.platform_id in configs/air_track.yaml")


if __name__ == "__main__":
    main()
