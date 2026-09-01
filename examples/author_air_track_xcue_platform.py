"""Author a **cross-domain cueing Air Track Triage platform** (SDK 2.1.3 preview).

Extends the base air-track triage policy (examples/19 verbatim) with one
cross-domain cueing clause: a track is *maritime-cued* when there exists a
related ``maritime_track`` in the same ``grid_square`` whose ``zone_status``
is ``exclusion_breach`` and ``ais_corroborated`` is true.  Maritime-cued tracks
are escalated (priority alongside zone breach).

The relation is declared at ``build_ontology`` time via the ``relations``
parameter (description-driven authoring, SDK 2.1.3 preview — see the
``build_ontology`` docstring).  At query time, maritime rows are supplied via
``platforms.query(relations={"maritime_track": [...]})`` and the kernel brings
the join inside the proof, deriving the cue from the attached rows.

**Gate results (NO-GO — 2026-09-01):**

  Build 1 (platform 1402):
    a. Holdout (50 rows, no relations): 50/50  GO
    b. Base probes (6 per-policy-branch): 6/6  GO
    c. Cross-domain probes (6): 1/6  NO-GO
       The builder did NOT produce an ``existsRelated`` derive rule for the
       declared maritime_track relation.  All relation rows are rejected at
       the certified-fact gate (HTTP 503: "certified relation rows were rejected
       by the certified-fact gate; no decision was certified over a partial
       relation").  The base policy is correct; the cross-domain cueing clause
       is accepted in the description but never composed into a machine condition.

  Build 2 (platform 1403, retry):
    a. Holdout: 22/50  NO-GO (many tracks returning "abstain")
    b. Base probes: 2/6  NO-GO
    c. Cross-domain probes: 1/6  NO-GO

  Root cause: ``build_ontology(relations=[...])`` accepts the relation declaration
  without error, and the description clause ("a track is maritime-cued when there
  exists a related maritime_track ...") matches the SDK's documented pattern
  exactly, but the builder does not compose it into an ``existsRelated`` derive
  rule.  Without the rule, relation rows fail per-cell certification and the
  query fails closed.  This is a platform build quality issue (platform-side
  issue #1672).

Dataset: ``data/air_tracks_xcue.csv`` (500 training rows with grid_square).
Relation reference: ``data/maritime_tracks.csv`` (200 seeded maritime rows).
Rules manifest: ``data/air_track_xcue_rules.json`` (not written — NO-GO).

Idempotent: re-runs detect the existing platform by exact name and reuse it.

Usage::

    set -a; source .env; set +a
    python examples/author_air_track_xcue_platform.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "air_tracks_xcue.csv"
HOLDOUT_PATH = REPO / "data" / "air_tracks_holdout.csv"
RULES_MANIFEST = REPO / "data" / "air_track_xcue_rules.json"

DOMAIN_NAME = "RLVR — Air Track Triage × Maritime Cueing"
PLATFORM_NAME = "RLVR — Air Track Triage × Maritime Cueing Platform"

# Base policy: verbatim from ambertraceai SDK examples/19_air_track_triage.py (MIT).
# Extended with the cross-domain cueing clause (last paragraph).
DOMAIN_DESCRIPTION = (
    "Air-track identification and triage decision support for compiling a recognized air "
    "picture. Each track has a sensor source, an IFF/SIF mode (mode3_valid, mode3_invalid, "
    "no_response, or emergency), an emergency-squawk flag, whether it correlates to a filed "
    "flight plan, whether its transponder is active, whether it is inside a restricted operating "
    "zone, whether it is corridor-compliant, an altitude in feet, a speed in knots, a climb rate "
    "in feet per minute, whether its origin is known, and a grid_square. "
    "Classify these named conditions: a track is an emergency when its emergency-squawk flag is "
    "set or its IFF mode is emergency; a track is identified when it correlates to a flight plan "
    "and its IFF mode is mode3_valid; a track is a zone breach when it is inside a restricted "
    "operating zone and is not corridor-compliant; a track is kinematically implausible when its "
    "speed is at least 600 knots and altitude at most 2000 feet, or its climb rate is at least "
    "8000 feet per minute; "
    "a track is maritime-cued when there exists a related maritime_track in the same grid_square "
    "whose zone_status is exclusion_breach and ais_corroborated is true. "
    "Triage each track by the first matching rule: escalate to an operator for an emergency "
    "track; escalate for a zone breach; escalate for a maritime-cued track; escalate for an "
    "unidentified track when it is not identified and its IFF mode is no_response or "
    "mode3_invalid; monitor a kinematically implausible track; "
    "monitor an unidentified track otherwise; clear the track when none of the above apply. Every "
    "triage decision must be auditable, and emergency tracks must always be escalated to a human "
    "operator and never suppressed."
)

# Relation schema for build_ontology (SDK 2.1.3 cross-domain cueing preview).
MARITIME_RELATION = {
    "name": "maritime_track",
    "join_key": "grid_square",
    "columns": [
        {"name": "zone_status", "type": "enum", "enum_values": ["normal", "advisory", "exclusion_breach"]},
        {"name": "ais_corroborated", "type": "bool"},
    ],
}

# ---- gate probes ---------------------------------------------------------

# Base policy probes (6, per-policy-branch — no relations attached).
BASE_PROBES = [
    ("emergency squawk → escalate", "escalate", {
        "sensor_source": "radar", "iff_mode": "emergency", "squawk_emergency": True,
        "flight_plan_correlated": False, "transponder_active": True, "in_restricted_zone": False,
        "corridor_compliant": False, "altitude_ft": 12000, "speed_kts": 320,
        "climb_rate_fpm": 0, "origin_known": False, "grid_square": "A1"}),
    ("restricted zone breach → escalate", "escalate", {
        "sensor_source": "fused", "iff_mode": "mode3_valid", "squawk_emergency": False,
        "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": True,
        "corridor_compliant": False, "altitude_ft": 8000, "speed_kts": 280,
        "climb_rate_fpm": 500, "origin_known": True, "grid_square": "B2"}),
    ("unidentified no_response → escalate", "escalate", {
        "sensor_source": "radar", "iff_mode": "no_response", "squawk_emergency": False,
        "flight_plan_correlated": False, "transponder_active": False, "in_restricted_zone": False,
        "corridor_compliant": True, "altitude_ft": 26000, "speed_kts": 410,
        "climb_rate_fpm": 1200, "origin_known": False, "grid_square": "C3"}),
    ("kinematically implausible → monitor", "monitor", {
        "sensor_source": "radar", "iff_mode": "mode3_valid", "squawk_emergency": False,
        "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
        "corridor_compliant": True, "altitude_ft": 1500, "speed_kts": 700,
        "climb_rate_fpm": 0, "origin_known": True, "grid_square": "D4"}),
    ("unidentified mode3_valid not correlated → monitor", "monitor", {
        "sensor_source": "radar", "iff_mode": "mode3_valid", "squawk_emergency": False,
        "flight_plan_correlated": False, "transponder_active": True, "in_restricted_zone": False,
        "corridor_compliant": True, "altitude_ft": 30000, "speed_kts": 400,
        "climb_rate_fpm": 500, "origin_known": False, "grid_square": "E5"}),
    ("correlated civil traffic → clear", "clear", {
        "sensor_source": "ads_b", "iff_mode": "mode3_valid", "squawk_emergency": False,
        "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
        "corridor_compliant": True, "altitude_ft": 34000, "speed_kts": 450,
        "climb_rate_fpm": 0, "origin_known": True, "grid_square": "F6"}),
]

# Cross-domain cueing probes (6).
XCUE_PROBES: list[tuple[str, str | None, dict, dict | None]] = [
    # (i) Clear track + matching-grid exclusion_breach + ais=true → escalate.
    ("xcue: matching cue → escalate", "escalate",
     {"sensor_source": "ads_b", "iff_mode": "mode3_valid", "squawk_emergency": False,
      "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
      "corridor_compliant": True, "altitude_ft": 34000, "speed_kts": 450,
      "climb_rate_fpm": 0, "origin_known": True, "grid_square": "G3"},
     {"maritime_track": [
         {"grid_square": "G3", "zone_status": "exclusion_breach", "ais_corroborated": True}]}),
    # (ii) Same but different grid_square → clear (no cue match).
    ("xcue: different grid → clear", "clear",
     {"sensor_source": "ads_b", "iff_mode": "mode3_valid", "squawk_emergency": False,
      "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
      "corridor_compliant": True, "altitude_ft": 34000, "speed_kts": 450,
      "climb_rate_fpm": 0, "origin_known": True, "grid_square": "G3"},
     {"maritime_track": [
         {"grid_square": "H8", "zone_status": "exclusion_breach", "ais_corroborated": True}]}),
    # (iii) Same grid but zone_status=normal → clear (no cue).
    ("xcue: normal zone_status → clear", "clear",
     {"sensor_source": "ads_b", "iff_mode": "mode3_valid", "squawk_emergency": False,
      "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
      "corridor_compliant": True, "altitude_ft": 34000, "speed_kts": 450,
      "climb_rate_fpm": 0, "origin_known": True, "grid_square": "G3"},
     {"maritime_track": [
         {"grid_square": "G3", "zone_status": "normal", "ais_corroborated": True}]}),
    # (iv) ais_corroborated=false → clear (no cue).
    ("xcue: ais_corroborated=false → clear", "clear",
     {"sensor_source": "ads_b", "iff_mode": "mode3_valid", "squawk_emergency": False,
      "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
      "corridor_compliant": True, "altitude_ft": 34000, "speed_kts": 450,
      "climb_rate_fpm": 0, "origin_known": True, "grid_square": "G3"},
     {"maritime_track": [
         {"grid_square": "G3", "zone_status": "exclusion_breach", "ais_corroborated": False}]}),
    # (v) Emergency track + cue → escalate (cue must not interfere with emergency).
    ("xcue: emergency + cue → escalate (no interference)", "escalate",
     {"sensor_source": "radar", "iff_mode": "emergency", "squawk_emergency": True,
      "flight_plan_correlated": False, "transponder_active": True, "in_restricted_zone": False,
      "corridor_compliant": False, "altitude_ft": 12000, "speed_kts": 320,
      "climb_rate_fpm": 0, "origin_known": False, "grid_square": "G3"},
     {"maritime_track": [
         {"grid_square": "G3", "zone_status": "exclusion_breach", "ais_corroborated": True}]}),
    # (vi) Fail-closed: relation row with an uncertifiable cell.
    # An out-of-enum zone_status should fail the per-cell certification.
    ("xcue: uncertifiable cell → refused/fail-closed", None,
     {"sensor_source": "ads_b", "iff_mode": "mode3_valid", "squawk_emergency": False,
      "flight_plan_correlated": True, "transponder_active": True, "in_restricted_zone": False,
      "corridor_compliant": True, "altitude_ft": 34000, "speed_kts": 450,
      "climb_rate_fpm": 0, "origin_known": True, "grid_square": "G3"},
     {"maritime_track": [
         {"grid_square": "G3", "zone_status": "INVALID_VALUE_XYZ", "ais_corroborated": True}]}),
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


def _retry_rate_limit(fn, *, retries: int = 5, base_delay: float = 10.0):
    """Call *fn*; on HTTP 429, back off and retry up to *retries* times."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if code == 429 and attempt < retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"  rate-limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise


def _write_rules_manifest(api: Any, platform_id: int) -> None:
    """Fetch platform rules and write the rules manifest."""
    rules_raw = api.platforms.list_rules(platform_id)

    manifest: list[dict[str, str]] = []
    for r in rules_raw:
        name = r.get("name")
        if not name:
            continue
        entry: dict[str, str] = {"name": str(name)}
        if r.get("rule_type"):
            entry["rule_type"] = str(r["rule_type"])
        if r.get("description"):
            entry["description"] = str(r["description"])
        manifest.append(entry)

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for entry in manifest:
        if entry["name"] not in seen:
            seen.add(entry["name"])
            deduped.append(entry)

    RULES_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RULES_MANIFEST.write_text(json.dumps(deduped, indent=2) + "\n")
    print(f"\nRules manifest written to {RULES_MANIFEST} ({len(deduped)} rules)")
    for entry in deduped:
        print(f"  - {entry['name']}")


# ---- build ---------------------------------------------------------------

def _build_platform(api: Any, *, attempt: int = 1) -> int:
    """Build the xcue platform. Returns the platform_id."""
    print(f"\n=== build attempt {attempt} ===")

    # 1. Domain (reuse by name if present).
    dom = _find_by_name(api.domains.list(), DOMAIN_NAME)
    if dom is None:
        dom = _retry_rate_limit(
            lambda: api.domains.create(name=DOMAIN_NAME, description=DOMAIN_DESCRIPTION))
        print(f"created domain_id={dom['id']}")
    else:
        # If retrying, update the description in case it changed.
        if attempt > 1:
            dom = api.domains.update(dom["id"], description=DOMAIN_DESCRIPTION)
        print(f"reusing domain_id={dom['id']}")
    domain_id = dom["id"]

    # 2. Dataset — UNSUPERVISED: features only, no decision_column.
    ds = next((d for d in api.datasets.list() if d.get("domain_id") == domain_id), None)
    if ds is None:
        ds = _retry_rate_limit(
            lambda: api.datasets.upload(domain_id=domain_id, file_path=str(CSV_PATH)))
        print(f"uploaded dataset_id={ds['id']} rows={ds.get('row_count')} "
              f"cols={ds.get('column_count')} decision_column={ds.get('decision_column')!r}")
    else:
        print(f"reusing dataset_id={ds['id']}")

    # 3. Build ontology with relation declaration.
    onto = _retry_rate_limit(
        lambda: api.domains.build_ontology(domain_id, relations=[MARITIME_RELATION]))
    print(f"building ontology (job {onto['job_id']}) -- waiting...")
    api.wait_for_job(onto["job_id"], timeout=900)

    # 4. Build a VERIFIED platform.
    result = _retry_rate_limit(
        lambda: api.platforms.create(
            domain_id=domain_id, dataset_id=ds["id"], name=PLATFORM_NAME,
            verified_profile=True, verified_min_confidence=0.6,
        ))
    platform_id = result["id"]
    print(f"building platform_id={platform_id} (job {result['job_id']}) -- waiting...")
    api.wait_for_job(result["job_id"], timeout=1800, stall_timeout=600)
    print(f"platform status={api.platforms.status(platform_id)!r}")
    return platform_id


# ---- gate ----------------------------------------------------------------

def _gate_holdout(api: Any, platform_id: int) -> tuple[int, int]:
    """Gate (a): holdout 50 rows with grid_square, no relations. Returns (pass, total)."""
    import csv
    from ambertrace_rlvr.reports import AmberReport

    print("\n--- gate (a): holdout 50/50 ---")

    with open(HOLDOUT_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    passed = 0
    for row in rows:
        expect = row.pop("decision")
        row.pop("triage_reason", None)
        # Add a grid_square (deterministic, seeded from track_id).
        rng_track = random.Random(hash(row["track_id"]) + SEED)
        row["grid_square"] = rng_track.choice(GRID_SQUARES_MODULE)
        # Convert numeric-looking values.
        facts = {}
        for k, v in row.items():
            if k == "track_id":
                continue
            if v in ("0", "1"):
                facts[k] = bool(int(v))
            elif v.lstrip("-").isdigit():
                facts[k] = int(v)
            else:
                facts[k] = v

        res = _retry_rate_limit(
            lambda facts=facts: api.platforms.query(
                platform_id, query="Triage this track.", facts=facts, explain=True))
        rep = AmberReport.from_query_result(res)
        raw = str(rep.decision).lower()
        ok = raw == expect
        if ok:
            passed += 1
        else:
            print(f"  FAIL [{row.get('track_id', '?')}] expect={expect} got={raw}")
    print(f"  holdout: {passed}/{len(rows)}")
    return passed, len(rows)

SEED = 74  # module-level for consistency with gen_air_tracks_xcue_data.py
GRID_SQUARES_MODULE = [f"{letter}{digit}" for letter in "ABCDEFGH" for digit in range(1, 9)]


def _gate_base_probes(api: Any, platform_id: int) -> tuple[int, int]:
    """Gate (b): 6 base policy probes, no relations."""
    from ambertrace_rlvr.reports import AmberReport

    print("\n--- gate (b): base probes 6/6 ---")
    passed = 0
    for label, expect, facts in BASE_PROBES:
        res = _retry_rate_limit(
            lambda facts=facts: api.platforms.query(
                platform_id, query="Triage this track.", facts=facts, explain=True))
        rep = AmberReport.from_query_result(res)
        raw = str(rep.decision).lower()
        ok = raw == expect
        if ok:
            passed += 1
        tag = "OK " if ok else "?? "
        deciding = [d.get("rule", "") for d in rep.deciding_rules]
        print(f"  {tag}[expect {expect:>8}] {label}  decision={rep.decision!r} "
              f"proof_checked={rep.proof_checked} deciding={deciding}")
    print(f"  base probes: {passed}/{len(BASE_PROBES)}")
    return passed, len(BASE_PROBES)


def _gate_xcue_probes(api: Any, platform_id: int) -> tuple[int, int, dict | None]:
    """Gate (c): 6 cross-domain cueing probes. Returns (pass, total, sample_provenance)."""
    from ambertrace_rlvr.reports import AmberReport

    print("\n--- gate (c): xcue probes 6/6 ---")
    passed = 0
    sample_provenance: dict | None = None

    for label, expect, facts, relations in XCUE_PROBES:
        is_fail_closed = expect is None
        try:
            res = _retry_rate_limit(
                lambda facts=facts, relations=relations: api.platforms.query(
                    platform_id, query="Triage this track.",
                    facts=facts, relations=relations, explain=True))
        except Exception as exc:
            if is_fail_closed:
                # Expected: fail-closed on uncertifiable relation cell.
                passed += 1
                print(f"  OK [fail-closed] {label}  error={type(exc).__name__}: {exc}")
                continue
            else:
                print(f"  FAIL {label}  unexpected error: {exc}")
                continue

        if is_fail_closed:
            # The query should have failed; if it returned, check proof_checked.
            rep = AmberReport.from_query_result(res)
            if not rep.proof_checked:
                passed += 1
                print(f"  OK [fail-closed: proof_checked=False] {label}")
            else:
                print(f"  ?? [expected fail-closed but proof_checked=True] {label} "
                      f"decision={rep.decision!r}")
            continue

        rep = AmberReport.from_query_result(res)
        raw = str(rep.decision).lower()
        ok = raw == expect
        if ok:
            passed += 1

        # Capture relation_provenance from the first passing xcue probe.
        expl = rep.raw.get("explanation", {})
        prov = expl.get("relation_provenance")
        if prov and sample_provenance is None:
            sample_provenance = prov

        tag = "OK " if ok else "?? "
        prov_summary = f"provenance={prov}" if prov else "provenance=None"
        print(f"  {tag}[expect {expect:>8}] {label}  decision={rep.decision!r} "
              f"proof_checked={rep.proof_checked} {prov_summary}")

    print(f"  xcue probes: {passed}/{len(XCUE_PROBES)}")
    return passed, len(XCUE_PROBES), sample_provenance


def _dump_failing_rules(api: Any, platform_id: int) -> None:
    """Dump rules for NO-GO diagnostics."""
    print("\n--- NO-GO: dumping rules for diagnostics ---")
    try:
        rules_raw = api.platforms.list_rules(platform_id)
        print(json.dumps(rules_raw, indent=2, default=str))
    except Exception as exc:
        print(f"  could not list rules: {exc}")


# ---- main ----------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"dataset not found: {CSV_PATH}\n"
            "run first: python examples/gen_air_tracks_xcue_data.py"
        )
    _load_dotenv()
    import ambertraceai

    api = ambertraceai.AmbertraceAPI.from_env()

    # Idempotent: reuse a built (active) platform by name.
    existing = next(
        (p for p in api.platforms.list()
         if p.get("name") == PLATFORM_NAME and p.get("status") == "active"),
        None,
    )
    if existing is not None:
        platform_id = existing["id"]
        print(f"reusing active platform_id={platform_id}")
    else:
        platform_id = _build_platform(api)

    # Gate.
    holdout_pass, holdout_total = _gate_holdout(api, platform_id)
    base_pass, base_total = _gate_base_probes(api, platform_id)
    xcue_pass, xcue_total, sample_prov = _gate_xcue_probes(api, platform_id)

    gate_a = holdout_pass == holdout_total
    gate_b = base_pass == base_total
    gate_c = xcue_pass == xcue_total

    print("\n=== GATE RESULTS ===")
    print(f"  (a) holdout:     {holdout_pass}/{holdout_total}  {'GO' if gate_a else 'NO-GO'}")
    print(f"  (b) base probes: {base_pass}/{base_total}  {'GO' if gate_b else 'NO-GO'}")
    print(f"  (c) xcue probes: {xcue_pass}/{xcue_total}  {'GO' if gate_c else 'NO-GO'}")

    if sample_prov:
        print(f"\n  relation_provenance sample: {json.dumps(sample_prov, indent=4)}")

    go = gate_a and gate_b and gate_c
    if go:
        print(f"\n>>> GO — platform_id={platform_id}")
        _write_rules_manifest(api, platform_id)
    else:
        # Retry once — builds are stochastic.
        print("\n--- first build failed gate; retrying once ---")
        platform_id = _build_platform(api, attempt=2)
        holdout_pass, holdout_total = _gate_holdout(api, platform_id)
        base_pass, base_total = _gate_base_probes(api, platform_id)
        xcue_pass, xcue_total, sample_prov = _gate_xcue_probes(api, platform_id)

        gate_a = holdout_pass == holdout_total
        gate_b = base_pass == base_total
        gate_c = xcue_pass == xcue_total

        print("\n=== GATE RESULTS (retry) ===")
        print(f"  (a) holdout:     {holdout_pass}/{holdout_total}  {'GO' if gate_a else 'NO-GO'}")
        print(f"  (b) base probes: {base_pass}/{base_total}  {'GO' if gate_b else 'NO-GO'}")
        print(f"  (c) xcue probes: {xcue_pass}/{xcue_total}  {'GO' if gate_c else 'NO-GO'}")

        if sample_prov:
            print(f"\n  relation_provenance sample: {json.dumps(sample_prov, indent=4)}")

        go = gate_a and gate_b and gate_c
        if go:
            print(f"\n>>> GO — platform_id={platform_id}")
            _write_rules_manifest(api, platform_id)
        else:
            print(f"\n>>> NO-GO — platform_id={platform_id}")
            _dump_failing_rules(api, platform_id)
            sys.exit(1)

    print(f"\nplatform_id={platform_id}")


if __name__ == "__main__":
    main()
