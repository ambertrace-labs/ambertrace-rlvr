"""Invariants for the cross-domain cueing datasets (seeded generators)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AIR_TRACKS_XCUE = REPO / "data" / "air_tracks_xcue.csv"
MARITIME_TRACKS = REPO / "data" / "maritime_tracks.csv"
GENERATOR = REPO / "examples" / "gen_air_tracks_xcue_data.py"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_air_tracks_xcue_present_and_correct_shape():
    assert AIR_TRACKS_XCUE.exists(), "air_tracks_xcue.csv missing"
    rows = _read_csv(AIR_TRACKS_XCUE)
    assert len(rows) == 500
    assert "grid_square" in rows[0]
    assert "track_id" in rows[0]


def test_maritime_tracks_present_and_correct_shape():
    assert MARITIME_TRACKS.exists(), "maritime_tracks.csv missing"
    rows = _read_csv(MARITIME_TRACKS)
    assert len(rows) == 200
    for col in ("maritime_track_id", "grid_square", "zone_status", "ais_corroborated"):
        assert col in rows[0], f"missing column: {col}"


def test_maritime_zone_status_values():
    rows = _read_csv(MARITIME_TRACKS)
    allowed = {"normal", "advisory", "exclusion_breach"}
    for row in rows:
        assert row["zone_status"] in allowed, f"bad zone_status: {row['zone_status']}"


def test_generator_is_deterministic():
    """Re-running the generator produces identical output."""
    before_air = AIR_TRACKS_XCUE.read_bytes()
    before_mar = MARITIME_TRACKS.read_bytes()
    subprocess.check_call([sys.executable, str(GENERATOR)], stdout=subprocess.DEVNULL)
    assert AIR_TRACKS_XCUE.read_bytes() == before_air, "air_tracks_xcue.csv changed on re-run"
    assert MARITIME_TRACKS.read_bytes() == before_mar, "maritime_tracks.csv changed on re-run"
