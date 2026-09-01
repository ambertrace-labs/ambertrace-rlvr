"""Generate seeded datasets for the cross-domain cueing air-track platform.

Produces two deterministic CSVs:

  1. ``data/air_tracks_xcue.csv`` — the original 500 air-track training rows with
     a ``grid_square`` column appended (deterministic, seeded).
  2. ``data/maritime_tracks.csv`` — 200 synthetic maritime-track rows, each with
     ``grid_square``, ``zone_status``, and ``ais_corroborated`` columns.

Both files are seeded (``seed=74``) and fully deterministic — re-running produces
identical output. The maritime tracks are NOT uploaded as a relation dataset; they
exist only as a reference for generating query-time ``relations`` payloads. The
relation schema is declared at ``build_ontology`` time; rows are supplied per query.

Usage::

    python examples/gen_air_tracks_xcue_data.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AIR_TRACKS_SRC = REPO / "data" / "air_tracks.csv"
AIR_TRACKS_XCUE = REPO / "data" / "air_tracks_xcue.csv"
MARITIME_TRACKS = REPO / "data" / "maritime_tracks.csv"

SEED = 74
GRID_SQUARES = [f"{letter}{digit}" for letter in "ABCDEFGH" for digit in range(1, 9)]

ZONE_STATUSES = ["normal", "advisory", "exclusion_breach"]
ZONE_STATUS_WEIGHTS = [0.6, 0.25, 0.15]


def generate_air_tracks_xcue() -> None:
    """Read the base air_tracks.csv, append a seeded grid_square column, write."""
    rng = random.Random(SEED)
    with open(AIR_TRACKS_SRC, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        fieldnames = list(reader.fieldnames) + ["grid_square"]
        rows = list(reader)

    for row in rows:
        row["grid_square"] = rng.choice(GRID_SQUARES)

    AIR_TRACKS_XCUE.parent.mkdir(parents=True, exist_ok=True)
    with open(AIR_TRACKS_XCUE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {AIR_TRACKS_XCUE} ({len(rows)} rows)")


def generate_maritime_tracks() -> None:
    """Generate 200 seeded maritime-track rows."""
    rng = random.Random(SEED + 1)
    fieldnames = ["maritime_track_id", "grid_square", "zone_status", "ais_corroborated"]
    rows: list[dict[str, str | bool]] = []
    for i in range(1, 201):
        gs = rng.choice(GRID_SQUARES)
        zs = rng.choices(ZONE_STATUSES, weights=ZONE_STATUS_WEIGHTS, k=1)[0]
        ais = rng.random() < 0.7
        rows.append({
            "maritime_track_id": f"MAR-{i:06d}",
            "grid_square": gs,
            "zone_status": zs,
            "ais_corroborated": ais,
        })

    MARITIME_TRACKS.parent.mkdir(parents=True, exist_ok=True)
    with open(MARITIME_TRACKS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {MARITIME_TRACKS} ({len(rows)} rows)")


def main() -> None:
    generate_air_tracks_xcue()
    generate_maritime_tracks()
    print("done")


if __name__ == "__main__":
    main()
