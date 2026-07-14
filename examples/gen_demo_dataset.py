"""Generate the features-only demo dataset for the Grant Eligibility platform.

UNSUPERVISED: no label / decision column — AmberTrace derives the rules from the
domain description + these feature rows. Purely local; no network, no SDK.

    python examples/gen_demo_dataset.py    # writes data/grant_eligibility_dataset.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "grant_eligibility_dataset.csv"
FIELDS = ["age", "annual_income", "resident", "has_active_grant"]

# A spread of realistic feature values either side of each rule boundary, so the
# build sees the full domain of each attribute. No labels.
AGES = (16, 17, 18, 25, 40, 67)
INCOMES = (12000, 25000, 30000, 30001, 45000, 80000)


def build_rows() -> list[dict]:
    rows = []
    for age in AGES:
        for income in INCOMES:
            for resident in (True, False):
                for active in (True, False):
                    rows.append({
                        "age": age,
                        "annual_income": income,
                        "resident": str(resident).lower(),
                        "has_active_grant": str(active).lower(),
                    })
    return rows


def main() -> None:
    rows = build_rows()
    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} feature-only rows to {CSV_PATH} (columns: {FIELDS})")


if __name__ == "__main__":
    main()
