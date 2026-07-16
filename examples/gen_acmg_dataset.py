"""Generate the features-only dataset for the ACMG Variant Classification platform.

UNSUPERVISED: no label / decision column — AmberTrace derives the rules from the
plain-English domain description + these feature rows (see author_acmg_platform.py).
Purely local; no network, no SDK.

The domain is a *simplified* ACMG/AMP sequence-variant classifier. Crucially it has
BOTH pathogenic and benign criteria (so benign is a positively-derived class, not a
bare "otherwise" default), with 'uncertain' (VUS) as the honest residual. Each row is
the evidence gathered for one variant, encoded as the presence/absence of four ACMG
criteria:

  * null_variant_in_disease_gene  — PVS1 (pathogenic): a predicted loss-of-function
                                     (null) variant in a gene where LoF causes disease.
  * functional_studies_damaging   — PS3 (pathogenic): well-established functional
                                     studies show a damaging effect.
  * common_in_population           — BA1 (benign): allele frequency > 5% in large
                                     population databases (gnomAD).
  * functional_studies_benign     — BS3 (benign): well-established functional studies
                                     show NO damaging effect.

    python examples/gen_acmg_dataset.py    # writes data/acmg_variants.csv
"""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "acmg_variants.csv"
FIELDS = [
    "null_variant_in_disease_gene",
    "functional_studies_damaging",
    "common_in_population",
    "functional_studies_benign",
]
# Each boolean combination repeated so the build sees a healthy, balanced domain
# for every feature (no labels — the classification lives in the description).
REPEATS = 6


def build_rows() -> list[dict]:
    rows = []
    for combo in product((True, False), repeat=len(FIELDS)):
        for _ in range(REPEATS):
            rows.append({f: str(v).lower() for f, v in zip(FIELDS, combo)})
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
