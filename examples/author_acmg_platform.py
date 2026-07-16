"""Author a **verified ACMG Variant Classification platform** with the `ambertraceai`
SDK, then verify it certifies a query — the "build" half of the create -> build ->
train journey (README) for a scientific-decision domain.

This is an **operator / setup script, NOT library code**. `ambertrace-rlvr`'s reward
runtime is read-only against AmberTrace; authoring a platform is a customer step done
with the SDK, which this script demonstrates. Nothing here is imported by
`src/ambertrace_rlvr/`.

The domain is a *simplified* ACMG/AMP sequence-variant classifier with BOTH pathogenic
and benign criteria, so every class is positively derived (fires its own rules) rather
than a bare "otherwise" default, with 'uncertain' (VUS) as the honest residual.

AmberTrace learns **unsupervised**: we upload the features-only dataset
(examples/gen_acmg_dataset.py) and a plain-English description, and the neurosymbolic
kernel derives the rules. On a verified build every query carries a machine-checked
proof (fail-closed on uncertifiable queries) — that certificate is what the RLVR reward
consumes.

Usage:

    # needs an AMBERTRACE_API_KEY with authoring scope (via env or .env)
    python examples/gen_acmg_dataset.py       # writes data/acmg_variants.csv
    python examples/author_acmg_platform.py   # builds the platform, prints platform_id

Prints the new ``platform_id`` — put it in ``configs/acmg.yaml`` (or set
``AMBERTRACE_PLATFORM_ID``) to train against it.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "acmg_variants.csv"

DOMAIN_NAME = "RLVR Demo — ACMG Variant Classification"
PLATFORM_NAME = "RLVR Demo — ACMG Variant Classification Platform"

# The classification logic lives here, in plain English — NOT in dataset labels.
# NB: it is framed as symmetric *label assignment* (not an imperative "classify X"),
# which keeps the verified builder's verdict vocabulary clean — the three labels
# surface as classify_pathogenic / classify_benign / abstain (= uncertain).
DOMAIN_DESCRIPTION = (
    "A genomic sequence variant is assigned exactly one classification label, chosen "
    "from three category values of equal standing: 'pathogenic', 'benign', or "
    "'uncertain'. The label is determined by simplified ACMG/AMP evidence criteria. "
    "Two criteria count as pathogenic evidence: (PVS1) the variant is a predicted "
    "loss-of-function variant in a gene where loss of function is a known disease "
    "mechanism; and (PS3) well-established functional studies show the variant has a "
    "damaging effect. Two criteria count as benign evidence: (BA1) the variant is common "
    "in the general population; and (BS3) well-established functional studies show the "
    "variant has no damaging effect. The variant has pathogenic evidence if at least one "
    "pathogenic criterion is met, and benign evidence if at least one benign criterion is "
    "met. The classification label is 'pathogenic' when the variant has pathogenic "
    "evidence and no benign evidence. The classification label is 'benign' when the "
    "variant has benign evidence and no pathogenic evidence. The classification label is "
    "'uncertain' when the variant has neither pathogenic nor benign evidence, or when it "
    "has both and the evidence conflicts."
)

# Known cases used to smoke-test the built platform. The verdict layer surfaces the
# class labels as `classify_<label>` and maps 'uncertain' to the canonical 'abstain'.
CASES = [
    ("pathogenic", {"null_variant_in_disease_gene": True, "functional_studies_damaging": False,
                    "common_in_population": False, "functional_studies_benign": False}),
    ("pathogenic", {"null_variant_in_disease_gene": False, "functional_studies_damaging": True,
                    "common_in_population": False, "functional_studies_benign": False}),
    ("benign", {"null_variant_in_disease_gene": False, "functional_studies_damaging": False,
                "common_in_population": True, "functional_studies_benign": False}),
    ("benign", {"null_variant_in_disease_gene": False, "functional_studies_damaging": False,
                "common_in_population": False, "functional_studies_benign": True}),
    ("uncertain", {"null_variant_in_disease_gene": False, "functional_studies_damaging": False,
                   "common_in_population": False, "functional_studies_benign": False}),
    ("uncertain", {"null_variant_in_disease_gene": True, "functional_studies_damaging": False,
                   "common_in_population": True, "functional_studies_benign": False}),
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


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"dataset not found: {CSV_PATH}\n"
            "generate it first: python examples/gen_acmg_dataset.py"
        )
    _load_dotenv()
    import ambertraceai

    api = ambertraceai.AmbertraceAPI.from_env()

    # Idempotent: reuse a built (active) platform by name so re-runs are cheap.
    existing = next((p for p in api.platforms.list()
                     if p.get("name") == PLATFORM_NAME and p.get("status") == "active"), None)
    if existing is not None:
        platform_id = existing["id"]
        print(f"reusing active platform_id={platform_id}")
        _verify(api, platform_id)
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

    # 2. Dataset — UNSUPERVISED: features only, no decision_column.
    ds = next((d for d in api.datasets.list() if d.get("domain_id") == domain_id), None)
    if ds is None:
        ds = api.datasets.upload(domain_id=domain_id, file_path=str(CSV_PATH))
        print(f"uploaded dataset_id={ds['id']} rows={ds.get('row_count')} "
              f"cols={ds.get('column_count')} decision_column={ds.get('decision_column')!r}")
    else:
        print(f"reusing dataset_id={ds['id']}")

    # 3. Build the ontology + rules from the description + data.
    onto = api.domains.build_ontology(domain_id)
    print(f"building ontology (job {onto['job_id']}) — waiting…")
    api.wait_for_job(onto["job_id"])

    # 4. Build a VERIFIED platform (machine-checked proof per query, fail-closed).
    result = api.platforms.create(
        domain_id=domain_id, dataset_id=ds["id"], name=PLATFORM_NAME,
        verified_profile=True, verified_min_confidence=0.85,
    )
    platform_id = result["id"]
    print(f"building platform_id={platform_id} (job {result['job_id']}) — waiting…")
    api.wait_for_job(result["job_id"])
    print(f"platform status={api.platforms.status(platform_id)!r}")

    _verify(api, platform_id)
    _report(platform_id)


def _verify(api, platform_id: int) -> None:
    """Query the known cases and print the certified outcome, read through the
    library's ``AmberReport`` — exactly the reward path in ``src/``."""
    from ambertrace_rlvr.reports import AmberReport

    for expect, facts in CASES:
        res = api.platforms.query(
            platform_id, query="Classify this sequence variant.", facts=facts, explain=True,
        )
        rep = AmberReport.from_query_result(res)
        raw = str(rep.decision).lower()
        got = {"abstain": "uncertain"}.get(raw, raw.removeprefix("classify_"))
        match = "OK " if got == expect else "?? "
        print(f"  {match}[expect {expect:>10}] decision={rep.decision!r} "
              f"proof_checked={rep.proof_checked} confidence={rep.confidence:.2f} "
              f"rules_fired={len(rep.rules_fired)}")


def _report(platform_id: int) -> None:
    print(f"\nDONE. platform_id={platform_id}")
    print("→ set this as domain.platform_id in configs/acmg.yaml")


if __name__ == "__main__":
    main()
