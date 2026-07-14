"""Author a small **demo verified platform** with the `ambertraceai` SDK, then
verify it certifies a query — the "build" half of the create → build → train
journey (README).

This is an **operator / setup script, NOT library code**. `ambertrace-rlvr`'s
reward runtime is read-only against AmberTrace; authoring a platform is a
customer step done with the SDK, which this script demonstrates. Nothing here is
imported by `src/ambertrace_rlvr/`.

AmberTrace learns **unsupervised**: we upload a *features-only* dataset (no
labels, no `decision_column`) and let the neurosymbolic kernel derive the rules
from the plain-English domain description + the data. On a verified build every
query carries a machine-checked proof (fail-closed on uncertifiable queries) —
that certificate is what the RLVR reward consumes.

Usage:

    # needs an AMBERTRACE_API_KEY with authoring scope (via env or .env)
    python examples/author_demo_platform.py

Prints the new ``platform_id`` — put it in ``configs/grant_eligibility.yaml``
(or set ``AMBERTRACE_PLATFORM_ID``) to train against it.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CSV_PATH = REPO / "data" / "grant_eligibility_dataset.csv"

DOMAIN_NAME = "RLVR Demo — Grant Eligibility"
PLATFORM_NAME = "RLVR Demo — Grant Eligibility Platform"

# The decision logic lives here, in plain English — NOT in dataset labels.
DOMAIN_DESCRIPTION = (
    "Decide whether an applicant is eligible for a basic means-tested support grant. "
    "An applicant is eligible (decision 'permit') only if ALL of the following hold: "
    "they are at least 18 years old; their annual income is at most 30000; "
    "they are a resident; and they do not already hold an active grant. "
    "If any condition fails, the decision is 'deny'."
)

# A known case used to smoke-test the built platform.
PERMIT_FACTS = {
    "age": 40, "annual_income": 25000, "resident": True, "has_active_grant": False,
}
DENY_FACTS = {
    "age": 40, "annual_income": 45000, "resident": True, "has_active_grant": False,
}


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
            "generate it first: python examples/gen_demo_dataset.py"
        )
    _load_dotenv()
    import ambertraceai

    api = ambertraceai.AmbertraceAPI.from_env()

    # Idempotent: if a built (active) platform already exists by name, reuse it
    # and skip straight to verification, so re-running this script is safe.
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

    # 2. Dataset — UNSUPERVISED: features only, no decision_column. Reuse an
    # existing dataset on this domain if present.
    # NB: don't pass name= — the SDK uses it as the multipart filename, and a
    # name without a ".csv" suffix makes the server reject the file type.
    ds = next((d for d in api.datasets.list() if d.get("domain_id") == domain_id), None)
    if ds is None:
        ds = api.datasets.upload(domain_id=domain_id, file_path=str(CSV_PATH))
        print(f"uploaded dataset_id={ds['id']} rows={ds.get('row_count')} "
              f"cols={ds.get('column_count')} decision_column={ds.get('decision_column')!r}")
    else:
        print(f"reusing dataset_id={ds['id']}")

    # 3. Build the ontology + rules from the description + data (entities must
    #    exist before a platform can be built).
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
    """Query a known permit and deny case and print the certified outcome, read
    through the library's ``AmberReport`` — exactly the reward path in ``src/``."""
    from ambertrace_rlvr.reports import AmberReport

    for label, facts in (("permit", PERMIT_FACTS), ("deny", DENY_FACTS)):
        res = api.platforms.query(
            platform_id, query="Assess this grant application.", facts=facts, explain=True,
        )
        rep = AmberReport.from_query_result(res)
        print(f"  [expect {label}] decision={rep.decision!r} "
              f"proof_checked={rep.proof_checked} confidence={rep.confidence} "
              f"rules_fired={len(rep.rules_fired)} rejected={len(rep.rejected_facts)}")


def _report(platform_id: int) -> None:
    print(f"\nDONE. platform_id={platform_id}")
    print("→ set this as domain.platform_id in configs/grant_eligibility.yaml")


if __name__ == "__main__":
    main()
