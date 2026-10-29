"""Offline tests for examples/author_decision_eval_v2.py (DryRunBackend + FakeVerifier; no network)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples"))

import author_decision_eval_v2 as A  # noqa: E402

from ambertrace_rlvr import load_decision_corpus  # noqa: E402
from ambertrace_rlvr.math_env.corpus_split import split_by_domain  # noqa: E402


@pytest.fixture(scope="module")
def specs():
    return A.load_specs(REPO / "data" / "decision_eval_v1.jsonl", A.SUBSET)


def test_subset_composition(specs):
    assert len(specs) == sum(A.SUBSET.values()) == 84
    groups = Counter((s.structure, s.family) for s in specs)
    assert dict(groups) == A.SUBSET
    assert min(groups.values()) >= 12
    assert {s.structure for s in specs} == {
        "baseline", "negation", "precedence", "ratio", "multi_trigger_disjunction"}
    assert {s.family for s in specs} == set(A.FAMILIES)


def test_rule_semantics_most_restrictive_wins(specs):
    prec = next(s for s in specs if s.structure == "precedence")
    hi = {f: 1000.0 for f in prec.fields}
    assert prec.label(hi) == "suspend"                       # both fire -> most restrictive
    assert prec.label({f: -1.0 for f in prec.fields}) == "approve"
    med = next(s for s in specs if s.family == "discharge_monitor_escalate")
    facts = {"heart_rate": 10.0, "max_safe_rate": 200.0, "temperature": 36.0,
             "oxygen_saturation": 80.0, "systolic_bp": 200.0, "patient_age": 50.0}
    assert med.label(facts) == "critical_escalate"           # beats monitor (v1 got this wrong)


def test_dry_run_pipeline_shape(specs, tmp_path):
    backend = A.DryRunBackend()
    out = tmp_path / "corpus.jsonl"
    res = A.run_pipeline(specs, backend, tmp_path / "wd", out, log=lambda _m: None)
    assert res["complete"] == 84 and res["incomplete"] == [] and res["items"] == 504
    items = load_decision_corpus(out)
    assert len(items) == 504
    assert len({i.domain for i in items}) == 84
    assert Counter(Counter(i.domain for i in items).values()) == {6: 84}
    assert {i.difficulty["structure"] for i in items} == {
        "baseline", "negation", "precedence", "ratio", "multi_trigger_disjunction"}
    assert {tuple(i.label_space) for i in items} == {tuple(f["verbs"]) for f in A.FAMILIES.values()}
    sp = split_by_domain(items, train_fraction=0.5)
    assert len(sp.structures) == 5
    assert len(sp.train_domains) >= 30 and len(sp.heldout_domains) >= 30
    # split_by_domain guarantees per-STRUCTURE balance (not per-(structure, vocab) group):
    # each structure has >= 6 domains on each side (baseline = all 3 families -> 18/18; the
    # four single-family structures -> 6/6). Do NOT assert per-family balance here: baseline's
    # families are only structure-balanced (e.g. seed 0 approve_deny 8/4), by design.
    dom_struct = {i.domain: i.difficulty["structure"] for i in items}
    tr = Counter(dom_struct[d] for d in sp.train_domains)
    ho = Counter(dom_struct[d] for d in sp.heldout_domains)
    assert all(tr[s] >= 6 and ho[s] >= 6 for s in sp.structures)
    m = json.loads((tmp_path / "wd" / "authoring_manifest.json").read_text())
    assert all(e["step"] == "items_generated" and e["n_items"] == 6
               and e["oracle_agreement"] == "6/6" for e in m["domains"].values())
    assert backend.calls["create_domain"] == 84


def test_dry_run_cli_limit(tmp_path):
    rc = A.main(["--dry-run", "--limit", "1", "--workdir", str(tmp_path / "wd"),
                 "--out", str(tmp_path / "c.jsonl")])
    assert rc == 0
    assert len(load_decision_corpus(tmp_path / "c.jsonl")) == 6


def test_resume_from_truncated_manifest(specs, tmp_path):
    sub = A.interleave_limit(specs, 10)
    wd, out = tmp_path / "wd", tmp_path / "c.jsonl"
    quiet = lambda _m: None  # noqa: E731
    A.run_pipeline(sub, A.DryRunBackend(), wd, out, log=quiet)
    mpath = wd / "authoring_manifest.json"
    m = json.loads(mpath.read_text())

    # Truncate to a mid-state: 4 stay complete; 2 regress to table_set, 2 to ontology_built
    # (ids kept, as after a kill mid-step); 1 loses everything; 1 never started.
    keys = [s.key for s in sub]
    complete, at_table, at_plat, wiped, absent = keys[:4], keys[4:6], keys[6:8], keys[8], keys[9]
    for k in at_table:
        m["domains"][k]["step"] = "table_set"
        for f in ("ontology_job_id", "platform_id", "platform_job_id", "n_items"):
            m["domains"][k].pop(f, None)
        (wd / "items" / f"{k}.jsonl").unlink()
    for k in at_plat:
        m["domains"][k]["step"] = "ontology_built"
        for f in ("platform_id", "platform_job_id", "n_items"):
            m["domains"][k].pop(f, None)
        (wd / "items" / f"{k}.jsonl").unlink()
    del m["domains"][wiped]
    del m["domains"][absent]
    (wd / "items" / f"{wiped}.jsonl").unlink()
    (wd / "items" / f"{absent}.jsonl").unlink()
    mpath.write_text(json.dumps(m))

    backend = A.DryRunBackend()
    res = A.run_pipeline(sub, backend, wd, out, log=quiet)
    assert res["complete"] == 10 and res["incomplete"] == [] and res["items"] == 60

    # Completed domains were skipped; earlier steps of partial ones were not redone.
    assert backend.calls["create_domain"] == 2            # wiped + absent only
    assert backend.calls["upload_dataset"] == 2
    assert backend.calls["set_decision_table"] == 2       # wiped + absent only (table_set ones skip it)
    assert backend.calls["start_ontology"] == 4           # 2x table_set + wiped + absent
    assert backend.calls["create_platform"] == 6          # all but the 4 complete
    assert backend.calls["build_quality"] == 6            # every non-complete domain
    assert backend.calls["verifier"] == 6
    final = json.loads(mpath.read_text())
    assert all(e["step"] == "items_generated" for e in final["domains"].values())
    assert final["domains"][complete[0]] == m["domains"][complete[0]]   # untouched
    assert len(load_decision_corpus(out)) == 60

    # A third run is a pure no-op.
    again = A.DryRunBackend()
    A.run_pipeline(sub, again, wd, out, log=quiet)
    assert sum(again.calls.values()) == 0


def test_resume_repolls_ontology_job_not_rebuild(specs, tmp_path):
    sub = A.interleave_limit(specs, 1)
    wd = tmp_path / "wd"
    mf = A.Manifest(wd / "authoring_manifest.json")
    e = mf.entry(sub[0].key)
    e.update(domain_id=7, dataset_id=8, step="table_set", ontology_job_id="job-X")
    mf.save()
    backend = A.DryRunBackend()
    A.run_pipeline(sub, backend, wd, tmp_path / "c.jsonl", log=lambda _m: None)
    assert backend.calls["start_ontology"] == 0            # reused persisted job id
    assert backend.calls["create_domain"] == 0
    assert backend.calls["wait_job"] == 2                  # ontology job + platform job


def test_failure_is_recorded_and_resumable(specs, tmp_path):
    sub = A.interleave_limit(specs, 2)

    class Flaky(A.DryRunBackend):
        boom = True

        def start_ontology(self, domain_id):
            if self.boom and domain_id == 1000 + int(sub[0].key[1:]):
                raise RuntimeError("transient")
            return super().start_ontology(domain_id)

    wd = tmp_path / "wd"
    res = A.run_pipeline(sub, Flaky(), wd, tmp_path / "c.jsonl", log=lambda _m: None)
    assert res["incomplete"] == [sub[0].key] and res["complete"] == 1
    m = json.loads((wd / "authoring_manifest.json").read_text())
    assert m["domains"][sub[0].key]["step"] == "table_set"
    assert "transient" in m["domains"][sub[0].key]["error"]
    ok = A.DryRunBackend()
    res2 = A.run_pipeline(sub, ok, wd, tmp_path / "c.jsonl", log=lambda _m: None)
    assert res2["incomplete"] == [] and ok.calls["create_domain"] == 0


def test_resume_regenerates_missing_items_cache(specs, tmp_path):
    # outputs/ is a disposable cache: an items file can vanish while the manifest still
    # says items_generated. A re-run must regenerate just that domain (oracle re-query only,
    # no platform rebuild), and assemble must not crash on the missing file.
    sub = A.interleave_limit(specs, 3)
    wd, out = tmp_path / "wd", tmp_path / "c.jsonl"
    quiet = lambda _m: None  # noqa: E731
    A.run_pipeline(sub, A.DryRunBackend(), wd, out, log=quiet)
    victim = sub[0].key
    (wd / "items" / f"{victim}.jsonl").unlink()                 # cache lost, manifest unchanged
    m = json.loads((wd / "authoring_manifest.json").read_text())
    assert m["domains"][victim]["step"] == "items_generated"    # still marked complete

    backend = A.DryRunBackend()
    res = A.run_pipeline(sub, backend, wd, out, log=quiet)
    assert res["complete"] == 3 and res["incomplete"] == [] and res["items"] == 18
    # only the victim regenerated: no domain/platform rebuild, build_quality not redone
    assert backend.calls["create_domain"] == 0
    assert backend.calls["create_platform"] == 0
    assert backend.calls["build_quality"] == 0
    assert backend.calls["verifier"] == 1                       # oracle re-query for the one domain
    assert (wd / "items" / f"{victim}.jsonl").exists()
    assert len(load_decision_corpus(out)) == 18
