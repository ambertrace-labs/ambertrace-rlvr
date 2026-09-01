"""Admission pipeline (#103): offline tests for certify_candidates, the registry,
and the adversarial loop. All network-free via FakeVerifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ambertrace_rlvr.eval_admission import (
    AdmissionRegistry,
    CandidateItem,
    GeneratorProvenance,
    certify_candidates,
)
from ambertrace_rlvr.eval_oracle import JudgmentSpec, LabelSpec
from ambertrace_rlvr.testing import FakeVerifier, make_report

# Access-control vocabulary (deny/escalate/approve) — same as the shipped set.
VOCAB = (
    LabelSpec("deny", rank=0, restrictive=True),
    LabelSpec("escalate", rank=3, restrictive=True),
    LabelSpec("approve", rank=9, restrictive=False),
)
SPEC = JudgmentSpec(labels=list(VOCAB))

PROV = GeneratorProvenance(generator_id="test", seed=42)


def _candidate(
    *,
    prompt: str = "A valid prompt that is long enough to pass sanity checks.",
    facts: dict[str, Any] | None = None,
    oracle: str = "deny",
    band: str = "restrictive",
    stratum: str = "baseline",
    cid: str | None = None,
) -> CandidateItem:
    return CandidateItem(
        prompt=prompt,
        facts=facts or {"_truth": oracle},
        vocabulary=VOCAB,
        intended_oracle=oracle,
        intended_direction=band,
        intended_stratum=stratum,
        intended_band=band,
        provenance=PROV,
        difficulty={"family": stratum, "structure": stratum},
        id=cid,
    )


def _truth_verifier() -> FakeVerifier:
    """A FakeVerifier that certifies the verdict from facts["_truth"]."""
    def report_fn(parsed: Any) -> Any:
        truth = parsed.facts.get("_truth")
        return make_report(proof_checked=True, decision=truth)
    return FakeVerifier(report_fn=report_fn)


# --- basic admit/reject paths ------------------------------------------------

def test_admit_valid_candidate():
    results = certify_candidates([_candidate()], _truth_verifier(), spec=SPEC)
    assert len(results) == 1
    r = results[0]
    assert r.admitted
    assert r.item is not None
    assert r.item.oracle == "deny"
    assert r.reasons == ()


def test_admit_permissive_candidate():
    c = _candidate(oracle="approve", band="permissive")
    results = certify_candidates([c], _truth_verifier(), spec=SPEC)
    assert results[0].admitted
    assert results[0].item is not None
    assert results[0].item.oracle == "approve"


def test_reject_undecidable():
    """An oracle that returns a certified-undecidable verdict should be rejected
    (not a decidable eval item)."""
    abstain_vocab = (
        LabelSpec("deny", rank=0, restrictive=True),
        LabelSpec("approve", rank=1, restrictive=False),
        LabelSpec("abstain", rank=2, restrictive=False, is_abstain=True),
    )
    spec = JudgmentSpec(labels=list(abstain_vocab))

    def report_fn(parsed: Any) -> Any:
        return make_report(proof_checked=True, decision="abstain")

    c = CandidateItem(
        prompt="An ambiguous case that the oracle cannot decide upon clearly.",
        facts={},
        vocabulary=abstain_vocab,
        intended_oracle="deny",
        intended_direction="restrictive",
        intended_stratum="baseline",
        intended_band="restrictive",
        provenance=PROV,
    )
    results = certify_candidates([c], FakeVerifier(report_fn=report_fn), spec=spec)
    assert not results[0].admitted
    assert any("undecidable" in r for r in results[0].reasons)


def test_reject_unverifiable():
    """An oracle that fails to certify (no checked proof) should be rejected."""
    def report_fn(parsed: Any) -> Any:
        return make_report(proof_checked=False, decision=None)

    results = certify_candidates(
        [_candidate()], FakeVerifier(report_fn=report_fn), spec=SPEC)
    assert not results[0].admitted
    assert any("did not certify" in r for r in results[0].reasons)


def test_reject_wrong_oracle():
    """If the oracle's verdict differs from the intended oracle, reject with evidence."""
    def report_fn(parsed: Any) -> Any:
        return make_report(proof_checked=True, decision="approve")

    c = _candidate(oracle="deny", band="restrictive")
    results = certify_candidates([c], FakeVerifier(report_fn=report_fn), spec=SPEC)
    r = results[0]
    assert not r.admitted
    assert any("intended oracle" in reason for reason in r.reasons)
    # The actual oracle is recorded for generator diagnostics.
    assert r.actual_oracle == "approve"


def test_reject_wrong_band():
    """If the actual severity band differs from the intended band, reject."""
    # oracle returns "approve" (permissive) but candidate says band is restrictive.
    c = _candidate(oracle="approve", band="restrictive")
    results = certify_candidates([c], _truth_verifier(), spec=SPEC)
    r = results[0]
    assert not r.admitted
    assert any("band" in reason for reason in r.reasons)


def test_reject_short_prompt():
    c = _candidate(prompt="too short")
    results = certify_candidates([c], _truth_verifier(), spec=SPEC)
    assert not results[0].admitted
    assert any("too short" in r for r in results[0].reasons)


def test_reject_long_prompt():
    c = _candidate(prompt="x" * 9000)
    results = certify_candidates([c], _truth_verifier(), spec=SPEC)
    assert not results[0].admitted
    assert any("too long" in r for r in results[0].reasons)


# --- deduplication ------------------------------------------------------------

def test_reject_duplicate_of_existing_corpus():
    c = _candidate()
    existing = frozenset([
        # Pre-compute the hash of the candidate's (prompt, facts).
        __import__("ambertrace_rlvr.eval_admission", fromlist=["_content_hash"])
        ._content_hash(c.prompt, c.facts)
    ])
    results = certify_candidates([c], _truth_verifier(), spec=SPEC,
                                  existing_hashes=existing)
    assert not results[0].admitted
    assert any("duplicate" in r for r in results[0].reasons)


def test_reject_duplicate_within_batch():
    c = _candidate()
    results = certify_candidates([c, c], _truth_verifier(), spec=SPEC)
    # First should be admitted, second rejected as duplicate.
    assert results[0].admitted
    assert not results[1].admitted
    assert any("duplicate within this batch" in r for r in results[1].reasons)


# --- registry -----------------------------------------------------------------

def test_registry_check_conflict():
    reg = AdmissionRegistry()
    reg.register("hash1", "train-env")
    assert reg.check_conflict("hash1", "eval") is not None
    assert reg.check_conflict("hash1", "train-env") is None
    assert reg.check_conflict("hash_new", "eval") is None


def test_registry_blocks_admission():
    reg = AdmissionRegistry()
    c = _candidate()
    from ambertrace_rlvr.eval_admission import _content_hash
    ch = _content_hash(c.prompt, c.facts)
    reg.register(ch, "train-env")

    results = certify_candidates([c], _truth_verifier(), spec=SPEC, registry=reg)
    assert not results[0].admitted
    assert any("registry conflict" in r for r in results[0].reasons)


def test_registry_round_trip(tmp_path: Path):
    reg = AdmissionRegistry(path=tmp_path / "registry.json")
    reg.register("hash_a", "eval", provenance="gen_1")
    reg.register("hash_b", "train-env", model="qwen-7b")
    reg.save()

    loaded = AdmissionRegistry.load(tmp_path / "registry.json")
    assert loaded.entries["hash_a"].role == "eval"
    assert loaded.entries["hash_b"].model == "qwen-7b"
    assert loaded.check_conflict("hash_a", "train-env") is not None


def test_registry_hashes_for_role():
    reg = AdmissionRegistry()
    reg.register("h1", "eval")
    reg.register("h2", "train-env")
    reg.register("h3", "eval")
    assert reg.hashes_for_role("eval") == frozenset(["h1", "h3"])
    assert reg.hashes_for_role("train-env") == frozenset(["h2"])


# --- provenance round-trip ----------------------------------------------------

def test_provenance_round_trip():
    c = _candidate()
    results = certify_candidates([c], _truth_verifier(), spec=SPEC)
    assert results[0].admitted
    assert results[0].candidate.provenance.generator_id == "test"
    assert results[0].candidate.provenance.seed == 42


# --- batch chunking -----------------------------------------------------------

def test_batch_chunking():
    """Verify that candidates are processed correctly even when the batch exceeds
    the chunk size (50)."""
    candidates = [
        _candidate(cid=f"item-{i}", prompt=f"Prompt number {i} which is long enough to pass checks.")
        for i in range(55)
    ]
    results = certify_candidates(candidates, _truth_verifier(), spec=SPEC)
    assert len(results) == 55
    admitted = [r for r in results if r.admitted]
    # First 55 unique prompts — the first should be admitted, rest may have dups.
    # Since each prompt is unique (different i), all should be admitted.
    assert len(admitted) == 55


# --- adversarial loop ---------------------------------------------------------

def test_adversarial_loop_finds_fail_open():
    """The adversarial loop should find fail-open traps when the model always
    approves (fails open on deny items)."""
    from ambertrace_rlvr.eval_gaps import AdversarialTarget

    from examples.gen_agent_evals import ACCESS_SPEC, adversarial_loop, adversarial_proposer

    verifier = _truth_verifier()
    proposer = adversarial_proposer(seed=42)

    # A model that always says "approve" — fails open on deny items.
    def always_approve(_prompt: str) -> str:
        return "approve"

    target = AdversarialTarget(
        stratum="ratio", band="restrictive", structure="ratio",
        fail_open_rate=0.5, model="fake-always-approve",
    )

    found = adversarial_loop(
        proposer, target, verifier, always_approve,
        spec=ACCESS_SPEC, max_rounds=1,
    )
    # Should find at least the deny-oracle item (model says approve, oracle says deny).
    assert len(found) >= 1
    for r in found:
        assert r.admitted
        assert r.item is not None
        assert r.item.oracle == "deny"


def test_adversarial_loop_skips_correct_model():
    """When the model answers correctly, the adversarial loop should not keep
    those items."""
    from ambertrace_rlvr.eval_gaps import AdversarialTarget

    from examples.gen_agent_evals import ACCESS_SPEC, adversarial_loop, adversarial_proposer

    verifier = _truth_verifier()
    proposer = adversarial_proposer(seed=42)

    # A model that always answers correctly (reads the oracle).
    def correct_model(prompt: str) -> str:
        # The adversarial templates have deny and approve as oracles.
        if "NOT restricted" in prompt:
            return "approve"
        return "deny"

    target = AdversarialTarget(
        stratum="ratio", band="restrictive", structure="ratio",
        fail_open_rate=0.5, model="fake-correct",
    )

    found = adversarial_loop(
        proposer, target, verifier, correct_model,
        spec=ACCESS_SPEC, max_rounds=1,
    )
    assert len(found) == 0


# --- empty inputs -------------------------------------------------------------

def test_empty_candidates():
    results = certify_candidates([], _truth_verifier(), spec=SPEC)
    assert results == []
