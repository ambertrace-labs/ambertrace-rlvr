"""Re-author a representative subset of ``decision_eval_v1`` CORRECTLY, via SDK decision tables.

Why: the v1 oracle labels came from platforms authored without a declared decision table,
so e.g. precedence ("most restrictive fired verb wins") was mislabelled. This script
re-authors each selected v1 domain on a live AmberTrace platform with a declared **ladder
decision table** (the current-SDK path; see the verified reference pipeline), then
re-labels fresh case prompts through the platform oracle with ``build_eval_items`` so the
vocabulary AND the verdict come from the same validated table.

Per domain (v1 policy text is reused verbatim; its rules are parsed to synthesise a
labelled feature CSV and the case facts):

    domains.create -> datasets.upload -> domains.set_decision_table -> build_ontology
    -> platforms.create -> platforms.build_quality -> build_eval_items (oracle labels)

RESUMABLE at per-domain, per-step granularity. ``<workdir>/authoring_manifest.json``
records, per domain key, the SDK ids and a ``step`` (one of ``STEPS``). A re-run skips
domains at ``items_generated`` and resumes the rest from their last completed step; async
builds (ontology, platform) persist their ``job_id`` the moment they are submitted, so an
interrupted wait is resumed by re-polling that job, never by rebuilding. The manifest is
rewritten (atomically) after every state transition. Reuse-by-name against the platform
is a secondary guard for a lost manifest. A domain that raises is recorded (``error``) and
the run continues with the next domain; the exit code is non-zero if any are incomplete.
If a server-side job FAILED (as opposed to timed out), clear its ``*_job_id`` in the
manifest to force a rebuild.

    # offline: whole pipeline with FakeVerifier, no network (writes to a scratch workdir)
    python examples/author_decision_eval_v2.py --dry-run --limit 3

    # print the LabelSpecs derived from an existing platform's decision table
    python examples/author_decision_eval_v2.py --validate-only 123

    # live (operator-gated; needs AMBERTRACE_API_KEY): proof-of-one, then full subset
    python examples/author_decision_eval_v2.py --limit 1
    python examples/author_decision_eval_v2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- constants

# (structure, vocabulary family) -> number of v1 domains to re-author (first N by domain id).
# v1 availability: baseline x {approve/deny 12, clear/flag 12, 4-way 21}, and 45 each for
# negation (approve/deny), precedence (approve/restrict/suspend), ratio (clear/flag),
# multi_trigger_disjunction (clear/flag). Every group needs >= 12 domains so the
# structure-stratified split_by_domain(train_fraction=0.5) leaves >= 6 train / >= 6 held-out.
# All seven (structure, vocabulary) groups are kept at 12 each (the two baseline 2-verb groups
# have exactly 12 domains available), so baseline carries all three of its vocab families and no
# family/structure cell is dropped. 7 * 12 = 84 domains = 504 items; all 5 structures, all 4
# families; every group splits 6 train / 6 held-out.
SUBSET: dict[tuple[str, str], int] = {
    ("baseline", "discharge_monitor_escalate"): 12,
    ("baseline", "approve_deny"): 12,
    ("baseline", "clear_flag"): 12,
    ("negation", "approve_deny"): 12,
    ("precedence", "approve_restrict_suspend"): 12,
    ("ratio", "clear_flag"): 12,
    ("multi_trigger_disjunction", "clear_flag"): 12,
}
SUBSETS: dict[str, dict[tuple[str, str], int]] = {"full4": SUBSET}

ITEMS_PER_DOMAIN = 6      # matches v1 (6 prompts per domain)
CSV_ROWS = 320            # authoring rows per domain (label-stratified)
SEED = 0

# Ordered most-permissive first; higher ``order`` = more restrictive (ladder semantics).
FAMILIES: dict[str, dict[str, Any]] = {
    "approve_deny": {
        "verbs": ["approve", "deny"], "ratio_verb": "deny",
        "table": {"mode": "ladder", "rows": [
            {"label": "approve", "order": 1, "permits": True, "default": False},
            {"label": "deny", "order": 2, "permits": False, "default": True}]},
    },
    "clear_flag": {
        "verbs": ["clear", "flag"], "ratio_verb": "flag",
        "table": {"mode": "ladder", "rows": [
            {"label": "clear", "order": 1, "permits": True, "default": False},
            {"label": "flag", "order": 2, "permits": False, "default": True}]},
    },
    "approve_restrict_suspend": {
        "verbs": ["approve", "restrict", "suspend"], "ratio_verb": "restrict",
        "table": {"mode": "ladder", "rows": [
            {"label": "approve", "order": 1, "permits": True, "default": False},
            {"label": "restrict", "order": 2, "permits": False, "default": False},
            {"label": "suspend", "order": 3, "permits": False, "default": True}]},
    },
    "discharge_monitor_escalate": {
        "verbs": ["discharge", "monitor", "escalate", "critical_escalate"],
        "ratio_verb": "escalate",
        "table": {"mode": "ladder", "rows": [
            {"label": "discharge", "order": 1, "permits": True, "default": False},
            {"label": "monitor", "order": 2, "permits": False, "default": False},
            {"label": "escalate", "order": 3, "permits": False, "default": False},
            {"label": "critical_escalate", "order": 4, "permits": False, "default": True}]},
    },
}

# Resume ladder: a domain's ``step`` is the LAST COMPLETED of these.
STEPS = ("created", "dataset_uploaded", "table_set", "ontology_built",
         "platform_active", "quality_ok", "items_generated")

CATEGORICAL = {"employment_status": ["employed", "unemployed", "self_employed"],
               "zone_type": ["protected", "industrial", "standard"]}
PARTICIPLE_VERB = {"denied": "deny", "flagged": "flag", "flagged for review": "flag",
                   "escalated": "escalate", "critically escalated": "critical_escalate",
                   "monitored": "monitor", "restricted": "restrict", "suspended": "suspend"}


# --------------------------------------------------------------------------- domain specs

@dataclass(frozen=True)
class Rule:
    """One parsed policy trigger. ``kind``: ratio | above | below | false | equals."""
    kind: str
    field: str
    value: float | str = 0.0
    other: str = ""        # denominator field for ``ratio``
    verb: str = ""


@dataclass
class DomainSpec:
    key: str               # v1 domain id, e.g. "d002"
    structure: str
    family: str
    policy: str            # v1 policy paragraph (reused verbatim)
    fields: list[str]      # feature names, in v1 order
    sample_facts: dict[str, Any]   # one v1 case, used only to infer field types
    rules: list[Rule] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"RLVR decision_eval_v2 {self.key} ({self.structure}/{self.family})"

    @property
    def platform_name(self) -> str:
        return self.name + " Platform"

    @property
    def table(self) -> dict[str, Any]:
        return FAMILIES[self.family]["table"]

    @property
    def verbs(self) -> list[str]:
        return list(FAMILIES[self.family]["verbs"])

    def label(self, facts: dict[str, Any]) -> str:
        """Intended verdict: the most restrictive fired verb wins; else the permit verb."""
        order = {r["label"]: r["order"] for r in self.table["rows"]}
        fired = [r.verb for r in self.rules if _fires(r, facts)]
        return max(fired, key=lambda v: order[v]) if fired else self.verbs[0]


def _fires(r: Rule, f: dict[str, Any]) -> bool:
    if r.kind == "ratio":
        return float(f[r.field]) > float(r.value) / 100.0 * float(f[r.other])
    if r.kind == "above":
        return float(f[r.field]) > float(r.value)
    if r.kind == "below":
        return float(f[r.field]) < float(r.value)
    if r.kind == "false":
        return f[r.field] is False
    return str(f[r.field]) == str(r.value)  # equals


_RATIO = re.compile(r"^The ([\w ]+?) must not exceed ([\d.]+) percent of the ([\w ]+)$")
_CMP = re.compile(r"^\w+ with (?:a |an )?([\w ]+?) (above|below) (-?[\d.]+)(?: degrees)? "
                  r"should (?:also )?be ([\w ]+)$")
_FALSE = re.compile(r"^Items where (\w+) is false should be denied$")
_ZONE = re.compile(r"^Sites located in a protected zone should be flagged for review$")


def parse_policy(policy: str, family: str) -> list[Rule]:
    """Parse a v1 policy paragraph into :class:`Rule`\\ s. Header / 'Otherwise' sentences are
    ignored. Raises ``ValueError`` if no rule is recognised or a verb is outside the family."""
    ratio_verb = FAMILIES[family]["ratio_verb"]
    rules: list[Rule] = []
    for sent in re.split(r"(?<=\.) ", policy.strip()):
        s = sent.rstrip(".")
        if m := _RATIO.match(s):
            rules.append(Rule("ratio", _fld(m[1]), float(m[2]), _fld(m[3]), ratio_verb))
        elif m := _CMP.match(s):
            verb = PARTICIPLE_VERB[m[4]]
            rules.append(Rule(m[2], _fld(m[1]), float(m[3]), verb=verb))
        elif m := _FALSE.match(s):
            rules.append(Rule("false", m[1], verb="deny"))
        elif _ZONE.match(s):
            rules.append(Rule("equals", "zone_type", "protected", verb="flag"))
    if not rules:
        raise ValueError(f"no parseable rules in policy: {policy!r}")
    bad = {r.verb for r in rules} - set(FAMILIES[family]["verbs"])
    if bad:
        raise ValueError(f"rule verbs {sorted(bad)} outside family {family}")
    return rules


def _fld(text: str) -> str:
    return text.strip().replace(" ", "_")


def _family_of(vocab: Sequence[dict[str, Any]]) -> str | None:
    verbs = {v["verb"] for v in vocab}
    for name, fam in FAMILIES.items():
        if verbs == set(fam["verbs"]):
            return name
    return None


def load_specs(source: Path, subset: dict[tuple[str, str], int]) -> list[DomainSpec]:
    """Read v1 and build :class:`DomainSpec`\\ s for the first N domains of each subset group,
    in group order then domain-id order."""
    groups: dict[tuple[str, str], dict[str, DomainSpec]] = {g: {} for g in subset}
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        fam = _family_of(rec["vocabulary"])
        group = (rec["difficulty"]["structure"], fam or "")
        if group not in groups or rec["domain"] in groups[group]:
            continue
        policy = rec["prompt"].split("\n\n")[1]
        facts = _parse_facts(rec["prompt"])
        spec = DomainSpec(rec["domain"], group[0], group[1], policy, list(facts), facts)
        spec.rules = parse_policy(policy, group[1])
        groups[group][rec["domain"]] = spec
    out: list[DomainSpec] = []
    for group, n in subset.items():
        avail = [groups[group][k] for k in sorted(groups[group])]
        if len(avail) < n:
            raise ValueError(f"group {group}: need {n} domains, v1 has {len(avail)}")
        out.extend(avail[:n])
    return out


def _parse_facts(prompt: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for k, v in re.findall(r"^- (\w+): (\S+)$", prompt.split("Case facts:")[1], re.M):
        if v in ("True", "False"):
            facts[k] = v == "True"
        elif re.fullmatch(r"-?[\d.]+", v):
            facts[k] = float(v)
        else:
            facts[k] = v
    return facts


def interleave_limit(specs: list[DomainSpec], limit: int | None) -> list[DomainSpec]:
    """First ``limit`` domains, round-robin across (structure, family) groups for variety."""
    if limit is None or limit >= len(specs):
        return specs
    by_group: dict[tuple[str, str], list[DomainSpec]] = {}
    for s in specs:
        by_group.setdefault((s.structure, s.family), []).append(s)
    picked: list[DomainSpec] = []
    while len(picked) < limit:
        for lst in by_group.values():
            if lst and len(picked) < limit:
                picked.append(lst.pop(0))
    return picked


# --------------------------------------------------------------------------- synthesis

def _sample_facts(spec: DomainSpec, rng: random.Random) -> dict[str, Any]:
    ratio_num = {r.field: r for r in spec.rules if r.kind == "ratio"}
    ratio_den = {r.other for r in spec.rules if r.kind == "ratio"}
    thresholds = {r.field: float(r.value) for r in spec.rules if r.kind in ("above", "below")}
    facts: dict[str, Any] = {}
    for f in spec.fields:   # denominators first so numerators can reference them
        if f in ratio_den:
            facts[f] = round(rng.uniform(100, 300), 2)
    for f in spec.fields:
        if f in facts:
            continue
        sample = spec.sample_facts[f]
        if isinstance(sample, bool):
            facts[f] = rng.random() < 0.5
        elif f in CATEGORICAL:
            facts[f] = rng.choice(CATEGORICAL[f])
        elif f in ratio_num:
            r = ratio_num[f]
            facts[f] = round(facts[r.other] * rng.uniform(0, 2 * float(r.value) / 100), 2)
        elif f in thresholds:
            t = thresholds[f]
            hi = min(1.2 * t, 100.0) if "saturation" in f else 1.2 * t
            facts[f] = round(rng.uniform(0.8 * t, hi), 2)
        else:
            facts[f] = round(rng.uniform(0, 100), 2)
    return {f: facts[f] for f in spec.fields}   # v1 field order


def sample_balanced(spec: DomainSpec, n: int, rng: random.Random) -> list[dict[str, Any]]:
    """``n`` fact dicts, label-stratified round-robin (every reachable verdict represented
    as evenly as the sampler allows), then shuffled."""
    buckets: dict[str, list[dict[str, Any]]] = {v: [] for v in spec.verbs}
    for _ in range(max(40 * n, 2000)):
        f = _sample_facts(spec, rng)
        buckets[spec.label(f)].append(f)
    live = [b for b in buckets.values() if b]
    picked: list[dict[str, Any]] = []
    i = 0
    while len(picked) < n:
        b = live[i % len(live)]
        if b:
            picked.append(b.pop())
        i += 1
    rng.shuffle(picked)
    return picked


def write_authoring_csv(spec: DomainSpec, path: Path, rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sample_balanced(spec, CSV_ROWS, rng)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[*spec.fields, "decision"])
        w.writeheader()
        for f in rows:
            w.writerow({**f, "decision": spec.label(f)})


def _fmt(v: Any) -> str:
    return str(v)


def case_prompt(spec: DomainSpec, facts: dict[str, Any]) -> str:
    lines = "\n".join(f"- {k}: {_fmt(v)}" for k, v in facts.items())
    verbs = ", ".join(spec.verbs)
    return ("You are the decision-maker for the following policy domain.\n\n"
            f"{spec.policy}\n\nCase facts:\n{lines}\n\n"
            f"Choose exactly one action from: {verbs}.\nRespond with only the chosen action.")


def make_cases(spec: DomainSpec, rng: random.Random) -> list[Any]:
    from ambertrace_rlvr import EvalCase
    return [
        EvalCase(prompt=case_prompt(spec, f), facts=f, id=f"{spec.key}-p{i}",
                 domain=spec.key, query="Decide the case.",
                 difficulty={"structure": spec.structure})
        for i, f in enumerate(sample_balanced(spec, ITEMS_PER_DOMAIN, rng))
    ]


# --------------------------------------------------------------------------- backends

class DryRunBackend:
    """No network. Deterministic fake ids; a FakeVerifier labels cases with the local
    rule evaluator. ``calls`` counts every operation (tests assert no duplicate work)."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    @staticmethod
    def _n(key: str) -> int:
        return int(re.sub(r"\D", "", key) or 0)

    def find_domain(self, name: str) -> int | None:
        return None

    def create_domain(self, spec: DomainSpec) -> int:
        self.calls["create_domain"] += 1
        return 1000 + self._n(spec.key)

    def find_dataset(self, domain_id: int) -> int | None:
        return None

    def upload_dataset(self, domain_id: int, path: Path, name: str) -> int:
        self.calls["upload_dataset"] += 1
        assert path.exists()
        return 2000 + domain_id

    def set_decision_table(self, domain_id: int, table: dict[str, Any]) -> None:
        self.calls["set_decision_table"] += 1

    def start_ontology(self, domain_id: int) -> str:
        self.calls["start_ontology"] += 1
        return f"job-onto-{domain_id}"

    def wait_job(self, job_id: str) -> None:
        self.calls["wait_job"] += 1

    def find_platform(self, name: str) -> tuple[int, str] | None:
        return None

    def create_platform(self, spec: DomainSpec, domain_id: int,
                        dataset_id: int) -> tuple[int, str | None]:
        self.calls["create_platform"] += 1
        return 3000 + domain_id, f"job-plat-{domain_id}"

    def platform_status(self, platform_id: int) -> str:
        return "active"

    def build_quality(self, platform_id: int) -> str:
        self.calls["build_quality"] += 1
        return "ok"

    def vocabulary(self, platform_id: int, spec: DomainSpec) -> list[Any]:
        from ambertrace_rlvr import vocabulary_from_decision_table
        return vocabulary_from_decision_table(spec.table)

    def verifier(self, platform_id: int, spec: DomainSpec) -> Any:
        from ambertrace_rlvr.testing import FakeVerifier, make_report
        self.calls["verifier"] += 1

        def report_fn(parsed: Any) -> Any:
            return make_report(proof_checked=True, decision=spec.label(parsed.facts))

        return FakeVerifier(report_fn=report_fn)


class LiveBackend:
    """Real AmberTrace SDK calls, mirroring the verified reference pipeline. Never
    constructed by tests or ``--dry-run``."""

    def __init__(self) -> None:
        import ambertraceai
        self.api: Any = ambertraceai.AmbertraceAPI.from_env()

    def find_domain(self, name: str) -> int | None:
        d = next((d for d in self.api.domains.list() if d.get("name") == name), None)
        return d.get("id") if d else None

    def create_domain(self, spec: DomainSpec) -> int:
        d = self.api.domains.create(name=spec.name, description=spec.policy)
        return int(d.get("id"))

    def find_dataset(self, domain_id: int) -> int | None:
        d = next((d for d in self.api.datasets.list() if d.get("domain_id") == domain_id), None)
        return d.get("id") if d else None

    def upload_dataset(self, domain_id: int, path: Path, name: str) -> int:
        ds = self.api.datasets.upload(domain_id=domain_id, file_path=str(path), name=name,
                                      decision_column="decision")
        return int(ds.get("id"))

    def set_decision_table(self, domain_id: int, table: dict[str, Any]) -> None:
        self.api.domains.set_decision_table(domain_id, table)

    def start_ontology(self, domain_id: int) -> str:
        return str(self.api.domains.build_ontology(domain_id)["job_id"])

    def wait_job(self, job_id: str) -> None:
        self.api.wait_for_job(job_id)

    def find_platform(self, name: str) -> tuple[int, str] | None:
        p = next((p for p in self.api.platforms.list() if p.get("name") == name), None)
        return (int(p["id"]), str(p.get("status"))) if p else None

    def create_platform(self, spec: DomainSpec, domain_id: int,
                        dataset_id: int) -> tuple[int, str | None]:
        p = self.api.platforms.create(domain_id=domain_id, dataset_id=dataset_id,
                                      name=spec.platform_name, verified_profile=True,
                                      verified_min_confidence=0.85)
        return int(p.get("id")), p.get("job_id")

    def platform_status(self, platform_id: int) -> str:
        return str(self.api.platforms.status(platform_id).get("status"))

    def build_quality(self, platform_id: int) -> str:
        bq = self.api.platforms.build_quality(platform_id)
        return str(bq.get("status") if hasattr(bq, "get") else bq)

    def vocabulary(self, platform_id: int, spec: DomainSpec) -> list[Any]:
        from ambertrace_rlvr import decision_table_for_platform, vocabulary_from_decision_table
        return vocabulary_from_decision_table(decision_table_for_platform(self.api, platform_id))

    def verifier(self, platform_id: int, spec: DomainSpec) -> Any:
        from ambertrace_rlvr import AmberVerifier, JSONBlockParser, VerifiableDomain
        domain = VerifiableDomain.from_env(platform_id=platform_id, parser=JSONBlockParser())
        return AmberVerifier(domain=domain)


# --------------------------------------------------------------------------- manifest

class Manifest:
    """JSON manifest, rewritten atomically after every state transition."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"version": 1, "domains": {}}
        if path.exists():
            self.data = json.loads(path.read_text())
            self.data.setdefault("domains", {})

    def entry(self, key: str) -> dict[str, Any]:
        e: dict[str, Any] = self.data["domains"].setdefault(key, {})
        return e

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)


def _step_idx(entry: dict[str, Any]) -> int:
    s = entry.get("step")
    return STEPS.index(s) if s in STEPS else -1


# --------------------------------------------------------------------------- pipeline

def author_domain(spec: DomainSpec, backend: Any, manifest: Manifest, workdir: Path,
                  log: Callable[[str], None] = print) -> None:
    """Advance one domain to ``items_generated``, resuming from its last completed step."""
    e = manifest.entry(spec.key)
    items_path = workdir / "items" / f"{spec.key}.jsonl"
    if e.get("step") == "items_generated" and not items_path.exists():
        e["step"] = "quality_ok"   # items cache lost: regenerate (oracle re-query only)
    if _step_idx(e) >= STEPS.index("items_generated"):
        return
    e.update(name=spec.name, structure=spec.structure, family=spec.family)
    e.pop("error", None)

    def done(step: str) -> None:
        e["step"] = step
        manifest.save()
        log(f"  [{spec.key}] {step}")

    if _step_idx(e) < 0:                                      # -> created
        if e.get("domain_id") is None:
            e["domain_id"] = backend.find_domain(spec.name) or backend.create_domain(spec)
        done("created")
    domain_id = e["domain_id"]

    if _step_idx(e) < STEPS.index("dataset_uploaded"):
        if e.get("dataset_id") is None:
            ds = backend.find_dataset(domain_id)
            if ds is None:
                csv_path = workdir / "csv" / f"{spec.key}.csv"
                write_authoring_csv(spec, csv_path, random.Random(f"{SEED}:csv:{spec.key}"))
                ds = backend.upload_dataset(domain_id, csv_path, f"{spec.key}_authoring.csv")
            e["dataset_id"] = ds
        done("dataset_uploaded")

    if _step_idx(e) < STEPS.index("table_set"):
        backend.set_decision_table(domain_id, spec.table)
        done("table_set")

    if _step_idx(e) < STEPS.index("ontology_built"):
        if e.get("ontology_job_id") is None:                  # persist job BEFORE waiting
            e["ontology_job_id"] = backend.start_ontology(domain_id)
            manifest.save()
        backend.wait_job(e["ontology_job_id"])                # resume = re-poll same job
        done("ontology_built")

    if _step_idx(e) < STEPS.index("platform_active"):
        if e.get("platform_id") is None:
            found = backend.find_platform(spec.platform_name)
            if found is not None:
                e["platform_id"] = found[0]
            else:
                pid, job = backend.create_platform(spec, domain_id, e["dataset_id"])
                e["platform_id"], e["platform_job_id"] = pid, job
            manifest.save()                                   # persist ids BEFORE waiting
        if e.get("platform_job_id"):
            backend.wait_job(e["platform_job_id"])
        status = backend.platform_status(e["platform_id"])
        if status != "active":
            raise RuntimeError(f"platform {e['platform_id']} status={status!r}, not active")
        done("platform_active")
    platform_id = e["platform_id"]

    if _step_idx(e) < STEPS.index("quality_ok"):
        e["quality_status"] = backend.build_quality(platform_id)
        done("quality_ok")

    from ambertrace_rlvr import build_eval_items, write_decision_corpus
    vocab = backend.vocabulary(platform_id, spec)
    cases = make_cases(spec, random.Random(f"{SEED}:cases:{spec.key}"))
    items = build_eval_items(backend.verifier(platform_id, spec), cases, vocab)
    facts_by_id = {c.id: c.facts for c in cases}
    agree = sum(1 for it in items if it.oracle == spec.label(facts_by_id[it.id]))
    write_decision_corpus(items_path, items)
    e["n_items"] = len(items)
    e["oracle_agreement"] = f"{agree}/{len(items)}"
    if len(items) < ITEMS_PER_DOMAIN or agree < len(items):
        log(f"  [{spec.key}] WARNING items={len(items)}/{ITEMS_PER_DOMAIN} "
            f"oracle==intended rule {agree}/{len(items)}")
    done("items_generated")


def assemble(specs: Sequence[DomainSpec], manifest: Manifest, workdir: Path, out: Path) -> int:
    """Concatenate every completed domain's items into ``out``. Returns items written."""
    from ambertrace_rlvr import load_decision_corpus, write_decision_corpus
    items: list[Any] = []
    for s in specs:
        items_path = workdir / "items" / f"{s.key}.jsonl"
        if manifest.entry(s.key).get("step") == "items_generated" and items_path.exists():
            items.extend(load_decision_corpus(items_path))
    write_decision_corpus(out, items)
    return len(items)


def run_pipeline(specs: Sequence[DomainSpec], backend: Any, workdir: Path, out: Path,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    manifest = Manifest(workdir / "authoring_manifest.json")
    for i, spec in enumerate(specs):
        items_path = workdir / "items" / f"{spec.key}.jsonl"
        if manifest.entry(spec.key).get("step") == "items_generated" and items_path.exists():
            log(f"[{i + 1}/{len(specs)}] {spec.key}: already items_generated, skipping")
            continue
        log(f"[{i + 1}/{len(specs)}] {spec.key} ({spec.structure}/{spec.family})")
        try:
            author_domain(spec, backend, manifest, workdir, log)
        except Exception as exc:  # noqa: BLE001 - record and continue; resume picks it up
            manifest.entry(spec.key)["error"] = f"{type(exc).__name__}: {exc}"
            manifest.save()
            log(f"  [{spec.key}] ERROR {exc}")
    n_items = assemble(specs, manifest, workdir, out)
    incomplete = [s.key for s in specs if manifest.entry(s.key).get("step") != "items_generated"]
    return {"domains": len(specs), "complete": len(specs) - len(incomplete),
            "incomplete": incomplete, "items": n_items, "out": str(out)}


def validate_only(platform_id: int, backend: Any | None = None) -> list[Any]:
    """Print the LabelSpecs derived from an existing platform's decision table."""
    from ambertrace_rlvr import decision_table_for_platform, vocabulary_from_decision_table
    api = (backend or LiveBackend()).api
    table = decision_table_for_platform(api, platform_id)
    vocab = vocabulary_from_decision_table(table)
    print(f"platform {platform_id} decision table: {json.dumps(table)}")
    for v in vocab:
        print(f"  verb={v.verb} rank={v.rank} "
              f"{'restrictive' if v.restrictive else 'permits'}")
    return vocab


def _load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="FakeVerifier + fake ids; no network (scratch workdir/out)")
    ap.add_argument("--validate-only", type=int, metavar="PLATFORM_ID",
                    help="print LabelSpecs from an existing platform's decision table")
    ap.add_argument("--subset", choices=sorted(SUBSETS), default="full4")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N domains (round-robin across groups)")
    ap.add_argument("--source", type=Path, default=REPO / "data" / "decision_eval_v1.jsonl")
    ap.add_argument("--workdir", type=Path, default=None,
                    help="manifest/csv/items cache (default outputs/decision_eval_v2, "
                         "or a separate *_dryrun dir with --dry-run)")
    ap.add_argument("--out", type=Path, default=None,
                    help="corpus path (default data/decision_eval_v2_subset.jsonl; "
                         "dry-run: <workdir>/corpus_dryrun.jsonl)")
    args = ap.parse_args(argv)

    if args.validate_only is not None:
        _load_env()
        validate_only(args.validate_only)
        return 0

    workdir = args.workdir or REPO / "outputs" / (
        "decision_eval_v2_dryrun" if args.dry_run else "decision_eval_v2")
    out = args.out or (workdir / "corpus_dryrun.jsonl" if args.dry_run
                       else REPO / "data" / "decision_eval_v2_subset.jsonl")
    specs = interleave_limit(load_specs(args.source, SUBSETS[args.subset]), args.limit)
    if args.dry_run:
        backend: Any = DryRunBackend()
    else:
        _load_env()
        backend = LiveBackend()
    res = run_pipeline(specs, backend, workdir, out)
    print(json.dumps(res, indent=2))
    return 0 if not res["incomplete"] else 1


if __name__ == "__main__":
    sys.exit(main())
