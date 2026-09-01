"""Generate GOLD-anchored training/eval prompts for the ACMG demo, with a GENUINE
held-out eval.

Each record is a chat-format prompt (system format-contract + a natural-language
variant scenario) plus a ``gold`` ACMG label. The reward combines the platform's
proof certificate with correctness against this gold label.

Gold anchoring is used here (rather than the label-free path) because 'uncertain' is
an evidence-poor residual: a label-free reward could let the policy drift toward the
easy class, whereas the curated ACMG label forces it to get pathogenic and benign
right too. (Spec §8: "gold anchoring where available".)

HELD-OUT SPLIT — the whole point of this generator: train and eval are split at the
level of the underlying **feature combination** (the six-criterion evidence tuple),
BEFORE it is expanded into surface phrasings. So no combo — and therefore no
(inputs, gold) pair — ever appears on both sides. An earlier version split after the
phrasing expansion, which put paraphrases of the *same* combo (identical gold) in
both splits, leaving the "eval" set 100% memorised. That is fixed here and locked by
tests/test_acmg_dataset.py.

The domain has THREE pathogenic and THREE benign criteria (see
author_acmg_platform.py), so pathogenic and benign each have seven distinct
evidence combinations — enough to hold some out per class while still training on
the others. The eval combos are therefore novel *combinations* of criteria the
policy never saw, so scoring on them measures whether it learned the rule
("any pathogenic criterion and no benign evidence → pathogenic"), not whether it
memorised specific inputs.

    python examples/gen_acmg_prompts.py   # writes data/acmg_{train,eval}.jsonl
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from ambertrace_rlvr.prompts import build_system_prompt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SYSTEM = build_system_prompt("ACMG Variant Classification",
                             answer_key="classification", facts_key="facts")

# Feature order: three pathogenic criteria then three benign criteria.
PATHOGENIC_FIELDS = [
    "null_variant_in_disease_gene",      # PVS1
    "functional_studies_damaging",       # PS3
    "computational_predicts_damaging",   # PP3
]
BENIGN_FIELDS = [
    "common_in_population",              # BA1
    "functional_studies_benign",        # BS3
    "computational_predicts_benign",    # BP4
]
FIELDS = PATHOGENIC_FIELDS + BENIGN_FIELDS

# Held-out split sizes, per class, at the COMBINATION level. pathogenic/benign each
# have exactly 7 combos, so 5 + 2 uses all of them disjointly; uncertain has many, so
# we take a matching, balanced slice.
N_TRAIN_PER_CLASS = 5
N_EVAL_PER_CLASS = 2
PHRASINGS = 3


def gold_label(bits: tuple[bool, ...]) -> str:
    """The simplified ACMG rule (see author_acmg_platform.py): pathogenic if any
    pathogenic criterion fires and no benign one does; benign if the mirror holds;
    'uncertain' when there is no evidence either way, or the evidence conflicts."""
    pathogenic_evidence = any(bits[:len(PATHOGENIC_FIELDS)])
    benign_evidence = any(bits[len(PATHOGENIC_FIELDS):])
    if pathogenic_evidence and not benign_evidence:
        return "pathogenic"
    if benign_evidence and not pathogenic_evidence:
        return "benign"
    return "uncertain"


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def _functional_studies(ps3: bool, bs3: bool) -> str:
    """Prose for the PS3/BS3 functional-study evidence. Must convey BOTH criteria
    independently: a variant can carry a damaging (PS3) *and* a benign (BS3)
    functional study at once — a conflicting case that is precisely what makes it
    'uncertain'."""
    if ps3 and bs3:
        return ("show conflicting results — one well-established study indicates a "
                "damaging effect while another indicates no damaging effect")
    if ps3:
        return "show a damaging effect"
    if bs3:
        return "show no damaging effect"
    return "are unavailable"


def _computational(pp3: bool, bp4: bool) -> str:
    """Prose for the PP3/BP4 in-silico evidence, again conveying both criteria
    independently (they can conflict)."""
    if pp3 and bp4:
        return ("disagree — some tools predict a damaging effect while others "
                "predict no impact")
    if pp3:
        return "predict a damaging effect"
    if bp4:
        return "predict no damaging effect"
    return "were not run"


def _scenario(bits: tuple[bool, ...], i: int) -> str:
    pvs1, ps3, pp3, ba1, bs3, bp4 = bits
    pvs1_s = ("a predicted loss-of-function variant in a disease gene" if pvs1
              else "not a loss-of-function variant")
    func_s = _functional_studies(ps3, bs3)
    comp_s = _computational(pp3, bp4)
    freq_s = "common in the general population" if ba1 else "rare in the population"
    templates = (
        (f"A sequence variant is {pvs1_s}. Functional studies {func_s}. "
         f"In-silico tools {comp_s}. The variant is {freq_s}. "
         f"Classify it as 'pathogenic', 'benign', or 'uncertain'."),
        (f"Variant evidence — LoF in disease gene: {_yn(pvs1)}; damaging functional "
         f"study: {_yn(ps3)}; computational tools predict damaging: {_yn(pp3)}; "
         f"common in population: {_yn(ba1)}; benign functional study: {_yn(bs3)}; "
         f"computational tools predict benign: {_yn(bp4)}. "
         f"Classify the variant (pathogenic / benign / uncertain)."),
        (f"Interpret this variant under ACMG criteria. PVS1 (LoF in disease gene): "
         f"{_yn(pvs1)}. PS3 (damaging functional study): {_yn(ps3)}. PP3 (computational "
         f"damaging): {_yn(pp3)}. BA1 (common in population): {_yn(ba1)}. BS3 (benign "
         f"functional study): {_yn(bs3)}. BP4 (computational benign): {_yn(bp4)}. "
         f"Is it pathogenic, benign, or uncertain?"),
    )
    return templates[i % len(templates)]


def _record(bits: tuple[bool, ...], i: int) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _scenario(bits, i)},
        ],
        "gold": gold_label(bits),
    }


def split_combos() -> tuple[list[tuple[bool, ...]], list[tuple[bool, ...]]]:
    """Partition the feature combinations into DISJOINT train/eval combo sets,
    balanced across the three classes. Deterministic (sorted) so the split is
    reproducible; the two returned sets never share a combo."""
    combos = sorted(product((True, False), repeat=len(FIELDS)))
    by_class: dict[str, list[tuple[bool, ...]]] = {
        "pathogenic": [], "benign": [], "uncertain": [],
    }
    for c in combos:
        by_class[gold_label(c)].append(c)

    train: list[tuple[bool, ...]] = []
    eval_: list[tuple[bool, ...]] = []
    for cls in ("pathogenic", "benign", "uncertain"):
        pool = by_class[cls]
        need = N_TRAIN_PER_CLASS + N_EVAL_PER_CLASS
        assert len(pool) >= need, (
            f"class {cls!r} has only {len(pool)} combos, need {need} for a "
            f"disjoint {N_TRAIN_PER_CLASS}/{N_EVAL_PER_CLASS} split"
        )
        train.extend(pool[:N_TRAIN_PER_CLASS])
        eval_.extend(pool[N_TRAIN_PER_CLASS:need])
    return train, eval_


def _records_for(combos: list[tuple[bool, ...]], start_i: int = 0) -> list[dict]:
    records, i = [], start_i
    for combo in combos:
        for _ in range(PHRASINGS):
            records.append(_record(combo, i))
            i += 1
    return records


def _write(path: Path, records: list) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    counts = {c: sum(r["gold"] == c for r in records)
              for c in ("pathogenic", "benign", "uncertain")}
    print(f"wrote {len(records)} prompts to {path} ({counts})")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    train_combos, eval_combos = split_combos()
    # Distinct phrasing-index offsets so train and eval don't lock-step to the same
    # template ordering (purely cosmetic; the split is already disjoint by combo).
    train_records = _records_for(train_combos, start_i=0)
    eval_records = _records_for(eval_combos, start_i=1)
    _write(DATA / "acmg_train.jsonl", train_records)
    _write(DATA / "acmg_eval.jsonl", eval_records)
    overlap = set(train_combos) & set(eval_combos)
    print(f"held-out check: {len(train_combos)} train combos, {len(eval_combos)} "
          f"eval combos, overlap={len(overlap)}")


if __name__ == "__main__":
    main()
