"""Generate GOLD-anchored training/eval prompts for the ACMG demo.

Each record is a chat-format prompt (system format-contract + a natural-language
variant scenario) plus a ``gold`` ACMG label. The reward combines the platform's
proof certificate with correctness against this gold label.

Gold anchoring is used here (rather than the label-free path) because 'uncertain' is
an evidence-poor residual: a label-free reward could let the policy drift toward the
easy class, whereas the curated ACMG label forces it to get pathogenic and benign
right too. (Spec §8: "gold anchoring where available".)

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

# (null_variant_in_disease_gene, functional_studies_damaging,
#  common_in_population, functional_studies_benign)
FIELDS = ["null_variant_in_disease_gene", "functional_studies_damaging",
          "common_in_population", "functional_studies_benign"]


def gold_label(pvs1: bool, ps3: bool, ba1: bool, bs3: bool) -> str:
    """The simplified ACMG rule (see author_acmg_platform.py)."""
    pathogenic_evidence = pvs1 or ps3
    benign_evidence = ba1 or bs3
    if pathogenic_evidence and not benign_evidence:
        return "pathogenic"
    if benign_evidence and not pathogenic_evidence:
        return "benign"
    return "uncertain"


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def _scenario(pvs1, ps3, ba1, bs3, i: int) -> str:
    pvs1_s = ("a predicted loss-of-function variant in a disease gene" if pvs1
              else "not a loss-of-function variant")
    ps3_s = ("show a damaging effect" if ps3 else
             ("show no damaging effect" if bs3 else "are unavailable"))
    common_s = "common in the general population" if ba1 else "rare in the population"
    templates = (
        (f"A sequence variant is {pvs1_s}. Functional studies {ps3_s}. "
         f"The variant is {common_s}. "
         f"Classify it as 'pathogenic', 'benign', or 'uncertain'."),
        (f"Variant evidence — LoF in disease gene: {_yn(pvs1)}; damaging functional "
         f"study: {_yn(ps3)}; common in population: {_yn(ba1)}; benign functional "
         f"study: {_yn(bs3)}. Classify the variant (pathogenic / benign / uncertain)."),
        (f"Interpret this variant under ACMG criteria. PVS1 (LoF in disease gene): "
         f"{_yn(pvs1)}. PS3 (damaging functional study): {_yn(ps3)}. BA1 (common in "
         f"population): {_yn(ba1)}. BS3 (benign functional study): {_yn(bs3)}. "
         f"Is it pathogenic, benign, or uncertain?"),
    )
    return templates[i % len(templates)]


def _record(pvs1, ps3, ba1, bs3, i):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _scenario(pvs1, ps3, ba1, bs3, i)},
        ],
        "gold": gold_label(pvs1, ps3, ba1, bs3),
    }


def build() -> list[dict]:
    combos = list(product((True, False), repeat=4))
    by_class: dict[str, list] = {"pathogenic": [], "benign": [], "uncertain": []}
    for c in combos:
        by_class[gold_label(*c)].append(c)
    # Balance the three classes so the run can't collapse to a single label.
    n = min(len(v) for v in by_class.values())
    records, i = [], 0
    for cls in ("pathogenic", "benign", "uncertain"):
        for combo in by_class[cls][:n]:
            for _ in range(3):     # 3 phrasings each
                records.append(_record(*combo, i))
                i += 1
    return records


def _write(path: Path, records: list) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    counts = {c: sum(r["gold"] == c for r in records) for c in ("pathogenic", "benign", "uncertain")}
    print(f"wrote {len(records)} prompts to {path} ({counts})")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    records = build()
    eval_set = [r for j, r in enumerate(records) if j % 5 == 0]
    train_set = [r for j, r in enumerate(records) if j % 5 != 0]
    _write(DATA / "acmg_train.jsonl", train_set)
    _write(DATA / "acmg_eval.jsonl", eval_set)


if __name__ == "__main__":
    main()
