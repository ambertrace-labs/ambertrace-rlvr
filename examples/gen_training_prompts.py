"""Generate LABEL-FREE training/eval prompts for the Grant Eligibility demo.

Each record is a chat-format prompt (system format-contract + a natural-language
applicant scenario). NO gold label is stored — the reward comes from AmberTrace's
certificate (proof + correctness vs the platform's own certified decision), so
the model learns the rules from the verifier, unsupervised.

    python examples/gen_training_prompts.py   # writes data/grant_eligibility_{train,eval}.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from ambertrace_rlvr.prompts import build_system_prompt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SYSTEM = build_system_prompt("Grant Eligibility", answer_key="classification", facts_key="facts")

# Feature grid (mirrors the platform's dataset domain), rendered to prose.
AGES = (16, 17, 18, 21, 34, 40, 55, 67)
INCOMES = (9000, 18000, 25000, 29000, 30000, 32000, 45000, 90000)
BOOLS = (True, False)


def _scenario(age: int, income: int, resident: bool, active: bool, i: int) -> str:
    res = "is a resident" if resident else "is not a resident"
    grant = "already holds an active grant" if active else "holds no active grant"
    templates = (
        f"An applicant is {age} years old, earns {income} per year, {res}, and {grant}. "
        f"Assess their eligibility for the support grant.",
        f"Applicant profile — age: {age}; annual income: {income}; resident: "
        f"{'yes' if resident else 'no'}; active grant: {'yes' if active else 'no'}. "
        f"Should this application be permitted or denied?",
        f"Please decide the grant application. The applicant is aged {age} with an "
        f"annual income of {income}. They {res} and {grant}.",
    )
    return templates[i % len(templates)]


def _record(age, income, resident, active, i):
    return {"prompt": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _scenario(age, income, resident, active, i)},
    ]}


def build():
    records, i = [], 0
    for age in AGES:
        for income in INCOMES:
            for resident in BOOLS:
                for active in BOOLS:
                    records.append(_record(age, income, resident, active, i))
                    i += 1
    return records


def _write(path: Path, records: list) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} prompts to {path}")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    records = build()
    # deterministic split: every 5th row to eval.
    eval_set = [r for j, r in enumerate(records) if j % 5 == 0]
    train_set = [r for j, r in enumerate(records) if j % 5 != 0]
    _write(DATA / "grant_eligibility_train.jsonl", train_set)
    _write(DATA / "grant_eligibility_eval.jsonl", eval_set)


if __name__ == "__main__":
    main()
