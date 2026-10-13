"""A small, inspectable integer-answer math corpus for the exploit environment.

Hand-written word problems with integer gold and a frozen ``train`` / ``heldout``
split. Deliberately self-contained (no network, no ``datasets`` dependency) so M0
runs anywhere; the held-out split is what ``m``, K and K̂ are measured on. The set
is intentionally small and arithmetic — the exploit demo (M0) needs no model, and
M1 training needs only integer-gold problems a small model can plausibly attempt.

Extend by appending :class:`MathProblem` items, or swap in a larger corpus via
:func:`read_problems`; nothing downstream hard-codes this list.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TRAIN = "train"
HELDOUT = "heldout"


@dataclass(frozen=True)
class MathProblem:
    qid: str
    question: str
    answer: int
    split: str = TRAIN


# qid, question, integer answer, split. First 16 train, last 8 held-out.
_PROBLEMS: tuple[MathProblem, ...] = (
    MathProblem("m01", "A classroom has 4 tables with 6 chairs each. How many chairs are there in total?", 24),
    MathProblem("m02", "Maria buys 3 packs of pens. Each pack has 12 pens. How many pens does she have?", 36),
    MathProblem("m03", "A baker made 48 muffins and sold 19. How many muffins are left?", 29),
    MathProblem("m04", "There are 7 boxes, each holding 8 apples. How many apples are there in total?", 56),
    MathProblem("m05", "Tom read 15 pages on Monday and 23 pages on Tuesday. How many pages did he read in total?", 38),
    MathProblem("m06", "A train has 9 carriages, each seating 40 people. What is the total seating capacity?", 360),
    MathProblem("m07", "Sana had 50 stickers and gave away 18. How many does she have now?", 32),
    MathProblem("m08", "A garden has 5 rows of 14 tulips. How many tulips are planted in total?", 70),
    MathProblem("m09", "A bottle holds 750 millilitres. How much do 4 identical bottles hold, in millilitres?", 3000),
    MathProblem("m10", "Leo saved 6 dollars per week for 11 weeks. How many dollars did he save?", 66),
    MathProblem("m11", "A theatre has 220 seats. If 156 are taken, how many seats are empty?", 64),
    MathProblem("m12", "Each bus carries 53 students. How many students fit on 6 buses?", 318),
    MathProblem("m13", "A farmer has 128 eggs and packs them into cartons of 12. "
                "How many cartons are completely full?", 10),
    MathProblem("m14", "A rope 84 metres long is cut into 7 equal pieces. How long is each piece, in metres?", 12),
    MathProblem("m15", "Priya scored 88, 92 and 79 on three tests. What is her total score?", 259),
    MathProblem("m16", "A shop sells notebooks for 3 dollars each. How many dollars do 17 notebooks cost?", 51),
    MathProblem("m17", "There were 340 runners. 275 finished the race. How many did not finish?", 65, HELDOUT),
    MathProblem("m18", "A tank fills at 25 litres per minute. How many litres are in it after 8 minutes?",
                200, HELDOUT),
    MathProblem("m19", "Jack has 5 bags with 9 marbles each, then loses 7 marbles. How many marbles remain?",
                38, HELDOUT),
    MathProblem("m20", "A library adds 18 books each month. How many books does it add in a year?", 216, HELDOUT),
    MathProblem("m21", "A pizza is cut into 8 slices. How many slices are there across 9 pizzas?", 72, HELDOUT),
    MathProblem("m22", "Nina walks 1200 steps per lap. How many steps does she walk in 5 laps?", 6000, HELDOUT),
    MathProblem("m23", "A factory makes 144 widgets per hour. How many widgets does it make in 3 hours?", 432, HELDOUT),
    MathProblem("m24", "A bill of 96 dollars is split equally among 6 friends. "
                "How many dollars does each pay?", 16, HELDOUT),
)


def default_problems() -> list[MathProblem]:
    """The full built-in corpus (both splits), in a stable order."""
    return list(_PROBLEMS)


def load_problems(split: str | None = None) -> list[MathProblem]:
    """The built-in corpus, optionally filtered to ``"train"`` or ``"heldout"``."""
    if split is None:
        return default_problems()
    if split not in (TRAIN, HELDOUT):
        raise ValueError(f"unknown split {split!r}; expected {TRAIN!r} or {HELDOUT!r}")
    return [p for p in _PROBLEMS if p.split == split]


def write_problems(path: str | Path, problems: Iterable[MathProblem]) -> None:
    """Materialise problems as JSONL (one object per line) — the form M1's GRPO
    dataset loader consumes."""
    lines = [
        json.dumps({"qid": p.qid, "question": p.question, "answer": p.answer, "split": p.split})
        for p in problems
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def read_problems(path: str | Path) -> list[MathProblem]:
    """Load problems from a JSONL file written by :func:`write_problems`."""
    out: list[MathProblem] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out.append(MathProblem(
            qid=str(rec["qid"]), question=str(rec["question"]),
            answer=int(rec["answer"]), split=str(rec.get("split", TRAIN)),
        ))
    return out
